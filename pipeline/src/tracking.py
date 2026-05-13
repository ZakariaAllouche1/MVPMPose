from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.cameras import CameraData
from src.detection import ViewDetection


BoxXYXY = Tuple[int, int, int, int]

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}

large_cost = 1e5


class TrackState(Enum):
    tentative = 1
    confirmed_2d = 2
    confirmed_3d = 3
    occluded = 4
    lost = 5


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def make_spd(matrix: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    matrix = symmetrize(matrix)

    try:
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, min_eig)
        return (vectors * values) @ vectors.T
    except np.linalg.LinAlgError:
        return symmetrize(matrix + min_eig * np.eye(matrix.shape[0], dtype=matrix.dtype))


def bbox_to_kalman(bbox: BoxXYXY) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox)

    width = max(1e-3, x2 - x1)
    height = max(1e-3, y2 - y1)

    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    aspect = width / height

    return np.array([cx, cy, aspect, height], dtype=np.float64)


def kalman_to_bbox(mean: np.ndarray) -> np.ndarray:
    cx, cy, aspect, height = mean[:4]
    width = aspect * height

    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0

    return np.array([x1, y1, x2, y2], dtype=np.float64)


def bbox_center(bbox: BoxXYXY) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array(
        [
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
        ],
        dtype=np.float64,
    )


def bbox_size(bbox: BoxXYXY) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return float(x2 - x1), float(y2 - y1)


def bbox_area(bbox: BoxXYXY) -> float:
    width, height = bbox_size(bbox)
    return float(max(0.0, width) * max(0.0, height))


def bbox_iou(a: BoxXYXY, b: BoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    if inter <= 0:
        return 0.0

    union = bbox_area(a) + bbox_area(b) - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def project_points(points_3d: np.ndarray, projection: np.ndarray) -> np.ndarray:
    points_3d = np.asarray(points_3d, dtype=np.float64)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d doit avoir shape [N, 3], reçu: {points_3d.shape}")

    points_h = np.hstack(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)]
    )

    projected_h = (projection @ points_h.T).T
    z = projected_h[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)

    return projected_h[:, :2] / z


def weighted_triangulate_point(
    projections: List[np.ndarray],
    points_2d: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    if len(projections) != len(points_2d):
        raise ValueError("projections et points_2d doivent avoir la même longueur")

    if len(projections) < 2:
        raise ValueError("Il faut au moins deux vues pour trianguler")

    if weights is None:
        weights = [1.0] * len(projections)

    matrix = np.zeros((2 * len(projections), 4), dtype=np.float64)

    for index, (projection, point, weight) in enumerate(zip(projections, points_2d, weights)):
        x, y = float(point[0]), float(point[1])
        w = float(weight)

        matrix[2 * index] = w * (x * projection[2] - projection[0])
        matrix[2 * index + 1] = w * (y * projection[2] - projection[1])

    _, _, vh = np.linalg.svd(matrix)
    point_h = vh[-1]

    if abs(point_h[3]) < 1e-12:
        raise ValueError("Triangulation instable")

    return (point_h[:3] / point_h[3]).astype(np.float32)


def check_cheirality(point_3d: np.ndarray, camera: CameraData) -> bool:
    point = np.asarray(point_3d, dtype=np.float64).reshape(3, 1)
    cam_point = camera.R @ point + camera.t
    return bool(cam_point[2, 0] > 0)


class KalmanFilter2D:
    def __init__(self) -> None:
        ndim = 4
        dt = 1.0

        self.motion_mat = np.eye(2 * ndim, 2 * ndim)
        self.update_mat = np.eye(ndim, 2 * ndim)

        for index in range(ndim):
            self.motion_mat[index, ndim + index] = dt

        self.std_weight_position = 1.0 / 20.0
        self.std_weight_velocity = 1.0 / 80.0

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement.astype(np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        height = max(1e-3, measurement[3])

        std = [
            2 * self.std_weight_position * height,
            2 * self.std_weight_position * height,
            1e-2,
            2 * self.std_weight_position * height,
            10 * self.std_weight_velocity * height,
            10 * self.std_weight_velocity * height,
            1e-5,
            10 * self.std_weight_velocity * height,
        ]

        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height = max(1e-3, mean[3])

        std_pos = [
            self.std_weight_position * height,
            self.std_weight_position * height,
            1e-2,
            self.std_weight_position * height,
        ]

        std_vel = [
            self.std_weight_velocity * height,
            self.std_weight_velocity * height,
            1e-5,
            self.std_weight_velocity * height,
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov

        return mean, make_spd(covariance, 1e-9)

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height = max(1e-3, mean[3])

        std = [
            self.std_weight_position * height,
            self.std_weight_position * height,
            1e-1,
            self.std_weight_position * height,
        ]

        innovation_cov = np.diag(np.square(std))

        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T

        return projected_mean, make_spd(projected_cov + innovation_cov, 1e-9)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.all(np.isfinite(measurement)):
            return mean, covariance

        projected_mean, projected_cov = self.project(mean, covariance)
        projected_cov = make_spd(projected_cov, 1e-9)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov,
            lower=True,
            check_finite=False,
        )

        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self.update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean, make_spd(new_covariance, 1e-9)

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        try:
            cholesky = np.linalg.cholesky(projected_cov)
            diff = measurements - projected_mean

            z = scipy.linalg.solve_triangular(
                cholesky,
                diff.T,
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )

            return np.sum(z * z, axis=0)

        except np.linalg.LinAlgError:
            return np.full(measurements.shape[0], large_cost, dtype=np.float64)


class KalmanFilter3D:
    def __init__(self) -> None:
        ndim = 3
        dt = 1.0

        self.motion_mat = np.eye(2 * ndim, 2 * ndim)
        self.update_mat = np.eye(ndim, 2 * ndim)

        for index in range(ndim):
            self.motion_mat[index, ndim + index] = dt

        self.std_weight_position = 1.0 / 25.0
        self.std_weight_velocity = 1.0 / 150.0

    def initiate(self, position_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = np.asarray(position_3d, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [2 * self.std_weight_position * 1000.0] * 3
        std += [10 * self.std_weight_velocity * 1000.0] * 3

        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [self.std_weight_position * 100.0] * 3
        std_vel = [self.std_weight_velocity * 100.0] * 3

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov

        return mean, make_spd(covariance, 1e-9)

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        std = [self.std_weight_position * 200.0] * 3
        innovation_cov = np.diag(np.square(std))

        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T

        return projected_mean, make_spd(projected_cov + innovation_cov, 1e-9)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        position_3d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        projected_cov = make_spd(projected_cov, 1e-9)

        chol, lower = scipy.linalg.cho_factor(
            projected_cov,
            lower=True,
            check_finite=False,
        )

        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower),
            (covariance @ self.update_mat.T).T,
            check_finite=False,
        ).T

        innovation = np.asarray(position_3d, dtype=np.float64) - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean, make_spd(new_covariance, 1e-9)

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements_3d: np.ndarray,
    ) -> np.ndarray:
        projected_mean, projected_cov = self.project(mean, covariance)

        try:
            cholesky = np.linalg.cholesky(projected_cov)
            diff = measurements_3d - projected_mean

            z = scipy.linalg.solve_triangular(
                cholesky,
                diff.T,
                lower=True,
                check_finite=False,
                overwrite_b=True,
            )

            return np.sum(z * z, axis=0)

        except np.linalg.LinAlgError:
            return np.full(measurements_3d.shape[0], large_cost, dtype=np.float64)


@dataclass
class DetectionGroup:
    detections: List[ViewDetection]

    @property
    def camera_names(self) -> List[str]:
        return [detection.camera_name for detection in self.detections]

    @property
    def camera_set(self) -> set[str]:
        return set(self.camera_names)

    @property
    def num_views(self) -> int:
        return len(self.camera_set)

    @property
    def mean_confidence(self) -> float:
        if not self.detections:
            return 0.0
        return float(np.mean([detection.confidence for detection in self.detections]))

    @property
    def visibility_ratio(self) -> float:
        ratios = []

        for detection in self.detections:
            if detection.joint_confidences is None:
                continue
            ratios.append(float(np.mean(detection.joint_confidences > 0.25)))

        if not ratios:
            return 0.0

        return float(np.mean(ratios))

    def features(self) -> List[np.ndarray]:
        return [
            detection.feature
            for detection in self.detections
            if detection.feature is not None
        ]

    def mean_feature(self) -> Optional[np.ndarray]:
        features = self.features()

        if not features:
            return None

        feature = np.mean(np.stack(features, axis=0), axis=0)
        norm = np.linalg.norm(feature)

        if norm <= 1e-12:
            return None

        return feature / norm


@dataclass
class GlobalTrack:
    global_id: int
    first_seen_frame: int
    last_seen_frame: int

    state: TrackState = TrackState.tentative
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    feature_gallery: deque = field(default_factory=lambda: deque(maxlen=100))

    current_detections: Dict[str, ViewDetection] = field(default_factory=dict)
    current_bbox: Dict[str, BoxXYXY] = field(default_factory=dict)
    current_pose_2d: Dict[str, np.ndarray] = field(default_factory=dict)
    current_joint_confidences: Dict[str, np.ndarray] = field(default_factory=dict)

    kf2d_mean_by_camera: Dict[str, np.ndarray] = field(default_factory=dict)
    kf2d_cov_by_camera: Dict[str, np.ndarray] = field(default_factory=dict)

    kf3d: KalmanFilter3D = field(default_factory=KalmanFilter3D)
    mean_3d: Optional[np.ndarray] = None
    covariance_3d: Optional[np.ndarray] = None

    current_pose_3d_17: Optional[np.ndarray] = None
    current_pose_3d_15: Optional[np.ndarray] = None
    learned_pose_score: float = 0.0
    learned_views_used: List[str] = field(default_factory=list)

    last_known_3d_pos: Optional[np.ndarray] = None
    last_triangulated_frame: int = -1

    single_view_streak: int = 0
    multi_view_history: deque = field(default_factory=lambda: deque(maxlen=30))

    last_tracklet_id: Dict[str, int] = field(default_factory=dict)
    is_occluded_by: Optional[int] = None
    occlusion_timer: int = 0

    last_visible_ratio: float = 1.0
    aspect_ratio_history: deque = field(default_factory=lambda: deque(maxlen=5))
    position_3d_history: deque = field(default_factory=lambda: deque(maxlen=50))
    position_variance_history: deque = field(default_factory=lambda: deque(maxlen=20))

    color: np.ndarray = field(
        default_factory=lambda: np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    )

    @property
    def is_active(self) -> bool:
        return self.state != TrackState.lost

    @property
    def has_3d(self) -> bool:
        return self.mean_3d is not None and self.covariance_3d is not None

    def clear_current_frame_data(self) -> None:
        self.current_detections.clear()
        self.current_bbox.clear()
        self.current_pose_2d.clear()
        self.current_joint_confidences.clear()

        self.current_pose_3d_17 = None
        self.current_pose_3d_15 = None
        self.learned_pose_score = 0.0
        self.learned_views_used = []
        self.is_occluded_by = None

    def mark_hit(self, frame_id: int, min_hits_to_confirm: int) -> None:
        self.time_since_update = 0
        self.last_seen_frame = frame_id
        self.hits += 1

        if self.state == TrackState.tentative and self.hits >= min_hits_to_confirm:
            self.state = TrackState.confirmed_2d

        elif self.state in {TrackState.occluded, TrackState.lost}:
            self.state = TrackState.confirmed_3d if self.has_3d else TrackState.confirmed_2d

    def mark_missed(self, max_age: int) -> None:
        self.time_since_update += 1

        if self.time_since_update > max_age:
            self.state = TrackState.lost
        else:
            self.state = TrackState.occluded

    def update_feature(self, feature: Optional[np.ndarray], weight: float) -> None:
        if feature is None:
            return

        feature = feature.astype(np.float32)
        norm = np.linalg.norm(feature)

        if norm <= 1e-12:
            return

        feature = feature / norm

        if not self.feature_gallery:
            self.feature_gallery.append(feature)
            return

        gallery = np.stack(list(self.feature_gallery), axis=0)
        distances = cdist([feature], gallery, metric="cosine")[0]

        min_index = int(np.argmin(distances))
        min_distance = float(distances[min_index])

        if min_distance < 0.15:
            updated = (1.0 - weight) * self.feature_gallery[min_index] + weight * feature
            updated_norm = np.linalg.norm(updated)

            if updated_norm > 1e-12:
                self.feature_gallery[min_index] = updated / updated_norm

        elif min_distance < 0.50:
            self.feature_gallery.append(feature)

    def appearance_distance(self, feature: Optional[np.ndarray]) -> float:
        if feature is None:
            return 1.0

        if not self.feature_gallery:
            return 1.0

        gallery = np.stack(list(self.feature_gallery), axis=0)
        distances = cdist([feature], gallery, metric="cosine")[0]

        return float(np.min(distances))

    def update_3d_pose(
        self,
        pose_3d_17: np.ndarray,
        pose_3d_15: np.ndarray,
        root_position: np.ndarray,
        score: float,
        views_used: List[str],
    ) -> None:
        self.current_pose_3d_17 = pose_3d_17
        self.current_pose_3d_15 = pose_3d_15
        self.learned_pose_score = float(score)
        self.learned_views_used = list(views_used)

        root_position = np.asarray(root_position, dtype=np.float64)

        if self.mean_3d is None or self.covariance_3d is None:
            self.mean_3d, self.covariance_3d = self.kf3d.initiate(root_position)
        else:
            self.mean_3d, self.covariance_3d = self.kf3d.update(
                self.mean_3d,
                self.covariance_3d,
                root_position,
            )

        self.state = TrackState.confirmed_3d
        self.last_known_3d_pos = root_position.copy()
        self.position_3d_history.append(root_position.copy())


class MultiViewTracker:
    def __init__(
        self,
        camera_names: List[str],
        track_patience: int,
        min_hits_to_confirm: int,
        intra_frame_appearance_threshold: float,
        association_threshold: float,
        min_confidence_for_new_track: float,
        min_views_for_new_track: int,
        feature_gallery_size: int,
        feature_update_weight: float,
        tracklet_conflict_max_age: int,
        motion_weight: float,
        appearance_weight: float,
        spatial_weight: float = 0.35,
        reprojection_error_threshold: float = 120.0,
        spatial_consistency_threshold_mm: float = 1500.0,
        duplicate_merge_distance_mm: float = 600.0,
        duplicate_merge_appearance: float = 0.30,
        group_safety_merge_distance_mm: float = 400.0,
        occlusion_xy_distance_mm: float = 900.0,
        max_occlusion_frames: int = 120,
        reactivation_appearance_threshold: float = 0.35,
        occlusion_reid_threshold_mm: float = 1200.0,
        min_visibility_for_new_track: float = 0.40,
        min_detection_confidence_for_matching: float = 0.10,
        min_detection_confidence_for_grouping: float = 0.20,
        min_detection_confidence_for_new_track_single_view: float = 0.70,
        min_detection_confidence_for_new_track_multiview: float = 0.55,
        min_bbox_height_for_new_track: int = 80,
        min_bbox_width_for_new_track: int = 30,
        allow_featureless_detection_for_matching: bool = True,
        allow_featureless_detection_for_new_track: bool = False,
        featureless_pair_cost: float = 0.85,
        strong_spatial_score_for_featureless_pair: float = 0.65,
    ) -> None:
        self.camera_names = list(camera_names)

        self.track_patience = int(track_patience)
        self.min_hits_to_confirm = int(min_hits_to_confirm)

        self.intra_frame_appearance_threshold = float(intra_frame_appearance_threshold)
        self.association_threshold = float(association_threshold)

        self.min_confidence_for_new_track = float(min_confidence_for_new_track)
        self.min_views_for_new_track = int(min_views_for_new_track)

        self.feature_gallery_size = int(feature_gallery_size)
        self.feature_update_weight = float(feature_update_weight)

        self.tracklet_conflict_max_age = int(tracklet_conflict_max_age)

        self.motion_weight = float(motion_weight)
        self.appearance_weight = float(appearance_weight)
        self.spatial_weight = float(spatial_weight)

        total = max(
            1e-12,
            self.motion_weight + self.appearance_weight + self.spatial_weight,
        )

        self.motion_weight /= total
        self.appearance_weight /= total
        self.spatial_weight /= total

        self.reprojection_error_threshold = float(reprojection_error_threshold)
        self.spatial_consistency_threshold_mm = float(spatial_consistency_threshold_mm)
        self.duplicate_merge_distance_mm = float(duplicate_merge_distance_mm)
        self.duplicate_merge_appearance = float(duplicate_merge_appearance)
        self.group_safety_merge_distance_mm = float(group_safety_merge_distance_mm)
        self.occlusion_xy_distance_mm = float(occlusion_xy_distance_mm)
        self.max_occlusion_frames = int(max_occlusion_frames)
        self.reactivation_appearance_threshold = float(reactivation_appearance_threshold)
        self.occlusion_reid_threshold_mm = float(occlusion_reid_threshold_mm)
        self.min_visibility_for_new_track = float(min_visibility_for_new_track)

        self.min_detection_confidence_for_matching = float(min_detection_confidence_for_matching)
        self.min_detection_confidence_for_grouping = float(min_detection_confidence_for_grouping)
        self.min_detection_confidence_for_new_track_single_view = float(
            min_detection_confidence_for_new_track_single_view
        )
        self.min_detection_confidence_for_new_track_multiview = float(
            min_detection_confidence_for_new_track_multiview
        )

        self.min_bbox_height_for_new_track = int(min_bbox_height_for_new_track)
        self.min_bbox_width_for_new_track = int(min_bbox_width_for_new_track)

        self.allow_featureless_detection_for_matching = bool(allow_featureless_detection_for_matching)
        self.allow_featureless_detection_for_new_track = bool(allow_featureless_detection_for_new_track)

        self.featureless_pair_cost = float(featureless_pair_cost)
        self.strong_spatial_score_for_featureless_pair = float(strong_spatial_score_for_featureless_pair)

        self.kf2d = KalmanFilter2D()

        self.tracks: Dict[int, GlobalTrack] = {}
        self.next_global_id = 1

        self.tracklet_registry: Dict[Tuple[str, int], Dict[str, int]] = {}
        self.current_frame_id = 0

    def active_tracks(self) -> List[GlobalTrack]:
        return [
            track
            for track in self.tracks.values()
            if track.state != TrackState.lost
        ]

    def confirmed_tracks(self) -> List[GlobalTrack]:
        return [
            track
            for track in self.tracks.values()
            if track.state in {TrackState.confirmed_2d, TrackState.confirmed_3d}
        ]

    def bbox_size_ok_for_new_track(self, bbox: BoxXYXY) -> bool:
        width, height = bbox_size(bbox)

        return (
            width >= self.min_bbox_width_for_new_track
            and height >= self.min_bbox_height_for_new_track
        )

    def detection_is_usable_for_matching(self, detection: ViewDetection) -> bool:
        if detection.confidence < self.min_detection_confidence_for_matching:
            return False

        if detection.feature is None and not self.allow_featureless_detection_for_matching:
            return False

        return True

    def detection_is_usable_for_grouping(self, detection: ViewDetection) -> bool:
        if detection.confidence < self.min_detection_confidence_for_grouping:
            return False

        return True

    def group_is_reliable_for_new_track(self, group: DetectionGroup) -> bool:
        if group.num_views >= 2:
            if group.mean_confidence < self.min_detection_confidence_for_new_track_multiview:
                return False
        else:
            if group.mean_confidence < self.min_detection_confidence_for_new_track_single_view:
                return False

        if group.num_views < self.min_views_for_new_track:
            return False

        if group.visibility_ratio < self.min_visibility_for_new_track and group.num_views == 1:
            return False

        if not self.allow_featureless_detection_for_new_track:
            if group.mean_feature() is None:
                return False

        for detection in group.detections:
            if not self.bbox_size_ok_for_new_track(detection.bbox_xyxy):
                return False

        return True

    def predict_tracks(self) -> None:
        for track in self.tracks.values():
            if track.state == TrackState.lost:
                continue

            track.age += 1
            track.clear_current_frame_data()

            for camera_name in list(track.kf2d_mean_by_camera.keys()):
                mean = track.kf2d_mean_by_camera[camera_name]
                covariance = track.kf2d_cov_by_camera[camera_name]

                mean, covariance = self.kf2d.predict(mean, covariance)

                track.kf2d_mean_by_camera[camera_name] = mean
                track.kf2d_cov_by_camera[camera_name] = covariance

            if track.mean_3d is not None and track.covariance_3d is not None:
                track.mean_3d, track.covariance_3d = track.kf3d.predict(
                    track.mean_3d,
                    track.covariance_3d,
                )

            if track.occlusion_timer > 0:
                track.occlusion_timer -= 1

    def all_detections(
        self,
        detections_by_camera: Dict[str, List[ViewDetection]],
    ) -> List[ViewDetection]:
        detections: List[ViewDetection] = []

        for camera_name in self.camera_names:
            detections.extend(detections_by_camera.get(camera_name, []))

        return detections

    def center_from_detection(self, detection: ViewDetection) -> tuple[np.ndarray, float]:
        if detection.pose_2d is not None and detection.joint_confidences is not None:
            torso_indices = [5, 6, 11, 12]
            valid_points = []
            valid_scores = []

            for index in torso_indices:
                if index >= len(detection.pose_2d) or index >= len(detection.joint_confidences):
                    continue

                confidence = float(detection.joint_confidences[index])

                if confidence < 0.25:
                    continue

                point = detection.pose_2d[index]

                if not np.isfinite(point).all():
                    continue

                valid_points.append(point)
                valid_scores.append(confidence)

            if len(valid_points) >= 2:
                return (
                    np.mean(np.asarray(valid_points, dtype=np.float64), axis=0),
                    float(np.mean(valid_scores)),
                )

        return bbox_center(detection.bbox_xyxy), 0.25

    def triangulate_group(
        self,
        group: DetectionGroup,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[tuple[np.ndarray, float]]:
        if cameras_by_camera is None:
            return None

        projections = []
        points_2d = []
        weights = []
        cameras = []

        for detection in group.detections:
            camera = cameras_by_camera.get(detection.camera_name)

            if camera is None:
                continue

            center, weight = self.center_from_detection(detection)

            projections.append(camera.P)
            points_2d.append(center)
            weights.append(weight)
            cameras.append(camera)

        if len(projections) < 2:
            return None

        try:
            point_3d = weighted_triangulate_point(projections, points_2d, weights)
        except Exception:
            return None

        valid_cheirality = sum(
            check_cheirality(point_3d, camera)
            for camera in cameras
        )

        if valid_cheirality < 2:
            return None

        errors = []

        for projection, point_2d in zip(projections, points_2d):
            projected = project_points(point_3d.reshape(1, 3), projection)[0]
            errors.append(float(np.linalg.norm(projected - point_2d)))

        mean_error = float(np.mean(errors))

        if mean_error > self.reprojection_error_threshold:
            return None

        confidence = max(
            0.0,
            1.0 - mean_error / max(1e-6, self.reprojection_error_threshold),
        )

        return point_3d.astype(np.float32), confidence

    def spatial_consistency_score(
        self,
        group_a: DetectionGroup,
        group_b: DetectionGroup,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> float:
        if not group_a.camera_set.isdisjoint(group_b.camera_set):
            return 0.0

        merged = DetectionGroup(group_a.detections + group_b.detections)
        result = self.triangulate_group(merged, cameras_by_camera)

        if result is None:
            return 0.0

        return float(result[1])

    def detection_pair_cost(
        self,
        det_a: ViewDetection,
        det_b: ViewDetection,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> float:
        if det_a.camera_name == det_b.camera_name:
            return large_cost

        if not self.detection_is_usable_for_grouping(det_a):
            return large_cost

        if not self.detection_is_usable_for_grouping(det_b):
            return large_cost

        group_a = DetectionGroup([det_a])
        group_b = DetectionGroup([det_b])

        spatial_score = self.spatial_consistency_score(
            group_a,
            group_b,
            cameras_by_camera,
        )

        has_both_features = det_a.feature is not None and det_b.feature is not None

        if has_both_features:
            appearance_cost = float(
                cdist([det_a.feature], [det_b.feature], metric="cosine")[0, 0]
            )

            if spatial_score > 0:
                return 0.5 * appearance_cost + 0.5 * (1.0 - spatial_score)

            return appearance_cost

        if spatial_score >= self.strong_spatial_score_for_featureless_pair:
            return 1.0 - spatial_score

        return large_cost

    def group_detections(
        self,
        detections_by_camera: Dict[str, List[ViewDetection]],
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> List[DetectionGroup]:
        detections = [
            detection
            for detection in self.all_detections(detections_by_camera)
            if self.detection_is_usable_for_matching(detection)
        ]

        if not detections:
            return []

        clusters: Dict[int, List[int]] = {
            index: [index]
            for index in range(len(detections))
        }

        pairwise_cost = np.full(
            (len(detections), len(detections)),
            fill_value=large_cost,
            dtype=np.float32,
        )

        for i, det_i in enumerate(detections):
            for j, det_j in enumerate(detections):
                if i == j:
                    pairwise_cost[i, j] = 0.0
                    continue

                pairwise_cost[i, j] = self.detection_pair_cost(
                    det_i,
                    det_j,
                    cameras_by_camera,
                )

        while True:
            best_pair = None
            best_cost = float("inf")

            cluster_keys = list(clusters.keys())

            for i in range(len(cluster_keys)):
                for j in range(i + 1, len(cluster_keys)):
                    key_a = cluster_keys[i]
                    key_b = cluster_keys[j]

                    group_a = DetectionGroup([detections[index] for index in clusters[key_a]])
                    group_b = DetectionGroup([detections[index] for index in clusters[key_b]])

                    if not group_a.camera_set.isdisjoint(group_b.camera_set):
                        continue

                    costs = pairwise_cost[np.ix_(clusters[key_a], clusters[key_b])]

                    if costs.size == 0:
                        continue

                    mean_cost = float(np.mean(costs))

                    if (
                        mean_cost < self.intra_frame_appearance_threshold
                        and mean_cost < best_cost
                    ):
                        best_cost = mean_cost
                        best_pair = (key_a, key_b)

            if best_pair is None:
                break

            key_a, key_b = best_pair
            clusters[key_a].extend(clusters[key_b])
            del clusters[key_b]

        groups = [
            DetectionGroup([detections[index] for index in indices])
            for indices in clusters.values()
        ]

        return self.safety_merge_close_groups(groups, cameras_by_camera)

    def safety_merge_close_groups(
        self,
        groups: List[DetectionGroup],
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> List[DetectionGroup]:
        if len(groups) <= 1:
            return groups

        merged = True

        while merged:
            merged = False
            i = 0

            while i < len(groups):
                j = i + 1

                while j < len(groups):
                    group_a = groups[i]
                    group_b = groups[j]

                    if not group_a.camera_set.isdisjoint(group_b.camera_set):
                        j += 1
                        continue

                    pos_a = self.triangulate_group(group_a, cameras_by_camera)
                    pos_b = self.triangulate_group(group_b, cameras_by_camera)

                    if pos_a is not None and pos_b is not None:
                        distance = np.linalg.norm(pos_a[0] - pos_b[0])

                        if distance < self.group_safety_merge_distance_mm:
                            groups[i] = DetectionGroup(group_a.detections + group_b.detections)
                            groups.pop(j)
                            merged = True
                            continue

                    j += 1

                i += 1

        return groups

    def compute_appearance_cost(
        self,
        tracks: List[GlobalTrack],
        groups: List[DetectionGroup],
    ) -> np.ndarray:
        cost = np.ones((len(tracks), len(groups)), dtype=np.float32)

        group_features = [group.mean_feature() for group in groups]

        for track_index, track in enumerate(tracks):
            for group_index, feature in enumerate(group_features):
                cost[track_index, group_index] = track.appearance_distance(feature)

        return cost

    def motion_cost_for_group(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
        gating_multiplier: float,
    ) -> float:
        costs = []
        gating_threshold = chi2inv95[4] * gating_multiplier

        for detection in group.detections:
            camera_name = detection.camera_name

            if camera_name not in track.kf2d_mean_by_camera:
                continue

            measurement = bbox_to_kalman(detection.bbox_xyxy)
            mean = track.kf2d_mean_by_camera[camera_name]
            covariance = track.kf2d_cov_by_camera[camera_name]

            distance = self.kf2d.gating_distance(
                mean,
                covariance,
                np.asarray([measurement]),
            )[0]

            if distance > gating_threshold:
                return large_cost

            costs.append(float(distance / gating_threshold))

        if not costs:
            return 0.5

        return float(np.mean(costs))

    def spatial_cost_for_group(
        self,
        track: GlobalTrack,
        group_3d: Optional[tuple[np.ndarray, float]],
        gating_multiplier: float,
    ) -> float:
        if not track.has_3d or group_3d is None:
            return 0.5

        point_3d = group_3d[0]
        gating_threshold = chi2inv95[3] * gating_multiplier

        distance = track.kf3d.gating_distance(
            track.mean_3d,
            track.covariance_3d,
            np.asarray([point_3d]),
        )[0]

        if distance > gating_threshold:
            return large_cost

        return float(distance / gating_threshold)

    def adaptive_weights(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
        group_3d_info: Optional[tuple[np.ndarray, float]],
    ) -> tuple[float, float, float, float, float]:
        group_visibility = group.visibility_ratio
        num_views = group.num_views

        recent_view_pattern = list(track.multi_view_history)[-5:] if track.multi_view_history else []
        is_transitioning = len(set(recent_view_pattern)) > 1 if recent_view_pattern else False

        if is_transitioning or track.single_view_streak > 0:
            if num_views == 1 and track.has_3d:
                return 0.60, 0.30, 0.10, 0.85, 3.0

            if num_views >= 2 and not track.has_3d:
                return 0.50, 0.30, 0.20, 0.80, 2.5

            return 0.50, 0.25, 0.25, 0.75, 2.0

        if group_visibility < 0.30:
            if track.state == TrackState.occluded:
                return 0.20, 0.30, 0.50, 0.90, 6.0

            return 0.30, 0.20, 0.50, 0.85, 5.0

        if track.state == TrackState.occluded:
            return 0.10, 0.45, 0.45, 0.95, 10.0

        if track.is_occluded_by is not None or track.time_since_update > 5:
            return 0.70, 0.15, 0.15, 0.80, 4.0

        if track.has_3d and group_3d_info is not None:
            return 0.50, 0.10, 0.40, 0.70, 2.0

        return 0.70, 0.30, 0.00, 0.80, 2.0

    def recent_tracklet_owner(
        self,
        camera_name: str,
        tracklet_id: int,
        exclude_global_id: Optional[int] = None,
    ) -> Optional[int]:
        key = (camera_name, tracklet_id)
        info = self.tracklet_registry.get(key)

        if info is None:
            return None

        if self.current_frame_id - info["frame_id"] > self.tracklet_conflict_max_age:
            return None

        owner = int(info["global_id"])

        if exclude_global_id is not None and owner == exclude_global_id:
            return None

        if owner not in self.tracks:
            return None

        return owner

    def is_same_tracklet_reliable(
        self,
        track: GlobalTrack,
        detection: ViewDetection,
    ) -> bool:
        if detection.tracklet_id is None:
            return False

        camera_name = detection.camera_name

        if track.last_tracklet_id.get(camera_name) != detection.tracklet_id:
            return False

        if camera_name in track.kf2d_mean_by_camera:
            pred_bbox = kalman_to_bbox(track.kf2d_mean_by_camera[camera_name])
            curr_center = bbox_center(detection.bbox_xyxy)

            pred_center = np.array(
                [
                    0.5 * (pred_bbox[0] + pred_bbox[2]),
                    0.5 * (pred_bbox[1] + pred_bbox[3]),
                ],
                dtype=np.float64,
            )

            distance = np.linalg.norm(curr_center - pred_center)
            max_distance = 120.0 + 25.0 * track.time_since_update

            if distance > max_distance:
                return False

        if track.feature_gallery and detection.feature is not None:
            app = track.appearance_distance(detection.feature)

            if app > 0.35:
                return False

        return True

    def has_tracklet_conflict(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
    ) -> bool:
        for detection in group.detections:
            if detection.tracklet_id is None:
                continue

            owner = self.recent_tracklet_owner(
                detection.camera_name,
                detection.tracklet_id,
                exclude_global_id=track.global_id,
            )

            if owner is not None:
                return True

        return False

    def compute_cost_matrix(
        self,
        tracks: List[GlobalTrack],
        groups: List[DetectionGroup],
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> np.ndarray:
        if not tracks or not groups:
            return np.empty((len(tracks), len(groups)), dtype=np.float32)

        appearance_cost = self.compute_appearance_cost(tracks, groups)
        group_3d_infos = [
            self.triangulate_group(group, cameras_by_camera)
            for group in groups
        ]

        track_positions_3d = {
            track.global_id: track.mean_3d[:3]
            for track in self.active_tracks()
            if track.has_3d
        }

        cost_matrix = np.full(
            (len(tracks), len(groups)),
            fill_value=large_cost,
            dtype=np.float32,
        )

        for track_index, track in enumerate(tracks):
            for group_index, group in enumerate(groups):
                group_3d_info = group_3d_infos[group_index]
                w_app, w_motion, w_spatial, threshold, gating_multiplier = self.adaptive_weights(
                    track,
                    group,
                    group_3d_info,
                )

                if self.has_tracklet_conflict(track, group):
                    continue

                motion_cost = self.motion_cost_for_group(
                    track,
                    group,
                    gating_multiplier,
                )

                if motion_cost >= large_cost:
                    continue

                spatial_cost = self.spatial_cost_for_group(
                    track,
                    group_3d_info,
                    gating_multiplier,
                )

                if spatial_cost >= large_cost:
                    continue

                total_cost = (
                    w_app * appearance_cost[track_index, group_index]
                    + w_motion * motion_cost
                    + w_spatial * spatial_cost
                )

                if track.has_3d and group_3d_info is not None:
                    proximity_penalty = 0.0
                    group_pos = group_3d_info[0]

                    for other_track_id, other_pos in track_positions_3d.items():
                        if other_track_id == track.global_id:
                            continue

                        if np.linalg.norm(track.mean_3d[:3] - other_pos) < 1200.0:
                            if np.linalg.norm(group_pos - other_pos) < np.linalg.norm(group_pos - track.mean_3d[:3]):
                                proximity_penalty = max(proximity_penalty, 0.2)

                    total_cost += proximity_penalty

                reliable_tracklets = 0

                for detection in group.detections:
                    if self.is_same_tracklet_reliable(track, detection):
                        reliable_tracklets += 1

                if reliable_tracklets > 0:
                    total_cost *= 0.85

                if total_cost < threshold:
                    cost_matrix[track_index, group_index] = float(total_cost)

        return cost_matrix

    def resolve_association_conflicts(
        self,
        matches: List[Tuple[int, int]],
        tracks: List[GlobalTrack],
        groups: List[DetectionGroup],
        cost_matrix: np.ndarray,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> List[Tuple[int, int]]:
        if not matches:
            return []

        group_positions = {
            index: self.triangulate_group(group, cameras_by_camera)
            for index, group in enumerate(groups)
        }

        invalid = set()

        for i in range(len(matches)):
            for j in range(i + 1, len(matches)):
                track_i, group_i = matches[i]
                track_j, group_j = matches[j]

                pos_i = group_positions.get(group_i)
                pos_j = group_positions.get(group_j)

                if pos_i is None or pos_j is None:
                    continue

                distance = np.linalg.norm(pos_i[0] - pos_j[0])

                if track_i == track_j:
                    if distance > self.spatial_consistency_threshold_mm:
                        if cost_matrix[track_i, group_i] > cost_matrix[track_j, group_j]:
                            invalid.add((track_i, group_i))
                        else:
                            invalid.add((track_j, group_j))

                elif distance < 500.0:
                    if cost_matrix[track_i, group_i] > cost_matrix[track_j, group_j]:
                        invalid.add((track_i, group_i))
                    else:
                        invalid.add((track_j, group_j))

        return [
            match for match in matches
            if match not in invalid
        ]

    def match_subset(
        self,
        tracks: List[GlobalTrack],
        groups: List[DetectionGroup],
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> List[Tuple[int, int]]:
        cost = self.compute_cost_matrix(tracks, groups, cameras_by_camera)

        if cost.size == 0:
            return []

        rows, cols = linear_sum_assignment(cost)

        matches = []

        for row, col in zip(rows, cols):
            if cost[row, col] < large_cost:
                matches.append((int(row), int(col)))

        return self.resolve_association_conflicts(
            matches,
            tracks,
            groups,
            cost,
            cameras_by_camera,
        )

    def matching_cascade(
        self,
        tracks: List[GlobalTrack],
        groups: List[DetectionGroup],
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not groups:
            return [], list(range(len(tracks))), list(range(len(groups)))

        unmatched_tracks = list(range(len(tracks)))
        unmatched_groups = list(range(len(groups)))
        final_matches: List[Tuple[int, int]] = []

        cascade = []

        transitioning = [
            index for index in unmatched_tracks
            if (
                0 < tracks[index].single_view_streak < 15
                or (
                    tracks[index].multi_view_history
                    and len(set(list(tracks[index].multi_view_history)[-5:])) > 1
                )
            )
        ]
        cascade.append(transitioning)

        confirmed = [
            index for index in unmatched_tracks
            if tracks[index].state in {TrackState.confirmed_2d, TrackState.confirmed_3d}
            and tracks[index].time_since_update < 5
            and not tracks[index].is_occluded_by
        ]
        cascade.append(confirmed)

        uncertain = [
            index for index in unmatched_tracks
            if tracks[index].state in {TrackState.occluded, TrackState.tentative}
        ]
        cascade.append(uncertain)

        for track_indices in cascade:
            track_indices = [
                index for index in track_indices
                if index in unmatched_tracks
            ]

            if not track_indices or not unmatched_groups:
                continue

            selected_tracks = [tracks[index] for index in track_indices]
            selected_groups = [groups[index] for index in unmatched_groups]

            local_matches = self.match_subset(
                selected_tracks,
                selected_groups,
                cameras_by_camera,
            )

            used_tracks = set()
            used_groups = set()

            for local_track, local_group in local_matches:
                original_track = track_indices[local_track]
                original_group = unmatched_groups[local_group]

                final_matches.append((original_track, original_group))
                used_tracks.add(original_track)
                used_groups.add(original_group)

            unmatched_tracks = [
                index for index in unmatched_tracks
                if index not in used_tracks
            ]

            unmatched_groups = [
                index for index in unmatched_groups
                if index not in used_groups
            ]

        partial_groups = [
            group_index for group_index in unmatched_groups
            if groups[group_index].visibility_ratio < 0.4
        ]

        if partial_groups and unmatched_tracks:
            selected_tracks = [tracks[index] for index in unmatched_tracks]
            selected_groups = [groups[index] for index in partial_groups]

            local_matches = self.match_subset(
                selected_tracks,
                selected_groups,
                cameras_by_camera,
            )

            used_tracks = set()
            used_groups = set()

            for local_track, local_group in local_matches:
                original_track = unmatched_tracks[local_track]
                original_group = partial_groups[local_group]

                final_matches.append((original_track, original_group))
                used_tracks.add(original_track)
                used_groups.add(original_group)

            unmatched_tracks = [
                index for index in unmatched_tracks
                if index not in used_tracks
            ]

            unmatched_groups = [
                index for index in unmatched_groups
                if index not in used_groups
            ]

        return final_matches, unmatched_tracks, unmatched_groups

    def handle_view_transition(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
        frame_id: int,
    ) -> None:
        num_views = group.num_views
        track.multi_view_history.append(num_views)

        if track.has_3d and num_views == 1:
            track.single_view_streak += 1

            if track.covariance_3d is not None:
                track.covariance_3d *= 1.1

            if track.single_view_streak < 10:
                track.state = TrackState.confirmed_2d

        elif not track.has_3d and num_views >= 2:
            track.single_view_streak = 0

        elif num_views >= 2:
            track.single_view_streak = 0
            track.last_triangulated_frame = frame_id

    def validate_3d_measurement(
        self,
        track: GlobalTrack,
        point_3d: np.ndarray,
    ) -> tuple[bool, np.ndarray]:
        point_3d = np.asarray(point_3d, dtype=np.float64)

        if not track.has_3d or not track.position_3d_history:
            return True, point_3d

        max_velocity_per_frame = 500.0
        max_velocity_per_frame *= max(1, track.time_since_update + 1)

        last_pos = np.asarray(track.position_3d_history[-1], dtype=np.float64)
        distance_from_last = np.linalg.norm(point_3d - last_pos)

        if distance_from_last > max_velocity_per_frame:
            predicted_pos = track.mean_3d[:3]
            distance_from_prediction = np.linalg.norm(point_3d - predicted_pos)

            if distance_from_prediction > max_velocity_per_frame * 2.0:
                return False, point_3d

            alpha = 0.6
            point_3d = alpha * point_3d + (1.0 - alpha) * predicted_pos

        if len(track.position_3d_history) >= 3:
            recent = np.asarray(list(track.position_3d_history)[-3:], dtype=np.float64)
            velocities = np.diff(recent, axis=0)
            avg_velocity = np.mean(velocities, axis=0)

            expected_pos = last_pos + avg_velocity
            deviation = np.linalg.norm(point_3d - expected_pos)

            if deviation > 300.0:
                weight = max(0.3, 1.0 - deviation / 1000.0)
                point_3d = weight * point_3d + (1.0 - weight) * expected_pos

        return True, point_3d

    def update_track_with_group(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> None:
        self.handle_view_transition(track, group, frame_id)

        track.mark_hit(frame_id, self.min_hits_to_confirm)

        feature = group.mean_feature()
        track.update_feature(feature, self.feature_update_weight)
        track.last_visible_ratio = group.visibility_ratio

        for detection in group.detections:
            camera_name = detection.camera_name

            track.current_detections[camera_name] = detection
            track.current_bbox[camera_name] = detection.bbox_xyxy

            if detection.pose_2d is not None:
                track.current_pose_2d[camera_name] = detection.pose_2d

            if detection.joint_confidences is not None:
                track.current_joint_confidences[camera_name] = detection.joint_confidences

            measurement = bbox_to_kalman(detection.bbox_xyxy)

            if camera_name not in track.kf2d_mean_by_camera:
                mean, covariance = self.kf2d.initiate(measurement)
            else:
                mean = track.kf2d_mean_by_camera[camera_name]
                covariance = track.kf2d_cov_by_camera[camera_name]
                mean, covariance = self.kf2d.update(mean, covariance, measurement)

            track.kf2d_mean_by_camera[camera_name] = mean
            track.kf2d_cov_by_camera[camera_name] = covariance

            if detection.tracklet_id is not None:
                track.last_tracklet_id[camera_name] = detection.tracklet_id
                self.tracklet_registry[(camera_name, detection.tracklet_id)] = {
                    "global_id": track.global_id,
                    "frame_id": frame_id,
                }

        avg_box = np.mean(
            np.asarray([detection.bbox_xyxy for detection in group.detections], dtype=np.float64),
            axis=0,
        )

        width = max(1.0, avg_box[2] - avg_box[0])
        height = max(1.0, avg_box[3] - avg_box[1])
        track.aspect_ratio_history.append(float(width / height))

        result_3d = self.triangulate_group(group, cameras_by_camera)

        if result_3d is not None:
            point_3d, _ = result_3d
            ok_3d, point_3d = self.validate_3d_measurement(track, point_3d)

            if ok_3d:
                if track.mean_3d is None or track.covariance_3d is None:
                    track.mean_3d, track.covariance_3d = track.kf3d.initiate(point_3d)
                else:
                    prediction_error = np.linalg.norm(point_3d - track.mean_3d[:3])

                    if prediction_error > 1500.0 and track.age > 100:
                        track.mean_3d, track.covariance_3d = track.kf3d.initiate(point_3d)
                    else:
                        track.mean_3d, track.covariance_3d = track.kf3d.update(
                            track.mean_3d,
                            track.covariance_3d,
                            point_3d,
                        )

                track.state = TrackState.confirmed_3d
                track.last_known_3d_pos = point_3d.copy()
                track.last_triangulated_frame = frame_id
                track.position_3d_history.append(point_3d.copy())

    def try_reassociate_single_view_group(
        self,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[GlobalTrack]:
        if group.num_views != 1:
            return None

        detection = group.detections[0]
        camera_name = detection.camera_name

        if detection.confidence < self.min_detection_confidence_for_matching:
            return None

        candidates = []

        for track in self.active_tracks():
            if track.time_since_update == 0:
                continue

            if camera_name not in track.kf2d_mean_by_camera:
                continue

            measurement = bbox_to_kalman(detection.bbox_xyxy)
            mean = track.kf2d_mean_by_camera[camera_name]
            covariance = track.kf2d_cov_by_camera[camera_name]

            maha = self.kf2d.gating_distance(
                mean,
                covariance,
                np.asarray([measurement]),
            )[0]

            gating_multiplier = 4.0 + 0.5 * min(track.time_since_update, 10)
            gating_threshold = chi2inv95[4] * gating_multiplier

            if maha > gating_threshold:
                continue

            motion_score = float(maha / gating_threshold)

            app_score = 0.5
            if detection.feature is not None and track.feature_gallery:
                app_score = track.appearance_distance(detection.feature)

            tracklet_bonus = 0.0
            if self.is_same_tracklet_reliable(track, detection):
                tracklet_bonus = -0.35

            if detection.feature is None and tracklet_bonus >= 0:
                if motion_score > 0.35:
                    continue

            if detection.feature is not None and app_score > 0.55 and tracklet_bonus >= 0:
                continue

            score = 0.60 * motion_score + 0.40 * app_score + tracklet_bonus
            candidates.append((score, track))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_track = candidates[0]

        if best_score > 0.85:
            return None

        self.update_track_with_group(
            best_track,
            group,
            frame_id,
            cameras_by_camera,
        )

        print(
            f"[tracking] réassociation mono-vue "
            f"track={best_track.global_id} "
            f"cam={camera_name} "
            f"score={best_score:.3f}"
        )

        return best_track

    def try_reactivate_existing_track(
        self,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[GlobalTrack]:
        if group.mean_confidence < self.min_detection_confidence_for_matching:
            return None

        feature = group.mean_feature()
        group_3d = self.triangulate_group(group, cameras_by_camera)

        candidates = []

        for track in self.active_tracks():
            if track.time_since_update == 0:
                continue

            tracklet_bonus = 0.0

            for detection in group.detections:
                if self.is_same_tracklet_reliable(track, detection):
                    tracklet_bonus = -0.25
                    break

            app = track.appearance_distance(feature)

            if feature is not None and app > max(0.5, self.reactivation_appearance_threshold) and tracklet_bonus >= 0:
                continue

            if group_3d is not None and track.has_3d:
                distance_3d = np.linalg.norm(group_3d[0] - track.mean_3d[:3])
                threshold = self.occlusion_reid_threshold_mm + track.time_since_update * 50.0

                if distance_3d > threshold:
                    continue

                score = app + distance_3d / threshold + tracklet_bonus

            else:
                if feature is None and tracklet_bonus >= 0:
                    continue

                score = app + 0.5 + tracklet_bonus

            candidates.append((score, track))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        score, track = candidates[0]

        if score < 0.8:
            self.update_track_with_group(
                track,
                group,
                frame_id,
                cameras_by_camera,
            )
            return track

        return None

    def create_track(
        self,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[GlobalTrack]:
        reassociated = self.try_reassociate_single_view_group(
            group,
            frame_id,
            cameras_by_camera,
        )

        if reassociated is not None:
            return reassociated

        reactivated = self.try_reactivate_existing_track(
            group,
            frame_id,
            cameras_by_camera,
        )

        if reactivated is not None:
            return reactivated

        if not self.group_is_reliable_for_new_track(group):
            return None

        for detection in group.detections:
            if detection.tracklet_id is None:
                continue

            owner = self.recent_tracklet_owner(
                detection.camera_name,
                detection.tracklet_id,
            )

            if owner is not None:
                return None

        group_3d = self.triangulate_group(group, cameras_by_camera)

        if group_3d is not None:
            new_pos = group_3d[0]

            for track in self.active_tracks():
                if not track.has_3d:
                    continue

                distance = np.linalg.norm(new_pos - track.mean_3d[:3])

                if distance < self.duplicate_merge_distance_mm:
                    feature = group.mean_feature()
                    app = track.appearance_distance(feature)

                    if app < 0.5:
                        self.update_track_with_group(
                            track,
                            group,
                            frame_id,
                            cameras_by_camera,
                        )
                        return track

        global_id = self.next_global_id
        self.next_global_id += 1

        track = GlobalTrack(
            global_id=global_id,
            first_seen_frame=frame_id,
            last_seen_frame=frame_id,
        )

        track.feature_gallery = deque(maxlen=self.feature_gallery_size)
        self.tracks[global_id] = track

        self.update_track_with_group(
            track,
            group,
            frame_id,
            cameras_by_camera,
        )

        print(
            f"[tracking] création track {global_id} "
            f"frame={frame_id} vues={group.num_views}"
        )

        return track

    def mark_unmatched_tracks(
        self,
        tracks: List[GlobalTrack],
        unmatched_track_indices: List[int],
        matched_tracks: List[GlobalTrack],
    ) -> None:
        matched_positions = [
            track.mean_3d[:3]
            for track in matched_tracks
            if track.has_3d
        ]

        for index in unmatched_track_indices:
            track = tracks[index]

            potentially_occluded = False

            if track.has_3d and matched_positions:
                distances = cdist(
                    np.asarray([track.mean_3d[:3]]),
                    np.asarray(matched_positions),
                )

                if distances.size > 0 and float(np.min(distances)) < self.occlusion_xy_distance_mm:
                    potentially_occluded = True

            is_recent_disappearance = track.time_since_update <= 3

            if (
                potentially_occluded or is_recent_disappearance
            ) and track.time_since_update < self.max_occlusion_frames:
                track.state = TrackState.occluded
                track.time_since_update += 1

                if track.covariance_3d is not None:
                    track.covariance_3d *= 1.2

            else:
                track.mark_missed(self.track_patience)

    def update_occlusion_state(self) -> None:
        confirmed_3d = [
            track
            for track in self.active_tracks()
            if track.has_3d and track.current_bbox
        ]

        for track in confirmed_3d:
            track.is_occluded_by = None

        for i in range(len(confirmed_3d)):
            for j in range(i + 1, len(confirmed_3d)):
                track_a = confirmed_3d[i]
                track_b = confirmed_3d[j]

                distance_xy = np.linalg.norm(track_a.mean_3d[:2] - track_b.mean_3d[:2])

                if distance_xy > self.occlusion_xy_distance_mm:
                    continue

                overlapping = False

                for camera_name in self.camera_names:
                    if camera_name not in track_a.current_bbox:
                        continue

                    if camera_name not in track_b.current_bbox:
                        continue

                    if bbox_iou(track_a.current_bbox[camera_name], track_b.current_bbox[camera_name]) > 0:
                        overlapping = True
                        break

                if not overlapping:
                    continue

                if track_a.mean_3d[2] < track_b.mean_3d[2]:
                    occluder, occluded = track_a, track_b
                else:
                    occluder, occluded = track_b, track_a

                occluded.is_occluded_by = occluder.global_id
                occluded.occlusion_timer = self.max_occlusion_frames

    def merge_duplicate_tracks(self) -> None:
        tracks = [
            track
            for track in self.active_tracks()
            if track.state == TrackState.confirmed_3d and track.has_3d
        ]

        merges = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                track_a = tracks[i]
                track_b = tracks[j]

                shared_current_camera = set(track_a.current_bbox.keys()) & set(track_b.current_bbox.keys())

                if shared_current_camera:
                    continue

                distance = np.linalg.norm(track_a.mean_3d[:3] - track_b.mean_3d[:3])

                if distance > self.duplicate_merge_distance_mm:
                    continue

                if track_a.feature_gallery and track_b.feature_gallery:
                    feature_a = np.mean(np.stack(list(track_a.feature_gallery)), axis=0)
                    feature_b = np.mean(np.stack(list(track_b.feature_gallery)), axis=0)

                    app_distance = float(cdist([feature_a], [feature_b], metric="cosine")[0, 0])
                else:
                    app_distance = 1.0

                if app_distance < self.duplicate_merge_appearance:
                    merges.append((track_a.global_id, track_b.global_id))

        for id_a, id_b in merges:
            if id_a not in self.tracks or id_b not in self.tracks:
                continue

            track_a = self.tracks[id_a]
            track_b = self.tracks[id_b]

            if track_a.first_seen_frame <= track_b.first_seen_frame:
                keep, remove = track_a, track_b
            else:
                keep, remove = track_b, track_a

            keep.current_detections.update(remove.current_detections)
            keep.current_bbox.update(remove.current_bbox)
            keep.current_pose_2d.update(remove.current_pose_2d)
            keep.current_joint_confidences.update(remove.current_joint_confidences)
            keep.kf2d_mean_by_camera.update(remove.kf2d_mean_by_camera)
            keep.kf2d_cov_by_camera.update(remove.kf2d_cov_by_camera)

            for feature in remove.feature_gallery:
                keep.update_feature(feature, self.feature_update_weight)

            for camera_name, tracklet_id in remove.last_tracklet_id.items():
                keep.last_tracklet_id[camera_name] = tracklet_id
                self.tracklet_registry[(camera_name, tracklet_id)] = {
                    "global_id": keep.global_id,
                    "frame_id": self.current_frame_id,
                }

            del self.tracks[remove.global_id]
            print(f"[tracking] fusion track {remove.global_id} -> {keep.global_id}")

    def purge_lost_tracks(self) -> None:
        to_delete = [
            global_id
            for global_id, track in self.tracks.items()
            if track.state == TrackState.lost
            and track.time_since_update > self.track_patience
        ]

        for global_id in to_delete:
            del self.tracks[global_id]
            print(f"[tracking] suppression track {global_id}")

    def purge_tracklet_registry(self, frame_id: int) -> None:
        to_delete = []

        for key, info in self.tracklet_registry.items():
            if frame_id - info["frame_id"] > 30:
                to_delete.append(key)

        for key in to_delete:
            del self.tracklet_registry[key]

    def check_and_reset_diverged_tracks(self) -> None:
        to_delete = []

        for global_id, track in self.tracks.items():
            if track.covariance_3d is not None:
                max_uncertainty = np.max(np.diag(track.covariance_3d[:3, :3]))

                if max_uncertainty > 1e6:
                    if track.last_known_3d_pos is not None:
                        track.mean_3d, track.covariance_3d = track.kf3d.initiate(
                            track.last_known_3d_pos
                        )
                    else:
                        to_delete.append(global_id)

        for global_id in to_delete:
            if global_id in self.tracks:
                del self.tracks[global_id]

    def update(
        self,
        frame_id: int,
        detections_by_camera: Dict[str, List[ViewDetection]],
        cameras_by_camera: Optional[Dict[str, CameraData]] = None,
    ) -> Dict[int, GlobalTrack]:
        self.current_frame_id = frame_id

        self.predict_tracks()

        groups = self.group_detections(
            detections_by_camera=detections_by_camera,
            cameras_by_camera=cameras_by_camera,
        )

        active_tracks = self.active_tracks()

        matches, unmatched_track_indices, unmatched_group_indices = self.matching_cascade(
            active_tracks,
            groups,
            cameras_by_camera,
        )

        matched_tracks = []

        for track_index, group_index in matches:
            track = active_tracks[track_index]
            group = groups[group_index]

            self.update_track_with_group(
                track,
                group,
                frame_id,
                cameras_by_camera,
            )

            matched_tracks.append(track)

        self.mark_unmatched_tracks(
            active_tracks,
            unmatched_track_indices,
            matched_tracks,
        )

        for group_index in unmatched_group_indices:
            self.create_track(
                groups[group_index],
                frame_id,
                cameras_by_camera,
            )

        self.update_occlusion_state()

        if frame_id % 30 == 0:
            self.merge_duplicate_tracks()

        self.check_and_reset_diverged_tracks()
        self.purge_lost_tracks()
        self.purge_tracklet_registry(frame_id)

        return self.tracks

    def print_summary(self, frame_id: int) -> None:
        active = self.active_tracks()
        confirmed = self.confirmed_tracks()

        print(
            f"[tracking] frame={frame_id} "
            f"tracks_actives={len(active)} "
            f"tracks_confirmées={len(confirmed)}"
        )

        for track in active:
            print(
                f"  id={track.global_id} "
                f"state={track.state.name} "
                f"views={sorted(track.current_bbox.keys())} "
                f"hits={track.hits} "
                f"miss={track.time_since_update}"
            )