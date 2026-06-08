from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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


def bbox_intersection_area(a: BoxXYXY, b: BoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return float(max(0, ix2 - ix1) * max(0, iy2 - iy1))


def bbox_containment_of_smaller(a: BoxXYXY, b: BoxXYXY) -> float:
    inter = bbox_intersection_area(a, b)

    if inter <= 0:
        return 0.0

    return float(inter / max(1.0, min(bbox_area(a), bbox_area(b))))


def bbox_center_distance_ratio(a: BoxXYXY, b: BoxXYXY) -> float:
    ca = bbox_center(a)
    cb = bbox_center(b)
    scale = np.sqrt(max(bbox_area(a), bbox_area(b), 1.0))
    return float(np.linalg.norm(ca - cb) / max(1e-6, scale))


def bbox_small_area_ratio(a: BoxXYXY, b: BoxXYXY) -> float:
    area_a = bbox_area(a)
    area_b = bbox_area(b)

    return float(min(area_a, area_b) / max(1.0, max(area_a, area_b)))


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

    current_pose_3d_19: Optional[np.ndarray] = None
    current_pose_3d_18: Optional[np.ndarray] = None
    learned_pose_score: float = 0.0
    learned_views_used: List[str] = field(default_factory=list)

    current_unreliable_pose_3d_19: Optional[np.ndarray] = None
    current_unreliable_pose_3d_18: Optional[np.ndarray] = None
    unreliable_pose_score: float = 0.0
    unreliable_views_used: List[str] = field(default_factory=list)
    unreliable_pose_reason: str = ""

    tracking_3d_updated_this_frame: bool = False
    tracking_3d_views_used: List[str] = field(default_factory=list)

    last_known_3d_pos: Optional[np.ndarray] = None
    last_triangulated_frame: int = -1

    single_view_streak: int = 0
    multi_view_history: deque = field(default_factory=lambda: deque(maxlen=30))

    last_tracklet_id: Dict[str, int] = field(default_factory=dict)
    last_seen_frame_by_camera: Dict[str, int] = field(default_factory=dict)
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

        self.current_pose_3d_19 = None
        self.current_pose_3d_18 = None
        self.learned_pose_score = 0.0
        self.learned_views_used = []
        self.current_unreliable_pose_3d_19 = None
        self.current_unreliable_pose_3d_18 = None
        self.unreliable_pose_score = 0.0
        self.unreliable_views_used = []
        self.unreliable_pose_reason = ""
        self.tracking_3d_updated_this_frame = False
        self.tracking_3d_views_used = []
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
        pose_3d_19: np.ndarray,
        pose_3d_18: np.ndarray,
        root_position: np.ndarray,
        score: float,
        views_used: List[str],
    ) -> None:
        self.current_pose_3d_19 = pose_3d_19
        self.current_pose_3d_18 = pose_3d_18
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

    def update_unreliable_3d_pose(
        self,
        pose_3d_19: np.ndarray,
        pose_3d_18: np.ndarray,
        score: float,
        views_used: List[str],
        reason: str,
    ) -> None:
        """Stocke une pose learned non fiable pour l'affichage/debug.
        """
        self.current_unreliable_pose_3d_19 = pose_3d_19
        self.current_unreliable_pose_3d_18 = pose_3d_18
        self.unreliable_pose_score = float(score)
        self.unreliable_views_used = list(views_used)
        self.unreliable_pose_reason = str(reason)
class AssociationDebugWriter:
    def __init__(self, output_path: str | Path | None) -> None:
        self.output_path = Path(output_path) if output_path is not None else None
        self.file = None
        self.writer = None

        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(self.output_path, "w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                "frame_id",
                "track_id",
                "group_index",
                "group_views",
                "app_cost",
                "motion_cost",
                "spatial_cost",
                "total_cost",
                "decision",
                "reason",
            ])
            self.file.flush()

    def write(
        self,
        frame_id: int,
        track_id: int,
        group_index: int,
        group_views: List[str],
        app_cost: float,
        motion_cost: float,
        spatial_cost: float,
        total_cost: float,
        decision: str,
        reason: str,
    ) -> None:
        if self.writer is None:
            return

        self.writer.writerow([
            int(frame_id),
            int(track_id),
            int(group_index),
            "|".join(group_views),
            float(app_cost),
            float(motion_cost),
            float(spatial_cost),
            float(total_cost),
            decision,
            reason,
        ])
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None


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
        association_margin: float = 0.15,
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
        single_view_reassociation_margin: float = 0.20,
        single_view_reassociation_max_score: float = 0.85,
        single_view_reassociation_max_missing: int = 80,
        single_view_motion_threshold_without_feature: float = 0.35,
        single_view_appearance_reject_threshold: float = 0.55,
        association_debug_path: str | Path | None = None,
    ) -> None:
        self.camera_names = list(camera_names)

        self.track_patience = int(track_patience)
        self.min_hits_to_confirm = int(min_hits_to_confirm)

        self.intra_frame_appearance_threshold = float(intra_frame_appearance_threshold)
        self.association_threshold = float(association_threshold)
        self.association_margin = float(association_margin)

        self.min_confidence_for_new_track = float(min_confidence_for_new_track)
        self.min_views_for_new_track = int(min_views_for_new_track)

        self.feature_gallery_size = int(feature_gallery_size)
        self.feature_update_weight = float(feature_update_weight)

        self.tracklet_conflict_max_age = int(tracklet_conflict_max_age)

        self.motion_weight = float(motion_weight)
        self.appearance_weight = float(appearance_weight)
        self.spatial_weight = float(spatial_weight)

        total_weight = max(
            1e-12,
            self.motion_weight + self.appearance_weight + self.spatial_weight,
        )

        self.motion_weight /= total_weight
        self.appearance_weight /= total_weight
        self.spatial_weight /= total_weight

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

        self.single_view_reassociation_margin = float(single_view_reassociation_margin)
        self.single_view_reassociation_max_score = float(single_view_reassociation_max_score)
        self.single_view_reassociation_max_missing = int(single_view_reassociation_max_missing)
        self.single_view_motion_threshold_without_feature = float(single_view_motion_threshold_without_feature)
        self.single_view_appearance_reject_threshold = float(single_view_appearance_reject_threshold)

        self.kf2d = KalmanFilter2D()

        self.tracks: Dict[int, GlobalTrack] = {}
        self.next_global_id = 1

        self.tracklet_registry: Dict[Tuple[str, int], Dict[str, int]] = {}
        self.current_frame_id = 0

        self.debug_writer = AssociationDebugWriter(association_debug_path)

        self.local_continuity_lock_enabled = True

        self.predicted_display_enabled = True
        self.predicted_display_max_missing = 12
        self.per_camera_predicted_display_enabled = True
        self.per_camera_predicted_display_max_missing = 3
        self.defer_new_track_if_recent_track_possible = True

        self.suppress_predicted_ghost_enabled = True
        self.predicted_ghost_iou_threshold = 0.42
        self.predicted_ghost_containment_threshold = 0.78
        self.predicted_ghost_center_ratio_threshold = 0.32

        self.current_frame_multiview_group_count = 0
        self.current_frame_group_count = 0
        self.current_frame_detections_by_camera: Dict[str, List[ViewDetection]] = {}

        self.partial_occlusion_handling_enabled = True
        self.partial_occlusion_larger_area_ratio = 1.65
        self.partial_occlusion_small_area_ratio = 0.62
        self.partial_occlusion_min_containment = 0.30
        self.partial_occlusion_min_iou = 0.12
        self.partial_occlusion_max_center_ratio = 0.70
        self.partial_occlusion_visibility_threshold = 0.65
        self.partial_occlusion_new_track_min_confidence = 0.82

        self.crowded_camera_detection_count = 3
        self.crowded_single_view_strict_motion = 0.25
        self.crowded_single_view_strict_appearance = 0.30
        self.crowded_single_view_penalty = 0.25

        self.require_reliable_views_for_tracking_3d = True

    def close(self) -> None:
        self.debug_writer.close()

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

        if not self.allow_featureless_detection_for_new_track and group.mean_feature() is None:
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

    def camera_is_crowded(self, camera_name: str) -> bool:
        return len(self.current_frame_detections_by_camera.get(camera_name, [])) >= int(
            self.crowded_camera_detection_count
        )

    def detection_is_reliable_for_tracking_3d(self, detection: ViewDetection) -> bool:
        if not bool(getattr(detection, "reliable_for_3d", True)):
            return False

        if bool(getattr(detection, "edge_truncated", False)):
            return False

        if bool(getattr(detection, "partial_body", False)):
            return False

        if int(getattr(detection, "valid_joint_count", 99)) < 8:
            return False

        return True

    def detection_is_risky_single_view(self, detection: ViewDetection) -> bool:
        """True si une bbox mono-vue est dangereuse pour une ID stable
        """
        if bool(getattr(detection, "partial_body", False)):
            return True

        if not bool(getattr(detection, "reliable_for_3d", True)):
            return True

        if bool(getattr(detection, "edge_truncated", False)):
            return True

        if int(getattr(detection, "valid_joint_count", 99)) < 8:
            return True

        return False

    def track_has_recent_multiview_support(
        self,
        track: GlobalTrack,
        window: int = 12,
        min_multiview_frames: int = 2,
    ) -> bool:
        """True si la track a récemment été supportée par plusieurs vues
        """
        if len(track.current_detections) >= 2:
            return True

        history = list(track.multi_view_history)[-int(window):]
        if sum(1 for value in history if int(value) >= 2) >= int(min_multiview_frames):
            return True

        return False

    def strong_evidence_for_risky_single_view(
        self,
        same_tracklet: bool,
        motion_score: float,
        appearance_score: float,
    ) -> bool:
        """Preuve minimale pour accepter une mono-vue partielle"""
        return bool(same_tracklet) and float(motion_score) <= 0.25 and float(appearance_score) <= 0.35

    def group_reliable_detections_for_tracking_3d(self, group: DetectionGroup) -> List[ViewDetection]:
        return [
            detection
            for detection in group.detections
            if self.detection_is_reliable_for_tracking_3d(detection)
        ]

    def group_reliable_views_for_tracking_3d(self, group: DetectionGroup) -> List[str]:
        return sorted({
            detection.camera_name
            for detection in self.group_reliable_detections_for_tracking_3d(group)
        })


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
        require_reliable_views: bool = False,
    ) -> Optional[tuple[np.ndarray, float]]:
        if cameras_by_camera is None:
            return None

        projections = []
        points_2d = []
        weights = []
        cameras = []

        for detection in group.detections:
            if require_reliable_views and not self.detection_is_reliable_for_tracking_3d(detection):
                continue

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

        if det_a.feature is not None and det_b.feature is not None:
            appearance_cost = float(
                cdist([det_a.feature], [det_b.feature], metric="cosine")[0, 0]
            )

            if spatial_score > 0:
                return 0.5 * appearance_cost + 0.5 * (1.0 - spatial_score)

            return appearance_cost

        if spatial_score >= self.strong_spatial_score_for_featureless_pair:
            return 1.0 - spatial_score

        return large_cost

    def local_continuity_candidate(
        self,
        detection: ViewDetection,
        max_missing: int = 30,
        max_motion_score: float = 0.25,
        max_appearance_score: float = 0.40,
        max_final_score: float = 0.30,
        ambiguity_margin: float = 0.20,
    ) -> Optional[tuple[int, float]]:
        """Return (global_id, score) only for a very confident local owner
        """
        camera_name = detection.camera_name

        if detection.confidence < self.min_detection_confidence_for_matching:
            return None

        measurement = bbox_to_kalman(detection.bbox_xyxy)
        candidates = []

        for track in self.active_tracks():
            if track.time_since_update > max_missing:
                continue

            if track.state == TrackState.tentative and track.hits < self.min_hits_to_confirm:
                continue

            if camera_name not in track.kf2d_mean_by_camera:
                continue

            mean = track.kf2d_mean_by_camera[camera_name]
            covariance = track.kf2d_cov_by_camera[camera_name]

            maha = self.kf2d.gating_distance(
                mean,
                covariance,
                np.asarray([measurement]),
            )[0]

            gating_multiplier = 4.0 + 0.5 * min(track.time_since_update, 10)
            gating_threshold = chi2inv95[4] * gating_multiplier
            motion_score = float(maha / max(1e-12, gating_threshold))

            if motion_score > max_motion_score:
                continue

            same_tracklet = self.is_same_tracklet_reliable(track, detection)
            app_score = 0.50
            has_feature = detection.feature is not None and bool(track.feature_gallery)

            if not has_feature and not same_tracklet:
                continue

            if has_feature:
                app_score = track.appearance_distance(detection.feature)

                if app_score > max_appearance_score and not same_tracklet:
                    continue

            tracklet_bonus = -0.20 if same_tracklet else 0.0
            score = 0.75 * motion_score + 0.25 * app_score + tracklet_bonus

            candidates.append((score, track.global_id, same_tracklet, motion_score, app_score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_id, same_tracklet, _, _ = candidates[0]

        if len(candidates) > 1:
            second_score = float(candidates[1][0])

            if best_score + ambiguity_margin >= second_score:
                return None

        if not same_tracklet and best_score > max_final_score:
            return None

        return int(best_id), float(best_score)

    def local_continuity_owner(
        self,
        detection: ViewDetection,
        max_missing: int = 30,
        max_motion_score: float = 0.25,
        ambiguity_margin: float = 0.20,
    ) -> Optional[int]:
        candidate = self.local_continuity_candidate(
            detection=detection,
            max_missing=max_missing,
            max_motion_score=max_motion_score,
            ambiguity_margin=ambiguity_margin,
        )

        if candidate is None:
            return None

        return int(candidate[0])

    def build_local_continuity_locks(
        self,
        detections: List[ViewDetection],
        duplicate_margin: float = 0.10,
    ) -> Dict[int, int]:
        """Build dynamic per-detection locks for the current frame
        """
        if not self.local_continuity_lock_enabled:
            return {}

        candidates_by_key: Dict[Tuple[str, int], List[tuple[float, int]]] = {}

        for detection in detections:
            candidate = self.local_continuity_candidate(detection)

            if candidate is None:
                continue

            owner, score = candidate
            key = (detection.camera_name, int(owner))
            candidates_by_key.setdefault(key, []).append((float(score), id(detection)))

        locked_owner_by_detection_id: Dict[int, int] = {}

        for (_camera_name, owner), candidates in candidates_by_key.items():
            candidates.sort(key=lambda item: item[0])

            if len(candidates) > 1:
                best_score = candidates[0][0]
                second_score = candidates[1][0]

                if best_score + duplicate_margin >= second_score:
                    continue

            _score, detection_id = candidates[0]
            locked_owner_by_detection_id[int(detection_id)] = int(owner)

        return locked_owner_by_detection_id

    def locked_owner_set(
        self,
        detections: List[ViewDetection],
        locked_owner_by_detection_id: Dict[int, int],
    ) -> set[int]:
        owners = set()

        for detection in detections:
            owner = locked_owner_by_detection_id.get(id(detection))

            if owner is not None:
                owners.add(int(owner))

        return owners

    def locked_owner_conflict(
        self,
        detections_a: List[ViewDetection],
        detections_b: List[ViewDetection],
        locked_owner_by_detection_id: Dict[int, int],
    ) -> bool:
        owners_a = self.locked_owner_set(detections_a, locked_owner_by_detection_id)
        owners_b = self.locked_owner_set(detections_b, locked_owner_by_detection_id)

        if not owners_a or not owners_b:
            return False

        return len(owners_a.union(owners_b)) > 1

    def detection_compatible_with_locked_owner(
        self,
        detection: ViewDetection,
        owner_global_id: int,
        max_motion_score: float = 0.65,
        max_appearance_score: float = 0.55,
    ) -> bool:
        """Check if an unlocked detection can safely join a locked owner's group
        """
        track = self.tracks.get(int(owner_global_id))

        if track is None or track.state == TrackState.lost:
            return False

        if self.is_same_tracklet_reliable(track, detection):
            return True

        camera_name = detection.camera_name
        motion_checked = False

        if camera_name in track.kf2d_mean_by_camera:
            measurement = bbox_to_kalman(detection.bbox_xyxy)
            mean = track.kf2d_mean_by_camera[camera_name]
            covariance = track.kf2d_cov_by_camera[camera_name]

            maha = self.kf2d.gating_distance(
                mean,
                covariance,
                np.asarray([measurement]),
            )[0]

            gating_multiplier = 6.0 + 0.8 * min(track.time_since_update, 10)
            gating_threshold = chi2inv95[4] * gating_multiplier
            motion_score = float(maha / max(1e-12, gating_threshold))
            motion_checked = True

            if motion_score > max_motion_score:
                return False

        if detection.feature is not None and track.feature_gallery:
            appearance_score = track.appearance_distance(detection.feature)

            if appearance_score > max_appearance_score:
                return False

            return True

        return True if not motion_checked else True

    def locked_owner_merge_allowed(
        self,
        detections_a: List[ViewDetection],
        detections_b: List[ViewDetection],
        locked_owner_by_detection_id: Dict[int, int],
    ) -> bool:
        owners_a = self.locked_owner_set(detections_a, locked_owner_by_detection_id)
        owners_b = self.locked_owner_set(detections_b, locked_owner_by_detection_id)
        owners = owners_a.union(owners_b)

        if len(owners) > 1:
            return False

        if not owners:
            return True

        owner = int(next(iter(owners)))

        for detection in detections_a + detections_b:
            locked_owner = locked_owner_by_detection_id.get(id(detection))

            if locked_owner is not None:
                if int(locked_owner) != owner:
                    return False
                continue

            if not self.detection_compatible_with_locked_owner(detection, owner):
                return False

        return True


    def detection_pose_center(
        self,
        detection: ViewDetection,
        min_joint_confidence: float = 0.25,
        min_valid_joints: int = 4,
    ) -> Optional[np.ndarray]:
        if detection.pose_2d is None or detection.joint_confidences is None:
            return None

        pose = detection.pose_2d
        scores = detection.joint_confidences

        if len(pose) == 0 or len(scores) == 0:
            return None

        valid = (
            np.isfinite(pose[:, 0])
            & np.isfinite(pose[:, 1])
            & (scores > min_joint_confidence)
        )

        if int(np.sum(valid)) < min_valid_joints:
            return None

        return np.mean(pose[valid], axis=0).astype(np.float64)

    def detections_pose_centers_clearly_different(
        self,
        detection_a: ViewDetection,
        detection_b: ViewDetection,
        threshold_ratio: float = 0.35,
    ) -> bool:
        center_a = self.detection_pose_center(detection_a)
        center_b = self.detection_pose_center(detection_b)

        if center_a is None or center_b is None:
            return False

        ax1, ay1, ax2, ay2 = detection_a.bbox_xyxy
        bx1, by1, bx2, by2 = detection_b.bbox_xyxy

        scale = max(
            float(ax2 - ax1),
            float(ay2 - ay1),
            float(bx2 - bx1),
            float(by2 - by1),
            1.0,
        )

        distance = float(np.linalg.norm(center_a - center_b))
        return distance / scale > threshold_ratio

    def track_has_real_detection_in_camera(
        self,
        track: GlobalTrack,
        camera_name: str,
    ) -> bool:
        return camera_name in track.current_detections

    def track_reference_bbox(
        self,
        track: GlobalTrack,
        camera_name: str,
    ) -> Optional[BoxXYXY]:
        if camera_name in track.current_bbox:
            return track.current_bbox[camera_name]

        return self.predicted_bbox_from_track(track, camera_name)

    def detection_duplicate_score_for_track(
        self,
        track: GlobalTrack,
        detection: ViewDetection,
    ) -> Optional[float]:
        """Score whether a detection is probably a duplicate of an existing track
        """
        if track.state == TrackState.lost:
            return None

        camera_name = detection.camera_name
        reference_bbox = self.track_reference_bbox(track, camera_name)

        if reference_bbox is None:
            return None

        current_detection = track.current_detections.get(camera_name)

        if current_detection is not None:
            if self.detections_pose_centers_clearly_different(
                current_detection,
                detection,
            ):
                return None

        iou = bbox_iou(reference_bbox, detection.bbox_xyxy)
        containment = bbox_containment_of_smaller(reference_bbox, detection.bbox_xyxy)
        center_ratio = bbox_center_distance_ratio(reference_bbox, detection.bbox_xyxy)
        area_ratio = bbox_small_area_ratio(reference_bbox, detection.bbox_xyxy)

        app_score: Optional[float] = None
        if detection.feature is not None and track.feature_gallery:
            app_score = track.appearance_distance(detection.feature)

            if app_score > 0.42 and not self.is_same_tracklet_reliable(track, detection):
                return None

        same_tracklet = self.is_same_tracklet_reliable(track, detection)

        motion_score = self.motion_score_for_detection(
            track,
            detection,
            gating_multiplier=7.0,
        )

        evidence_scores: List[float] = []

        if same_tracklet:
            evidence_scores.append(-0.30)

        if app_score is not None and app_score <= 0.28:
            evidence_scores.append(float(app_score))

        if motion_score is not None and motion_score <= 0.85:
            evidence_scores.append(float(motion_score))

        if iou >= 0.65:
            evidence_scores.append(float(1.0 - iou))

        if containment >= 0.84 and center_ratio <= 0.32:
            partial_score = (1.0 - containment) + center_ratio

            if area_ratio < 0.70:
                partial_score *= 0.85

            evidence_scores.append(float(partial_score))

        if (
            app_score is None
            and not same_tracklet
            and iou < 0.65
            and not (containment >= 0.92 and center_ratio <= 0.18)
        ):
            return None

        if not evidence_scores:
            return None

        score = float(np.mean(evidence_scores))

        if current_detection is not None and (iou >= 0.45 or containment >= 0.80):
            score -= 0.10

        return score

    def group_duplicate_score_for_track(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
    ) -> Optional[float]:
        scores = []

        for detection in group.detections:
            score = self.detection_duplicate_score_for_track(track, detection)

            if score is not None:
                scores.append(float(score))

        if not scores:
            return None

        if group.num_views > 1 and len(scores) < max(1, int(np.ceil(len(group.detections) / 2))):
            return None

        return float(np.mean(scores))

    def find_duplicate_owner_for_unmatched_group(
        self,
        group: DetectionGroup,
        max_score: float = 0.62,
        ambiguity_margin: float = 0.12,
    ) -> Optional[tuple[GlobalTrack, float]]:
        candidates = []

        for track in self.active_tracks():
            if track.state == TrackState.lost:
                continue

            if self.group_has_foreign_recent_tracklet(track, group):
                continue

            score = self.group_duplicate_score_for_track(track, group)

            if score is None:
                continue

            if score <= max_score:
                candidates.append((float(score), track))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_track = candidates[0]

        if len(candidates) > 1:
            second_score = float(candidates[1][0])

            if best_score + ambiguity_margin >= second_score:
                self.debug_writer.write(
                    frame_id=self.current_frame_id,
                    track_id=best_track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=0.0,
                    motion_cost=0.0,
                    spatial_cost=0.0,
                    total_cost=best_score,
                    decision="rejected",
                    reason="duplicate_owner_ambiguous",
                )
                return None

        return best_track, best_score

    def duplicate_absorption_allowed_for_group(
        self,
        group: DetectionGroup,
    ) -> bool:
        """Return True when the duplicate/occlusion absorber is safe
        """
        if group.num_views <= 1:
            return True

        return self.current_frame_multiview_group_count <= 1

    def try_absorb_duplicate_unmatched_group(
        self,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[GlobalTrack]:
        """Handle unmatched groups that are likely duplicate/partial boxes
        """
        if not self.duplicate_absorption_allowed_for_group(group):
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=-1,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=0.0,
                motion_cost=0.0,
                spatial_cost=0.0,
                total_cost=0.0,
                decision="rejected",
                reason="duplicate_absorb_skipped_crowded_multiview",
            )
            return None

        candidate = self.find_duplicate_owner_for_unmatched_group(group)

        if candidate is None:
            return None

        owner, score = candidate

        owner_has_real_detection_in_group_camera = any(
            self.track_has_real_detection_in_camera(owner, detection.camera_name)
            for detection in group.detections
        )

        if owner_has_real_detection_in_group_camera:
            added = self.complete_missing_views_from_duplicate_group(
                owner,
                group,
                frame_id,
            )

            self.debug_writer.write(
                frame_id=frame_id,
                track_id=owner.global_id,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=0.0,
                motion_cost=0.0,
                spatial_cost=0.0,
                total_cost=score,
                decision="accepted",
                reason=(
                    "duplicate_group_completed_missing_views"
                    if added > 0
                    else "duplicate_group_suppressed_existing_track"
                ),
            )
            return owner

        if self.current_frame_multiview_group_count >= 2:
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=owner.global_id,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=0.0,
                motion_cost=0.0,
                spatial_cost=0.0,
                total_cost=score,
                decision="rejected",
                reason="duplicate_absorb_update_skipped_crowded",
            )
            return None

        self.update_track_with_group(
            owner,
            group,
            frame_id,
            cameras_by_camera,
        )

        self.debug_writer.write(
            frame_id=frame_id,
            track_id=owner.global_id,
            group_index=-1,
            group_views=group.camera_names,
            app_cost=0.0,
            motion_cost=0.0,
            spatial_cost=0.0,
            total_cost=score,
            decision="accepted",
            reason="duplicate_group_reassociated_existing_track",
        )

        print(
            f"[tracking] groupe doublon/occlusion rattaché "
            f"track={owner.global_id} vues={group.camera_names} score={score:.3f}"
        )

        return owner

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

        locked_owner_by_detection_id = self.build_local_continuity_locks(detections)

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

                    detections_a = [detections[index] for index in clusters[key_a]]
                    detections_b = [detections[index] for index in clusters[key_b]]

                    group_a = DetectionGroup(detections_a)
                    group_b = DetectionGroup(detections_b)

                    if not group_a.camera_set.isdisjoint(group_b.camera_set):
                        continue

                    if not self.locked_owner_merge_allowed(
                        detections_a,
                        detections_b,
                        locked_owner_by_detection_id,
                    ):
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

        return self.safety_merge_close_groups(
            groups,
            cameras_by_camera,
            locked_owner_by_detection_id,
        )

    def safety_merge_close_groups(
        self,
        groups: List[DetectionGroup],
        cameras_by_camera: Optional[Dict[str, CameraData]],
        locked_owner_by_detection_id: Optional[Dict[int, int]] = None,
    ) -> List[DetectionGroup]:
        if len(groups) <= 1:
            return groups

        if locked_owner_by_detection_id is None:
            locked_owner_by_detection_id = {}

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

                    if not self.locked_owner_merge_allowed(
                        group_a.detections,
                        group_b.detections,
                        locked_owner_by_detection_id,
                    ):
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

        recent_view_pattern = (
            list(track.multi_view_history)[-5:]
            if track.multi_view_history
            else []
        )

        is_transitioning = (
            len(set(recent_view_pattern)) > 1
            if recent_view_pattern
            else False
        )

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


    def group_local_continuity_conflict(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
        max_missing: int = 25,
        max_motion_score: float = 0.38,
        max_appearance_score: float = 0.48,
        max_final_score: float = 0.42,
        ambiguity_margin: float = 0.12,
    ) -> Optional[tuple[int, float]]:
        """Detect when a group is locally owned by another existing track
        """
        own_support = 0
        foreign_support: Dict[int, float] = {}

        for detection in group.detections:
            candidate = self.local_continuity_candidate(
                detection=detection,
                max_missing=max_missing,
                max_motion_score=max_motion_score,
                max_appearance_score=max_appearance_score,
                max_final_score=max_final_score,
                ambiguity_margin=ambiguity_margin,
            )

            if candidate is None:
                continue

            owner_id, owner_score = candidate

            if int(owner_id) == int(track.global_id):
                own_support += 1
                continue

            current_best = foreign_support.get(int(owner_id))
            if current_best is None or owner_score < current_best:
                foreign_support[int(owner_id)] = float(owner_score)

        if not foreign_support:
            return None

        if own_support > 0:
            return None

        foreign_owner, foreign_score = min(
            foreign_support.items(),
            key=lambda item: item[1],
        )

        return int(foreign_owner), float(foreign_score)


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

                (
                    w_app,
                    w_motion,
                    w_spatial,
                    threshold,
                    gating_multiplier,
                ) = self.adaptive_weights(track, group, group_3d_info)

                app_cost = float(appearance_cost[track_index, group_index])
                motion_cost = large_cost
                spatial_cost = large_cost
                total_cost = large_cost

                if self.has_tracklet_conflict(track, group):
                    self.debug_writer.write(
                        frame_id=self.current_frame_id,
                        track_id=track.global_id,
                        group_index=group_index,
                        group_views=group.camera_names,
                        app_cost=app_cost,
                        motion_cost=motion_cost,
                        spatial_cost=spatial_cost,
                        total_cost=total_cost,
                        decision="rejected",
                        reason="tracklet_conflict",
                    )
                    continue

                local_owner_conflict = self.group_local_continuity_conflict(
                    track,
                    group,
                )

                if local_owner_conflict is not None:
                    owner_id, owner_score = local_owner_conflict
                    self.debug_writer.write(
                        frame_id=self.current_frame_id,
                        track_id=track.global_id,
                        group_index=group_index,
                        group_views=group.camera_names,
                        app_cost=app_cost,
                        motion_cost=owner_score,
                        spatial_cost=0.0,
                        total_cost=large_cost,
                        decision="rejected",
                        reason=f"local_continuity_owner_conflict_{owner_id}",
                    )
                    continue

                motion_cost = self.motion_cost_for_group(
                    track,
                    group,
                    gating_multiplier,
                )

                if motion_cost >= large_cost:
                    self.debug_writer.write(
                        frame_id=self.current_frame_id,
                        track_id=track.global_id,
                        group_index=group_index,
                        group_views=group.camera_names,
                        app_cost=app_cost,
                        motion_cost=motion_cost,
                        spatial_cost=spatial_cost,
                        total_cost=total_cost,
                        decision="rejected",
                        reason="motion_gate",
                    )
                    continue

                reliable_tracklets = 0

                for detection in group.detections:
                    if self.is_same_tracklet_reliable(track, detection):
                        reliable_tracklets += 1

                spatial_cost = self.spatial_cost_for_group(
                    track,
                    group_3d_info,
                    gating_multiplier,
                )

                spatial_soft_fallback = False

                if spatial_cost >= large_cost:
                    strong_2d_reid = (
                        group.num_views >= 2
                        and app_cost <= 0.12
                        and motion_cost <= 0.25
                    )
                    strong_tracklet_motion = (
                        group.num_views >= 2
                        and reliable_tracklets > 0
                        and motion_cost <= 0.40
                        and app_cost <= 0.40
                    )

                    if strong_2d_reid or strong_tracklet_motion:
                        spatial_cost = 0.75
                        spatial_soft_fallback = True
                    else:
                        self.debug_writer.write(
                            frame_id=self.current_frame_id,
                            track_id=track.global_id,
                            group_index=group_index,
                            group_views=group.camera_names,
                            app_cost=app_cost,
                            motion_cost=motion_cost,
                            spatial_cost=spatial_cost,
                            total_cost=total_cost,
                            decision="rejected",
                            reason="spatial_gate",
                        )
                        continue

                total_cost = (
                    w_app * app_cost
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
                            if np.linalg.norm(group_pos - other_pos) < np.linalg.norm(
                                group_pos - track.mean_3d[:3]
                            ):
                                proximity_penalty = max(proximity_penalty, 0.2)

                    total_cost += proximity_penalty

                if reliable_tracklets > 0:
                    total_cost *= 0.85

                if spatial_soft_fallback:
                    total_cost *= 0.95

                if group.num_views == 1 and group.detections:
                    det0 = group.detections[0]
                    if (
                        self.detection_is_risky_single_view(det0)
                        and self.track_has_recent_multiview_support(track)
                    ):
                        strong_partial_single_view = self.strong_evidence_for_risky_single_view(
                            same_tracklet=reliable_tracklets > 0,
                            motion_score=motion_cost,
                            appearance_score=app_cost,
                        )
                        if not strong_partial_single_view:
                            self.debug_writer.write(
                                frame_id=self.current_frame_id,
                                track_id=track.global_id,
                                group_index=group_index,
                                group_views=group.camera_names,
                                app_cost=app_cost,
                                motion_cost=motion_cost,
                                spatial_cost=spatial_cost,
                                total_cost=large_cost,
                                decision="rejected",
                                reason="single_view_rejected_partial_unreliable",
                            )
                            continue

                        total_cost += 0.30

                    elif (
                        self.track_has_recent_multiview_support(track)
                        and reliable_tracklets == 0
                        and motion_cost > 0.45
                        and app_cost > 0.50
                    ):
                        self.debug_writer.write(
                            frame_id=self.current_frame_id,
                            track_id=track.global_id,
                            group_index=group_index,
                            group_views=group.camera_names,
                            app_cost=app_cost,
                            motion_cost=motion_cost,
                            spatial_cost=spatial_cost,
                            total_cost=large_cost,
                            decision="rejected",
                            reason="single_view_rejected_multiview_history",
                        )
                        continue

                    if self.camera_is_crowded(det0.camera_name):
                        strong_crowded_single_view = (
                            reliable_tracklets > 0
                            or (
                                motion_cost <= self.crowded_single_view_strict_motion
                                and app_cost <= self.crowded_single_view_strict_appearance
                            )
                        )

                        if not strong_crowded_single_view:
                            self.debug_writer.write(
                                frame_id=self.current_frame_id,
                                track_id=track.global_id,
                                group_index=group_index,
                                group_views=group.camera_names,
                                app_cost=app_cost,
                                motion_cost=motion_cost,
                                spatial_cost=spatial_cost,
                                total_cost=large_cost,
                                decision="rejected",
                                reason="single_view_rejected_crowded_camera",
                            )
                            continue

                        if reliable_tracklets == 0:
                            total_cost += self.crowded_single_view_penalty

                if total_cost < threshold:
                    cost_matrix[track_index, group_index] = float(total_cost)
                    decision = "candidate"
                    reason = "candidate"
                else:
                    decision = "rejected"
                    reason = "above_threshold"

                self.debug_writer.write(
                    frame_id=self.current_frame_id,
                    track_id=track.global_id,
                    group_index=group_index,
                    group_views=group.camera_names,
                    app_cost=app_cost,
                    motion_cost=motion_cost,
                    spatial_cost=spatial_cost,
                    total_cost=total_cost,
                    decision=decision,
                    reason=reason,
                )

        return cost_matrix
    def filter_ambiguous_assignments(
        self,
        matches: List[Tuple[int, int]],
        cost_matrix: np.ndarray,
    ) -> List[Tuple[int, int]]:
        if not matches:
            return []

        filtered = []

        for track_index, group_index in matches:
            best_cost = float(cost_matrix[track_index, group_index])

            row = cost_matrix[track_index].copy()
            row[group_index] = large_cost
            second_track_cost = float(np.min(row)) if row.size else large_cost

            col = cost_matrix[:, group_index].copy()
            col[track_index] = large_cost
            second_group_cost = float(np.min(col)) if col.size else large_cost

            second_best = min(second_track_cost, second_group_cost)

            if second_best < large_cost and best_cost + self.association_margin > second_best:
                self.debug_writer.write(
                    frame_id=self.current_frame_id,
                    track_id=-1,
                    group_index=group_index,
                    group_views=[],
                    app_cost=0.0,
                    motion_cost=0.0,
                    spatial_cost=0.0,
                    total_cost=best_cost,
                    decision="rejected",
                    reason="ambiguous_margin",
                )
                continue

            filtered.append((track_index, group_index))

        return filtered

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

        matches = self.filter_ambiguous_assignments(matches, cost)

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
    ) -> Tuple[bool, np.ndarray]:
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
            track.last_seen_frame_by_camera[camera_name] = frame_id

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
            np.asarray(
                [detection.bbox_xyxy for detection in group.detections],
                dtype=np.float64,
            ),
            axis=0,
        )

        width = max(1.0, avg_box[2] - avg_box[0])
        height = max(1.0, avg_box[3] - avg_box[1])
        track.aspect_ratio_history.append(float(width / height))

        reliable_views_for_tracking_3d = self.group_reliable_views_for_tracking_3d(group)
        result_3d = self.triangulate_group(
            group,
            cameras_by_camera,
            require_reliable_views=self.require_reliable_views_for_tracking_3d,
        )

        if result_3d is None and group.num_views >= 2:
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=track.global_id,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=0.0,
                motion_cost=0.0,
                spatial_cost=0.0,
                total_cost=0.0,
                decision="skipped",
                reason=(
                    "tracking_3d_not_enough_reliable_views"
                    if len(reliable_views_for_tracking_3d) < 2
                    else "tracking_3d_failed_reliable_triangulation"
                ),
            )

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
                track.tracking_3d_updated_this_frame = True
                track.tracking_3d_views_used = list(reliable_views_for_tracking_3d)
                track.position_3d_history.append(point_3d.copy())
    def add_real_detection_to_existing_track(
        self,
        track: GlobalTrack,
        detection: ViewDetection,
        frame_id: int,
    ) -> None:
        """Ajoute une vraie détection dans une caméra manquante
        """
        camera_name = detection.camera_name

        track.current_detections[camera_name] = detection
        track.current_bbox[camera_name] = detection.bbox_xyxy
        track.last_seen_frame_by_camera[camera_name] = frame_id

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

        track.update_feature(detection.feature, self.feature_update_weight)

    def detection_can_complete_track_view(
        self,
        track: GlobalTrack,
        detection: ViewDetection,
        max_motion_score: float = 0.95,
        max_appearance_score: float = 0.55,
    ) -> bool:
        """Vérifie si une détection peut compléter une caméra manquante
        """
        if detection.confidence < self.min_detection_confidence_for_matching:
            return False

        group = DetectionGroup([detection])
        if self.group_has_foreign_recent_tracklet(track, group):
            return False

        local_conflict = self.group_local_continuity_conflict(
            track,
            group,
            max_missing=25,
            max_motion_score=0.45,
            max_appearance_score=0.52,
            max_final_score=0.48,
            ambiguity_margin=0.10,
        )
        if local_conflict is not None:
            return False

        if (
            self.detection_is_risky_single_view(detection)
            and self.track_has_recent_multiview_support(track)
            and not self.is_same_tracklet_reliable(track, detection)
        ):
            return False

        same_tracklet = self.is_same_tracklet_reliable(track, detection)
        motion = self.motion_score_for_detection(
            track,
            detection,
            gating_multiplier=7.0,
        )

        has_motion = motion is not None
        good_motion = has_motion and float(motion) <= max_motion_score

        has_app = detection.feature is not None and bool(track.feature_gallery)
        good_app = False
        if has_app:
            app = track.appearance_distance(detection.feature)
            if app > max_appearance_score and not same_tracklet:
                return False
            good_app = app <= max_appearance_score

        if same_tracklet or good_motion or good_app:
            return True

        if detection.camera_name not in track.kf2d_mean_by_camera:
            return False

        return False

    def complete_missing_views_from_duplicate_group(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
        frame_id: int,
    ) -> int:
        """Complète une track déjà matchée avec les vues réelles manquantes
        """
        added = 0

        for detection in group.detections:
            camera_name = detection.camera_name

            if camera_name in track.current_detections:
                continue

            if not self.detection_can_complete_track_view(track, detection):
                continue

            self.add_real_detection_to_existing_track(
                track=track,
                detection=detection,
                frame_id=frame_id,
            )
            added += 1

        return added

    def group_has_foreign_recent_tracklet(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
    ) -> bool:
        """True when a local ByteTrack id was recently owned by another global id
        """
        for detection in group.detections:
            if detection.tracklet_id is None:
                continue

            owner = self.recent_tracklet_owner(
                detection.camera_name,
                detection.tracklet_id,
                exclude_global_id=None,
            )

            if owner is not None and owner != track.global_id:
                return True

        return False

    def motion_score_for_detection(
        self,
        track: GlobalTrack,
        detection: ViewDetection,
        gating_multiplier: float = 5.0,
    ) -> Optional[float]:
        camera_name = detection.camera_name

        if camera_name not in track.kf2d_mean_by_camera:
            return None

        measurement = bbox_to_kalman(detection.bbox_xyxy)
        mean = track.kf2d_mean_by_camera[camera_name]
        covariance = track.kf2d_cov_by_camera[camera_name]

        maha = self.kf2d.gating_distance(
            mean,
            covariance,
            np.asarray([measurement]),
        )[0]

        missing_relax = 0.5 * min(track.time_since_update, 10)
        threshold = chi2inv95[4] * (gating_multiplier + missing_relax)

        return float(maha / max(1e-12, threshold))

    def current_camera_detections(
        self,
        camera_name: str,
    ) -> List[ViewDetection]:
        return list(self.current_frame_detections_by_camera.get(camera_name, []))

    def detection_occluder_candidates(
        self,
        detection: ViewDetection,
    ) -> List[tuple[ViewDetection, float, float, float, float]]:
        """Return large overlapping detections in the same camera
        """
        if not self.partial_occlusion_handling_enabled:
            return []

        area = bbox_area(detection.bbox_xyxy)
        if area <= 1.0:
            return []

        candidates: List[tuple[ViewDetection, float, float, float, float]] = []

        for other in self.current_camera_detections(detection.camera_name):
            if other is detection:
                continue

            other_area = bbox_area(other.bbox_xyxy)
            if other_area <= area:
                continue

            area_ratio = float(other_area / max(1.0, area))
            if area_ratio < self.partial_occlusion_larger_area_ratio:
                continue

            iou = bbox_iou(detection.bbox_xyxy, other.bbox_xyxy)
            containment = bbox_containment_of_smaller(detection.bbox_xyxy, other.bbox_xyxy)
            center_ratio = bbox_center_distance_ratio(detection.bbox_xyxy, other.bbox_xyxy)

            geometry_overlap = (
                iou >= self.partial_occlusion_min_iou
                or containment >= self.partial_occlusion_min_containment
            )

            if not geometry_overlap:
                continue

            if center_ratio > self.partial_occlusion_max_center_ratio:
                continue

            candidates.append((other, area_ratio, iou, containment, center_ratio))

        candidates.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        return candidates

    def detection_is_partial_occlusion_candidate(
        self,
        detection: ViewDetection,
    ) -> bool:
        """True for partial/intermittent detections behind a larger bbox
        """
        candidates = self.detection_occluder_candidates(detection)
        if not candidates:
            return False

        if detection.joint_confidences is not None and len(detection.joint_confidences) > 0:
            visibility = float(np.mean(detection.joint_confidences > 0.25))
            if visibility <= self.partial_occlusion_visibility_threshold:
                return True

        _other, area_ratio, iou, containment, _center_ratio = candidates[0]
        if area_ratio >= 2.20:
            return True

        if containment >= 0.48 or iou >= 0.22:
            return True

        return False

    def group_partial_occlusion_score(
        self,
        group: DetectionGroup,
    ) -> float:
        if not group.detections:
            return 0.0

        hits = 0
        for detection in group.detections:
            if self.detection_is_partial_occlusion_candidate(detection):
                hits += 1

        return float(hits / max(1, len(group.detections)))

    def group_is_partial_occlusion_candidate(
        self,
        group: DetectionGroup,
    ) -> bool:
        if not self.partial_occlusion_handling_enabled:
            return False

        if not group.detections:
            return False

        score = self.group_partial_occlusion_score(group)

        if group.num_views <= 1:
            return score > 0.0

        return score >= 0.75 and group.visibility_ratio < 0.55

    def track_has_real_larger_overlap_with_group(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
    ) -> bool:
        """True when track currently has a real larger bbox over this group
        """
        for detection in group.detections:
            camera_name = detection.camera_name
            if camera_name not in track.current_detections:
                continue

            track_box = track.current_bbox.get(camera_name)
            if track_box is None:
                continue

            track_area = bbox_area(track_box)
            det_area = bbox_area(detection.bbox_xyxy)
            if track_area < det_area * self.partial_occlusion_larger_area_ratio:
                continue

            iou = bbox_iou(track_box, detection.bbox_xyxy)
            containment = bbox_containment_of_smaller(track_box, detection.bbox_xyxy)
            center_ratio = bbox_center_distance_ratio(track_box, detection.bbox_xyxy)

            if (
                iou >= self.partial_occlusion_min_iou
                or containment >= self.partial_occlusion_min_containment
            ) and center_ratio <= self.partial_occlusion_max_center_ratio:
                return True

        return False

    def should_defer_partial_occlusion_new_track(
        self,
        group: DetectionGroup,
    ) -> bool:
        """Avoid creating a fresh ID from a risky partial occlusion group.
        """
        if not self.group_is_partial_occlusion_candidate(group):
            return False

        if group.num_views >= 2 and group.mean_confidence >= self.min_detection_confidence_for_new_track_multiview:
            return False

        stable_tracks = [
            track for track in self.active_tracks()
            if track.state in {TrackState.confirmed_2d, TrackState.confirmed_3d, TrackState.occluded}
            and track.hits >= self.min_hits_to_confirm
        ]
        if not stable_tracks:
            return False

        if group.mean_confidence >= self.partial_occlusion_new_track_min_confidence:
            return False

        self.debug_writer.write(
            frame_id=self.current_frame_id,
            track_id=-1,
            group_index=-1,
            group_views=group.camera_names,
            app_cost=0.0,
            motion_cost=0.0,
            spatial_cost=0.0,
            total_cost=float(group.mean_confidence),
            decision="deferred",
            reason="new_track_deferred_partial_occlusion",
        )
        return True

    def try_reassociate_partial_occlusion_group(
        self,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[GlobalTrack]:
        """Recover an intermittent person behind an occluder without new IDs.
        """
        if not self.group_is_partial_occlusion_candidate(group):
            return None

        if group.mean_confidence < self.min_detection_confidence_for_matching:
            return None

        feature = group.mean_feature()
        group_3d = self.triangulate_group(group, cameras_by_camera)
        candidates = []

        for track in self.active_tracks():
            if track.state == TrackState.lost:
                continue

            if track.time_since_update == 0:
                continue

            if track.time_since_update > self.single_view_reassociation_max_missing:
                continue

            if self.track_has_real_larger_overlap_with_group(track, group):
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=1.0,
                    motion_cost=1.0,
                    spatial_cost=1.0,
                    total_cost=large_cost,
                    decision="rejected",
                    reason="partial_occlusion_front_occluder_reject",
                )
                continue

            if self.group_has_foreign_recent_tracklet(track, group):
                continue

            if self.group_local_continuity_conflict(track, group) is not None:
                continue

            if (
                self.group_is_partial_occlusion_candidate(group)
                and self.track_has_real_larger_overlap_with_group(track, group)
            ):
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=1.0,
                    motion_cost=1.0,
                    spatial_cost=1.0,
                    total_cost=large_cost,
                    decision="rejected",
                    reason="unmatched_partial_front_occluder_reject",
                )
                continue

            same_tracklet_count = 0
            motion_scores = []

            for detection in group.detections:
                if self.is_same_tracklet_reliable(track, detection):
                    same_tracklet_count += 1

                motion = self.motion_score_for_detection(
                    track,
                    detection,
                    gating_multiplier=8.0,
                )
                if motion is not None:
                    motion_scores.append(float(motion))

            has_motion = bool(motion_scores)
            motion_score = float(np.mean(motion_scores)) if motion_scores else 0.80
            strong_motion = has_motion and motion_score <= 0.70

            appearance_score = 0.60
            has_app = feature is not None and bool(track.feature_gallery)
            strong_app = False
            if has_app:
                appearance_score = track.appearance_distance(feature)
                strong_app = appearance_score <= 0.42

                if same_tracklet_count == 0 and appearance_score > 0.58:
                    continue

            spatial_score = 0.65
            strong_spatial = False
            if group_3d is not None and track.has_3d:
                distance_3d = np.linalg.norm(group_3d[0] - track.mean_3d[:3])
                threshold_3d = self.occlusion_reid_threshold_mm + 70.0 * min(track.time_since_update, 20)
                spatial_score = float(distance_3d / max(1e-12, threshold_3d))
                strong_spatial = spatial_score <= 0.65

            has_proof = same_tracklet_count > 0 or strong_motion or strong_app or strong_spatial
            if not has_proof:
                continue

            if strong_app and not (same_tracklet_count > 0 or strong_motion or strong_spatial):
                active_count = len([t for t in self.active_tracks() if t.state != TrackState.lost])
                if active_count >= 3:
                    continue

            tracklet_bonus = -0.35 if same_tracklet_count > 0 else 0.0
            score = 0.40 * motion_score + 0.35 * appearance_score + 0.25 * spatial_score + tracklet_bonus
            candidates.append((score, track, motion_score, appearance_score, spatial_score, same_tracklet_count))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_track, best_motion, best_app, best_spatial, best_tracklets = candidates[0]

        max_score = 0.72 if best_tracklets > 0 else 0.58
        if best_score > max_score:
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=best_track.global_id,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=best_app,
                motion_cost=best_motion,
                spatial_cost=best_spatial,
                total_cost=best_score,
                decision="rejected",
                reason="partial_occlusion_reassociate_score_too_high",
            )
            return None

        if len(candidates) > 1:
            second_score = float(candidates[1][0])
            if best_score + 0.16 >= second_score:
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=best_track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=best_app,
                    motion_cost=best_motion,
                    spatial_cost=best_spatial,
                    total_cost=best_score,
                    decision="rejected",
                    reason="partial_occlusion_reassociate_ambiguous",
                )
                return None

        self.update_track_with_group(best_track, group, frame_id, cameras_by_camera)
        self.debug_writer.write(
            frame_id=frame_id,
            track_id=best_track.global_id,
            group_index=-1,
            group_views=group.camera_names,
            app_cost=best_app,
            motion_cost=best_motion,
            spatial_cost=best_spatial,
            total_cost=best_score,
            decision="accepted",
            reason="partial_occlusion_reassociated",
        )
        print(
            f"[tracking] réassociation occlusion partielle "
            f"track={best_track.global_id} vues={group.camera_names} score={best_score:.3f}"
        )
        return best_track

    def try_attach_missing_single_view_to_matched_track(
        self,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[GlobalTrack]:
        """Attache une vue mono-caméra à une track déjà matchée ailleurs
        """
        if group.num_views != 1 or not group.detections:
            return None

        detection = group.detections[0]
        camera_name = detection.camera_name

        if detection.confidence < self.min_detection_confidence_for_matching:
            return None

        candidates = []

        for track in self.stable_active_tracks():
            if track.state == TrackState.lost:
                continue

            if track.time_since_update != 0:
                continue

            if camera_name in track.current_detections:
                continue

            if len(track.current_detections) == 0:
                continue

            if camera_name not in track.kf2d_mean_by_camera:
                continue

            if not self.detection_can_complete_track_view(
                track,
                detection,
                max_motion_score=0.90,
                max_appearance_score=0.50,
            ):
                continue

            motion = self.motion_score_for_detection(
                track,
                detection,
                gating_multiplier=7.0,
            )
            motion_score = 0.80 if motion is None else float(motion)

            app_score = 0.55
            if detection.feature is not None and track.feature_gallery:
                app_score = track.appearance_distance(detection.feature)

            same_tracklet = self.is_same_tracklet_reliable(track, detection)
            tracklet_bonus = -0.35 if same_tracklet else 0.0

            score = 0.55 * motion_score + 0.45 * app_score + tracklet_bonus
            candidates.append((float(score), track, motion_score, app_score, same_tracklet))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_track, best_motion, best_app, best_same_tracklet = candidates[0]

        crowded_camera = self.camera_is_crowded(camera_name)
        if crowded_camera:
            strong_crowded_evidence = bool(best_same_tracklet) or (
                float(best_motion) <= self.crowded_single_view_strict_motion
                and float(best_app) <= self.crowded_single_view_strict_appearance
            )
            if not strong_crowded_evidence:
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=best_track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=best_app,
                    motion_cost=best_motion,
                    spatial_cost=0.0,
                    total_cost=best_score,
                    decision="rejected",
                    reason="missing_view_attach_rejected_crowded_camera",
                )
                return None

        if (
            self.detection_is_risky_single_view(detection)
            and self.track_has_recent_multiview_support(best_track)
        ):
            strong_partial_attach = self.strong_evidence_for_risky_single_view(
                same_tracklet=best_same_tracklet,
                motion_score=best_motion,
                appearance_score=best_app,
            )
            if not strong_partial_attach:
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=best_track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=best_app,
                    motion_cost=best_motion,
                    spatial_cost=0.0,
                    total_cost=best_score,
                    decision="rejected",
                    reason="missing_view_attach_rejected_partial_unreliable",
                )
                return None

        max_score = 0.62 if best_same_tracklet else (0.42 if crowded_camera else 0.52)
        if best_score > max_score:
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=best_track.global_id,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=best_app,
                motion_cost=best_motion,
                spatial_cost=0.0,
                total_cost=best_score,
                decision="rejected",
                reason="missing_view_attach_score_too_high",
            )
            return None

        if len(candidates) > 1:
            second_score = float(candidates[1][0])
            ambiguity_margin = 0.22 if crowded_camera else 0.14
            if best_score + ambiguity_margin >= second_score:
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=best_track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=best_app,
                    motion_cost=best_motion,
                    spatial_cost=0.0,
                    total_cost=best_score,
                    decision="rejected",
                    reason=(
                        "missing_view_attach_rejected_crowded_ambiguous"
                        if crowded_camera
                        else "missing_view_attach_ambiguous"
                    ),
                )
                return None

        self.add_real_detection_to_existing_track(
            track=best_track,
            detection=detection,
            frame_id=frame_id,
        )
        self.debug_writer.write(
            frame_id=frame_id,
            track_id=best_track.global_id,
            group_index=-1,
            group_views=group.camera_names,
            app_cost=best_app,
            motion_cost=best_motion,
            spatial_cost=0.0,
            total_cost=best_score,
            decision="accepted",
            reason="missing_single_view_attached_to_matched_track",
        )
        print(
            f"[tracking] vue manquante attachée "
            f"track={best_track.global_id} cam={camera_name} score={best_score:.3f}"
        )
        return best_track

    def should_defer_crowded_single_view_new_track(
        self,
        group: DetectionGroup,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> bool:
        """Évite de confirmer trop vite des IDs mono-vue parasites
        """
        if group.num_views != 1:
            return False

        if self.current_frame_id < 20:
            return False

        stable_tracks = self.stable_active_tracks()
        if len(stable_tracks) < 2:
            return False

        if self.group_is_partial_occlusion_candidate(group):
            self.debug_writer.write(
                frame_id=self.current_frame_id,
                track_id=-1,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=0.0,
                motion_cost=0.0,
                spatial_cost=0.0,
                total_cost=float(group.mean_confidence),
                decision="deferred",
                reason="new_track_deferred_crowded_single_view_partial",
            )
            return True

        candidates = []
        feature = group.mean_feature()

        for track in stable_tracks:
            scores = []

            if feature is not None and track.feature_gallery:
                app = track.appearance_distance(feature)
                if app < 0.52:
                    scores.append(float(app))

            for detection in group.detections:
                motion = self.motion_score_for_detection(
                    track,
                    detection,
                    gating_multiplier=8.0,
                )
                if motion is not None and motion < 1.05:
                    scores.append(float(motion))

                if self.is_same_tracklet_reliable(track, detection):
                    scores.append(-0.25)

            if scores:
                candidates.append((float(np.mean(scores)), track.global_id))

        if not candidates:
            return False

        candidates.sort(key=lambda item: item[0])
        best_score, best_id = candidates[0]

        if best_score > 0.80:
            return False

        self.debug_writer.write(
            frame_id=self.current_frame_id,
            track_id=int(best_id),
            group_index=-1,
            group_views=group.camera_names,
            app_cost=0.0,
            motion_cost=0.0,
            spatial_cost=0.0,
            total_cost=float(best_score),
            decision="deferred",
            reason="new_track_deferred_crowded_single_view_possible_existing",
        )
        return True

    def group_possible_recent_track_score(
        self,
        track: GlobalTrack,
        group: DetectionGroup,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[float]:
        """Score whether an unmatched group may still be an existing track
        """
        if track.state == TrackState.lost:
            return None

        if track.time_since_update > self.single_view_reassociation_max_missing:
            return None

        if self.group_has_foreign_recent_tracklet(track, group):
            return None

        scores = []

        feature = group.mean_feature()
        if feature is not None and track.feature_gallery:
            app = track.appearance_distance(feature)
            if app < 0.60:
                scores.append(float(app))

        motion_scores = []
        for detection in group.detections:
            motion = self.motion_score_for_detection(
                track,
                detection,
                gating_multiplier=7.0,
            )
            if motion is not None:
                motion_scores.append(float(motion))

        if motion_scores:
            motion = float(np.mean(motion_scores))
            if motion < 1.25:
                scores.append(motion)

        group_3d = self.triangulate_group(group, cameras_by_camera)
        if group_3d is not None and track.has_3d:
            distance_3d = np.linalg.norm(group_3d[0] - track.mean_3d[:3])
            threshold_3d = self.occlusion_reid_threshold_mm + 80.0 * min(
                track.time_since_update,
                20,
            )
            spatial = float(distance_3d / max(1e-12, threshold_3d))
            if spatial < 1.20:
                scores.append(spatial)

        for detection in group.detections:
            if self.is_same_tracklet_reliable(track, detection):
                scores.append(-0.25)
                break

        if not scores:
            return None

        return float(np.mean(scores))

    def group_may_belong_to_recent_track(
        self,
        group: DetectionGroup,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> bool:
        """Return True when a new ID would likely be a duplicate
        """
        if not self.defer_new_track_if_recent_track_possible:
            return False

        candidates = []

        for track in self.active_tracks():
            if track.time_since_update == 0:
                continue

            score = self.group_possible_recent_track_score(
                track,
                group,
                cameras_by_camera,
            )

            if score is not None:
                candidates.append((score, track.global_id))

        if not candidates:
            return False

        candidates.sort(key=lambda item: item[0])
        best_score, _best_id = candidates[0]

        return bool(best_score < 0.95)

    def predicted_bbox_from_track(
        self,
        track: GlobalTrack,
        camera_name: str,
    ) -> Optional[BoxXYXY]:
        if camera_name not in track.kf2d_mean_by_camera:
            return None

        bbox = kalman_to_bbox(track.kf2d_mean_by_camera[camera_name])

        if not np.isfinite(bbox).all():
            return None

        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            return None

        return (
            int(round(float(x1))),
            int(round(float(y1))),
            int(round(float(x2))),
            int(round(float(y2))),
        )

    def fill_predicted_display_for_track(
        self,
        track: GlobalTrack,
        frame_id: int,
    ) -> None:
        """Keep the same ID visible briefly using Kalman-predicted bboxes
        """
        if not self.predicted_display_enabled:
            return

        if track.state == TrackState.lost:
            return

        if track.time_since_update <= 0:
            return

        if track.time_since_update > self.predicted_display_max_missing:
            return

        for camera_name in self.camera_names:
            if camera_name in track.current_bbox:
                continue

            last_seen = track.last_seen_frame_by_camera.get(camera_name)
            if last_seen is None:
                continue

            if frame_id - int(last_seen) > self.predicted_display_max_missing:
                continue

            predicted = self.predicted_bbox_from_track(track, camera_name)
            if predicted is None:
                continue

            track.current_bbox[camera_name] = predicted

    def fill_missing_camera_predictions_for_matched_track(
        self,
        track: GlobalTrack,
        frame_id: int,
    ) -> None:
        """Affichage Kalman court pour une caméra manquante seulement
        """
        if not self.per_camera_predicted_display_enabled:
            return

        if not self.predicted_display_enabled:
            return

        if track.state == TrackState.lost:
            return

        if track.time_since_update != 0:
            return

        if len(track.current_detections) == 0:
            return

        for camera_name in self.camera_names:
            if camera_name in track.current_bbox:
                continue

            last_seen = track.last_seen_frame_by_camera.get(camera_name)
            if last_seen is None:
                continue

            if frame_id - int(last_seen) > self.per_camera_predicted_display_max_missing:
                continue

            predicted = self.predicted_bbox_from_track(track, camera_name)
            if predicted is None:
                continue

            track.current_bbox[camera_name] = predicted

    def fill_missing_camera_predictions_for_matched_tracks(
        self,
        frame_id: int,
    ) -> None:
        for track in self.active_tracks():
            self.fill_missing_camera_predictions_for_matched_track(track, frame_id)

    def try_reassociate_unmatched_group(
        self,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[GlobalTrack]:
        """Conservative rescue before creating a new global ID
        """
        if group.mean_confidence < self.min_detection_confidence_for_matching:
            return None

        if self.current_frame_multiview_group_count >= 2 and group.num_views >= 2:
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=-1,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=0.0,
                motion_cost=0.0,
                spatial_cost=0.0,
                total_cost=0.0,
                decision="rejected",
                reason="unmatched_group_rescue_skipped_crowded_multiview",
            )
            return None

        feature = group.mean_feature()
        group_3d = self.triangulate_group(group, cameras_by_camera)
        candidates = []

        for track in self.active_tracks():
            if track.state == TrackState.lost:
                continue

            if track.time_since_update == 0:
                continue

            if track.time_since_update > self.single_view_reassociation_max_missing:
                continue

            if self.group_has_foreign_recent_tracklet(track, group):
                continue

            if self.group_local_continuity_conflict(track, group) is not None:
                continue

            same_tracklet_count = 0
            motion_scores = []

            for detection in group.detections:
                if self.is_same_tracklet_reliable(track, detection):
                    same_tracklet_count += 1

                motion_score = self.motion_score_for_detection(
                    track,
                    detection,
                    gating_multiplier=5.0,
                )

                if motion_score is not None:
                    motion_scores.append(motion_score)

            has_motion_evidence = len(motion_scores) > 0
            motion_score = float(np.mean(motion_scores)) if motion_scores else 0.75

            if same_tracklet_count == 0 and has_motion_evidence and motion_score > 0.55:
                continue

            appearance_score = 0.50
            has_appearance_evidence = feature is not None and bool(track.feature_gallery)

            if has_appearance_evidence:
                appearance_score = track.appearance_distance(feature)

                if same_tracklet_count == 0 and appearance_score > 0.45:
                    continue

            spatial_score = 0.50
            has_spatial_evidence = False

            if group_3d is not None and track.has_3d:
                distance_3d = np.linalg.norm(group_3d[0] - track.mean_3d[:3])
                threshold_3d = (
                    self.occlusion_reid_threshold_mm
                    + 40.0 * min(track.time_since_update, 20)
                )
                spatial_score = float(distance_3d / max(1e-12, threshold_3d))
                has_spatial_evidence = True

                if same_tracklet_count == 0 and spatial_score > 0.70:
                    continue

            if same_tracklet_count == 0 and not has_motion_evidence and not has_spatial_evidence:
                continue

            tracklet_bonus = -0.30 if same_tracklet_count > 0 else 0.0
            score = (
                0.45 * motion_score
                + 0.35 * appearance_score
                + 0.20 * spatial_score
                + tracklet_bonus
            )

            candidates.append((score, track, motion_score, appearance_score, spatial_score))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_track, best_motion, best_app, best_spatial = candidates[0]

        active_not_lost = [track for track in self.active_tracks() if track.state != TrackState.lost]
        max_score = 0.62 if len(active_not_lost) > 1 else 0.82

        if best_score > max_score:
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=best_track.global_id,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=best_app,
                motion_cost=best_motion,
                spatial_cost=best_spatial,
                total_cost=best_score,
                decision="rejected",
                reason="unmatched_group_rescue_score_too_high",
            )
            return None

        if len(candidates) > 1:
            second_score = float(candidates[1][0])
            margin = 0.18 if len(active_not_lost) > 1 else 0.10

            if best_score + margin >= second_score:
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=best_track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=best_app,
                    motion_cost=best_motion,
                    spatial_cost=best_spatial,
                    total_cost=best_score,
                    decision="rejected",
                    reason="unmatched_group_rescue_ambiguous",
                )
                return None

        self.update_track_with_group(
            best_track,
            group,
            frame_id,
            cameras_by_camera,
        )

        self.debug_writer.write(
            frame_id=frame_id,
            track_id=best_track.global_id,
            group_index=-1,
            group_views=group.camera_names,
            app_cost=best_app,
            motion_cost=best_motion,
            spatial_cost=best_spatial,
            total_cost=best_score,
            decision="accepted",
            reason="unmatched_group_reassociated",
        )

        print(
            f"[tracking] réassociation groupe non matché "
            f"track={best_track.global_id} "
            f"vues={group.camera_names} "
            f"score={best_score:.3f}"
        )

        return best_track

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

            if self.group_has_foreign_recent_tracklet(track, group):
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=[camera_name],
                    app_cost=1.0,
                    motion_cost=1.0,
                    spatial_cost=0.5,
                    total_cost=large_cost,
                    decision="rejected",
                    reason="single_view_foreign_tracklet",
                )
                continue

            local_owner_conflict = self.group_local_continuity_conflict(
                track,
                group,
                max_missing=25,
                max_motion_score=0.40,
                max_appearance_score=0.50,
                max_final_score=0.45,
                ambiguity_margin=0.10,
            )

            if local_owner_conflict is not None:
                owner_id, owner_score = local_owner_conflict
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=[camera_name],
                    app_cost=1.0,
                    motion_cost=owner_score,
                    spatial_cost=0.5,
                    total_cost=large_cost,
                    decision="rejected",
                    reason=f"single_view_local_owner_conflict_{owner_id}",
                )
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
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=[camera_name],
                    app_cost=1.0,
                    motion_cost=float(maha / gating_threshold),
                    spatial_cost=0.5,
                    total_cost=float(maha / gating_threshold),
                    decision="rejected",
                    reason="single_view_motion_gate",
                )
                continue

            motion_score = float(maha / gating_threshold)

            app_score = 0.5
            has_feature = detection.feature is not None and bool(track.feature_gallery)

            if has_feature:
                app_score = track.appearance_distance(detection.feature)

            tracklet_bonus = 0.0
            if self.is_same_tracklet_reliable(track, detection):
                tracklet_bonus = -0.35

            if not has_feature and tracklet_bonus >= 0:
                if motion_score > self.single_view_motion_threshold_without_feature:
                    self.debug_writer.write(
                        frame_id=frame_id,
                        track_id=track.global_id,
                        group_index=-1,
                        group_views=[camera_name],
                        app_cost=app_score,
                        motion_cost=motion_score,
                        spatial_cost=0.5,
                        total_cost=motion_score,
                        decision="rejected",
                        reason="single_view_no_feature_motion_not_enough",
                    )
                    continue

            if (
                has_feature
                and app_score > self.single_view_appearance_reject_threshold
                and tracklet_bonus >= 0
            ):
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=[camera_name],
                    app_cost=app_score,
                    motion_cost=motion_score,
                    spatial_cost=0.5,
                    total_cost=app_score,
                    decision="rejected",
                    reason="single_view_appearance_reject",
                )
                continue

            score = 0.60 * motion_score + 0.40 * app_score + tracklet_bonus
            candidates.append((score, track, motion_score, app_score, tracklet_bonus))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])

        best_score, best_track, best_motion, best_app, _ = candidates[0]

        if best_score > self.single_view_reassociation_max_score:
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=best_track.global_id,
                group_index=-1,
                group_views=[camera_name],
                app_cost=best_app,
                motion_cost=best_motion,
                spatial_cost=0.5,
                total_cost=best_score,
                decision="rejected",
                reason="single_view_score_too_high",
            )
            return None

        if len(candidates) > 1:
            second_score = float(candidates[1][0])

            if best_score + self.single_view_reassociation_margin >= second_score:
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=best_track.global_id,
                    group_index=-1,
                    group_views=[camera_name],
                    app_cost=best_app,
                    motion_cost=best_motion,
                    spatial_cost=0.5,
                    total_cost=best_score,
                    decision="rejected",
                    reason="single_view_ambiguous_candidates",
                )
                return None

        self.update_track_with_group(
            best_track,
            group,
            frame_id,
            cameras_by_camera,
        )

        self.debug_writer.write(
            frame_id=frame_id,
            track_id=best_track.global_id,
            group_index=-1,
            group_views=[camera_name],
            app_cost=best_app,
            motion_cost=best_motion,
            spatial_cost=0.5,
            total_cost=best_score,
            decision="accepted",
            reason="single_view_reassociated",
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
        """Réactive une ancienne track seulement avec une vraie continuité
        """
        if group.mean_confidence < self.min_detection_confidence_for_matching:
            return None

        feature = group.mean_feature()
        group_3d = self.triangulate_group(group, cameras_by_camera)

        active_not_lost = [
            track
            for track in self.active_tracks()
            if track.state != TrackState.lost
        ]
        crowded_scene = (
            self.current_frame_multiview_group_count >= 2
            or len(active_not_lost) >= 3
        )

        candidates = []

        for track in self.active_tracks():
            if track.state == TrackState.lost:
                continue

            if track.time_since_update == 0:
                continue

            if track.time_since_update > self.single_view_reassociation_max_missing:
                continue

            if self.group_has_foreign_recent_tracklet(track, group):
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=1.0,
                    motion_cost=1.0,
                    spatial_cost=1.0,
                    total_cost=large_cost,
                    decision="rejected",
                    reason="reactivation_foreign_tracklet",
                )
                continue

            local_owner_conflict = self.group_local_continuity_conflict(
                track,
                group,
                max_missing=30,
                max_motion_score=0.45,
                max_appearance_score=0.52,
                max_final_score=0.48,
                ambiguity_margin=0.10,
            )

            if local_owner_conflict is not None:
                owner_id, owner_score = local_owner_conflict
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=1.0,
                    motion_cost=owner_score,
                    spatial_cost=1.0,
                    total_cost=large_cost,
                    decision="rejected",
                    reason=f"reactivation_local_owner_conflict_{owner_id}",
                )
                continue

            if (
                self.group_is_partial_occlusion_candidate(group)
                and self.track_has_real_larger_overlap_with_group(track, group)
            ):
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=1.0,
                    motion_cost=1.0,
                    spatial_cost=1.0,
                    total_cost=large_cost,
                    decision="rejected",
                    reason="reactivation_partial_front_occluder_reject",
                )
                continue

            same_tracklet_count = 0
            motion_scores = []

            for detection in group.detections:
                if self.is_same_tracklet_reliable(track, detection):
                    same_tracklet_count += 1

                motion_score = self.motion_score_for_detection(
                    track,
                    detection,
                    gating_multiplier=5.5,
                )

                if motion_score is not None:
                    motion_scores.append(float(motion_score))

            has_motion_evidence = len(motion_scores) > 0
            motion_score = float(np.mean(motion_scores)) if motion_scores else 0.75
            strong_motion = has_motion_evidence and motion_score <= 0.40

            appearance_score = 0.50
            has_appearance_evidence = feature is not None and bool(track.feature_gallery)

            if has_appearance_evidence:
                appearance_score = track.appearance_distance(feature)

            if has_appearance_evidence and same_tracklet_count == 0:
                max_app = 0.42 if crowded_scene else max(0.50, self.reactivation_appearance_threshold)
                if appearance_score > max_app:
                    self.debug_writer.write(
                        frame_id=frame_id,
                        track_id=track.global_id,
                        group_index=-1,
                        group_views=group.camera_names,
                        app_cost=appearance_score,
                        motion_cost=motion_score,
                        spatial_cost=0.5,
                        total_cost=appearance_score,
                        decision="rejected",
                        reason="reactivation_appearance_reject",
                    )
                    continue

            spatial_score = 0.50
            has_spatial_evidence = False
            strong_spatial = False

            if group_3d is not None and track.has_3d:
                distance_3d = np.linalg.norm(group_3d[0] - track.mean_3d[:3])
                threshold_3d = (
                    self.occlusion_reid_threshold_mm
                    + 45.0 * min(track.time_since_update, 20)
                )
                spatial_score = float(distance_3d / max(1e-12, threshold_3d))
                has_spatial_evidence = True
                strong_spatial = spatial_score <= 0.45

                if same_tracklet_count == 0:
                    max_spatial = 0.65 if crowded_scene else 1.00
                    if spatial_score > max_spatial:
                        self.debug_writer.write(
                            frame_id=frame_id,
                            track_id=track.global_id,
                            group_index=-1,
                            group_views=group.camera_names,
                            app_cost=appearance_score,
                            motion_cost=motion_score,
                            spatial_cost=spatial_score,
                            total_cost=spatial_score,
                            decision="rejected",
                            reason="reactivation_spatial_reject",
                        )
                        continue

            has_continuity_proof = (
                same_tracklet_count > 0
                or strong_motion
                or strong_spatial
            )

            if crowded_scene and not has_continuity_proof:
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=appearance_score,
                    motion_cost=motion_score,
                    spatial_cost=spatial_score,
                    total_cost=1.0,
                    decision="rejected",
                    reason="reactivation_no_continuity_proof_crowded",
                )
                continue

            if (
                crowded_scene
                and track.time_since_update > 20
                and same_tracklet_count == 0
                and not strong_spatial
            ):
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=appearance_score,
                    motion_cost=motion_score,
                    spatial_cost=spatial_score,
                    total_cost=1.0,
                    decision="rejected",
                    reason="reactivation_old_track_without_strong_spatial",
                )
                continue

            if not has_appearance_evidence and same_tracklet_count == 0 and not has_spatial_evidence:
                continue

            tracklet_bonus = -0.35 if same_tracklet_count > 0 else 0.0
            score = (
                0.35 * appearance_score
                + 0.35 * motion_score
                + 0.30 * spatial_score
                + tracklet_bonus
            )

            candidates.append((
                score,
                track,
                motion_score,
                appearance_score,
                spatial_score,
                same_tracklet_count,
            ))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        best_score, best_track, best_motion, best_app, best_spatial, best_tracklets = candidates[0]

        max_score = 0.48 if crowded_scene else 0.80

        if best_tracklets > 0:
            max_score += 0.12

        if best_score > max_score:
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=best_track.global_id,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=best_app,
                motion_cost=best_motion,
                spatial_cost=best_spatial,
                total_cost=best_score,
                decision="rejected",
                reason="reactivation_score_too_high",
            )
            return None

        if len(candidates) > 1:
            second_score = float(candidates[1][0])
            margin = 0.22 if crowded_scene else 0.12

            if best_score + margin >= second_score:
                self.debug_writer.write(
                    frame_id=frame_id,
                    track_id=best_track.global_id,
                    group_index=-1,
                    group_views=group.camera_names,
                    app_cost=best_app,
                    motion_cost=best_motion,
                    spatial_cost=best_spatial,
                    total_cost=best_score,
                    decision="rejected",
                    reason="reactivation_ambiguous_candidates",
                )
                return None

        self.update_track_with_group(
            best_track,
            group,
            frame_id,
            cameras_by_camera,
        )

        self.debug_writer.write(
            frame_id=frame_id,
            track_id=best_track.global_id,
            group_index=-1,
            group_views=group.camera_names,
            app_cost=best_app,
            motion_cost=best_motion,
            spatial_cost=best_spatial,
            total_cost=best_score,
            decision="accepted",
            reason="reactivation_strict_continuity",
        )

        print(
            f"[tracking] réactivation stricte "
            f"track={best_track.global_id} "
            f"vues={group.camera_names} "
            f"score={best_score:.3f}"
        )

        return best_track


    def stable_active_tracks(self) -> List[GlobalTrack]:
        """Tracks that are established enough to be protected from new IDs/merges."""
        return [
            track
            for track in self.active_tracks()
            if track.state in {TrackState.confirmed_2d, TrackState.confirmed_3d, TrackState.occluded}
            and track.hits >= self.min_hits_to_confirm
            and (self.current_frame_id - track.first_seen_frame) >= 8
        ]

    def should_defer_new_track_in_crowded_scene(
        self,
        group: DetectionGroup,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> bool:

        if self.current_frame_id < 20:
            return False

        if self.current_frame_multiview_group_count < 2:
            return False

        if group.num_views < 2:
            return False

        stable_tracks = self.stable_active_tracks()

        if len(stable_tracks) < self.current_frame_multiview_group_count:
            return False

        candidates = []

        for track in stable_tracks:
            score = self.group_possible_recent_track_score(
                track,
                group,
                cameras_by_camera,
            )

            if score is not None:
                candidates.append((float(score), track.global_id))
                continue

            feature = group.mean_feature()
            if feature is not None and track.feature_gallery:
                app = track.appearance_distance(feature)
                if app < 0.55:
                    candidates.append((float(app), track.global_id))

        if not candidates:
            return False

        candidates.sort(key=lambda item: item[0])
        best_score, best_id = candidates[0]

        self.debug_writer.write(
            frame_id=self.current_frame_id,
            track_id=int(best_id),
            group_index=-1,
            group_views=group.camera_names,
            app_cost=0.0,
            motion_cost=0.0,
            spatial_cost=0.0,
            total_cost=float(best_score),
            decision="deferred",
            reason="new_track_deferred_crowded_existing_identities",
        )

        return True

    def create_track(
        self,
        group: DetectionGroup,
        frame_id: int,
        cameras_by_camera: Optional[Dict[str, CameraData]],
    ) -> Optional[GlobalTrack]:
        duplicate_owner = self.try_absorb_duplicate_unmatched_group(
            group,
            frame_id,
            cameras_by_camera,
        )

        if duplicate_owner is not None:
            return duplicate_owner

        partial_reassociated = self.try_reassociate_partial_occlusion_group(
            group,
            frame_id,
            cameras_by_camera,
        )

        if partial_reassociated is not None:
            return partial_reassociated

        missing_view_attached = self.try_attach_missing_single_view_to_matched_track(
            group,
            frame_id,
            cameras_by_camera,
        )

        if missing_view_attached is not None:
            return missing_view_attached

        reassociated = self.try_reassociate_single_view_group(
            group,
            frame_id,
            cameras_by_camera,
        )

        if reassociated is not None:
            return reassociated

        rescued = self.try_reassociate_unmatched_group(
            group,
            frame_id,
            cameras_by_camera,
        )

        if rescued is not None:
            return rescued

        reactivated = self.try_reactivate_existing_track(
            group,
            frame_id,
            cameras_by_camera,
        )

        if reactivated is not None:
            return reactivated

        if self.should_defer_new_track_in_crowded_scene(group, cameras_by_camera):
            return None

        if self.should_defer_crowded_single_view_new_track(group, cameras_by_camera):
            return None

        if self.should_defer_partial_occlusion_new_track(group):
            return None

        if self.group_may_belong_to_recent_track(group, cameras_by_camera):
            self.debug_writer.write(
                frame_id=frame_id,
                track_id=-1,
                group_index=-1,
                group_views=group.camera_names,
                app_cost=0.0,
                motion_cost=0.0,
                spatial_cost=0.0,
                total_cost=0.0,
                decision="deferred",
                reason="new_track_deferred_possible_existing",
            )
            return None

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

                if (
                    distances.size > 0
                    and float(np.min(distances)) < self.occlusion_xy_distance_mm
                ):
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

            self.fill_predicted_display_for_track(
                track,
                frame_id=self.current_frame_id,
            )

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

                    if (
                        bbox_iou(
                            track_a.current_bbox[camera_name],
                            track_b.current_bbox[camera_name],
                        )
                        > 0
                    ):
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


    def gallery_distance_between_tracks(
        self,
        track_a: GlobalTrack,
        track_b: GlobalTrack,
    ) -> Optional[float]:
        if not track_a.feature_gallery or not track_b.feature_gallery:
            return None

        feature_a = np.mean(np.stack(list(track_a.feature_gallery)), axis=0)
        feature_b = np.mean(np.stack(list(track_b.feature_gallery)), axis=0)

        norm_a = np.linalg.norm(feature_a)
        norm_b = np.linalg.norm(feature_b)

        if norm_a <= 1e-12 or norm_b <= 1e-12:
            return None

        feature_a = feature_a / norm_a
        feature_b = feature_b / norm_b

        return float(cdist([feature_a], [feature_b], metric="cosine")[0, 0])

    def shared_camera_duplicate_score(
        self,
        track_a: GlobalTrack,
        track_b: GlobalTrack,
    ) -> Optional[float]:
        shared_cameras = set(track_a.current_bbox.keys()) & set(track_b.current_bbox.keys())

        if not shared_cameras:
            return None

        scores = []

        for camera_name in shared_cameras:
            box_a = track_a.current_bbox[camera_name]
            box_b = track_b.current_bbox[camera_name]

            det_a = track_a.current_detections.get(camera_name)
            det_b = track_b.current_detections.get(camera_name)

            if det_a is not None and det_b is not None:
                if self.detections_pose_centers_clearly_different(det_a, det_b):
                    continue

                if det_b.feature is not None and track_a.feature_gallery:
                    app_ab = track_a.appearance_distance(det_b.feature)
                    if app_ab > 0.42:
                        continue

                if det_a.feature is not None and track_b.feature_gallery:
                    app_ba = track_b.appearance_distance(det_a.feature)
                    if app_ba > 0.42:
                        continue

            iou = bbox_iou(box_a, box_b)
            containment = bbox_containment_of_smaller(box_a, box_b)
            center_ratio = bbox_center_distance_ratio(box_a, box_b)
            area_ratio = bbox_small_area_ratio(box_a, box_b)

            if iou >= 0.60:
                scores.append(float(1.0 - iou))
                continue

            if containment >= 0.82 and center_ratio <= 0.30:
                score = (1.0 - containment) + center_ratio

                if area_ratio < 0.70:
                    score *= 0.85

                scores.append(float(score))

        if not scores:
            return None

        app_distance = self.gallery_distance_between_tracks(track_a, track_b)

        if app_distance is not None:
            if app_distance > 0.38:
                return None
            scores.append(app_distance)

        return float(np.mean(scores))

    def should_merge_occlusion_duplicate_pair(
        self,
        track_a: GlobalTrack,
        track_b: GlobalTrack,
    ) -> Optional[tuple[GlobalTrack, GlobalTrack, float]]:
        score = self.shared_camera_duplicate_score(track_a, track_b)

        if score is None:
            return None

        stable_a = (
            track_a.hits >= max(8, self.min_hits_to_confirm)
            and (self.current_frame_id - track_a.first_seen_frame) >= 30
        )
        stable_b = (
            track_b.hits >= max(8, self.min_hits_to_confirm)
            and (self.current_frame_id - track_b.first_seen_frame) >= 30
        )

        if stable_a and stable_b:
            return None

        if self.current_frame_multiview_group_count >= 2:
            return None

        ghost_a = (
            track_a.state == TrackState.occluded
            or track_a.hits <= 3
            or self.current_frame_id - track_a.first_seen_frame <= 20
            or any(
                camera_name in track_a.current_bbox
                and camera_name not in track_a.current_detections
                for camera_name in track_a.current_bbox
            )
        )
        ghost_b = (
            track_b.state == TrackState.occluded
            or track_b.hits <= 3
            or self.current_frame_id - track_b.first_seen_frame <= 20
            or any(
                camera_name in track_b.current_bbox
                and camera_name not in track_b.current_detections
                for camera_name in track_b.current_bbox
            )
        )

        if not ghost_a and not ghost_b:
            return None

        if score > 0.48:
            return None

        if track_a.first_seen_frame < track_b.first_seen_frame:
            keep, remove = track_a, track_b
        elif track_b.first_seen_frame < track_a.first_seen_frame:
            keep, remove = track_b, track_a
        elif track_a.hits >= track_b.hits:
            keep, remove = track_a, track_b
        else:
            keep, remove = track_b, track_a

        return keep, remove, score

    def absorb_track(
        self,
        keep: GlobalTrack,
        remove: GlobalTrack,
    ) -> None:
        """Merge remove into keep while preserving the stable global_id
        """
        keep.last_seen_frame = max(keep.last_seen_frame, remove.last_seen_frame)
        keep.hits = max(keep.hits, remove.hits)
        keep.age = max(keep.age, remove.age)

        if keep.time_since_update > remove.time_since_update:
            keep.time_since_update = remove.time_since_update
            keep.state = remove.state if remove.state != TrackState.lost else keep.state

        for camera_name, detection in remove.current_detections.items():
            if camera_name not in keep.current_detections:
                keep.current_detections[camera_name] = detection
                keep.current_bbox[camera_name] = remove.current_bbox[camera_name]

                if camera_name in remove.current_pose_2d:
                    keep.current_pose_2d[camera_name] = remove.current_pose_2d[camera_name]

                if camera_name in remove.current_joint_confidences:
                    keep.current_joint_confidences[camera_name] = remove.current_joint_confidences[camera_name]

        for camera_name, bbox in remove.current_bbox.items():
            if camera_name not in keep.current_bbox:
                keep.current_bbox[camera_name] = bbox

        keep.last_seen_frame_by_camera.update(remove.last_seen_frame_by_camera)

        for camera_name, mean in remove.kf2d_mean_by_camera.items():
            if camera_name not in keep.kf2d_mean_by_camera:
                keep.kf2d_mean_by_camera[camera_name] = mean
                keep.kf2d_cov_by_camera[camera_name] = remove.kf2d_cov_by_camera[camera_name]

        if keep.mean_3d is None and remove.mean_3d is not None:
            keep.mean_3d = remove.mean_3d
            keep.covariance_3d = remove.covariance_3d
            keep.last_known_3d_pos = remove.last_known_3d_pos
            keep.state = TrackState.confirmed_3d if remove.has_3d else keep.state

        for feature in remove.feature_gallery:
            keep.update_feature(feature, self.feature_update_weight)

        for camera_name, tracklet_id in remove.last_tracklet_id.items():
            keep.last_tracklet_id[camera_name] = tracklet_id
            self.tracklet_registry[(camera_name, tracklet_id)] = {
                "global_id": keep.global_id,
                "frame_id": self.current_frame_id,
            }

        for position in remove.position_3d_history:
            keep.position_3d_history.append(position)

        if remove.global_id in self.tracks:
            del self.tracks[remove.global_id]

    def merge_occlusion_duplicate_tracks(self) -> None:
        if self.current_frame_multiview_group_count >= 2:
            return

        tracks = self.active_tracks()
        merges: List[tuple[GlobalTrack, GlobalTrack, float]] = []

        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                track_a = tracks[i]
                track_b = tracks[j]

                decision = self.should_merge_occlusion_duplicate_pair(track_a, track_b)

                if decision is None:
                    continue

                keep, remove, score = decision

                # Avoid scheduling a track twice.
                if any(remove.global_id == existing_remove.global_id for _, existing_remove, _ in merges):
                    continue
                if any(keep.global_id == existing_remove.global_id for _, existing_remove, _ in merges):
                    continue

                merges.append((keep, remove, score))

        for keep, remove, score in merges:
            if keep.global_id not in self.tracks or remove.global_id not in self.tracks:
                continue

            removed_id = remove.global_id
            self.absorb_track(keep, remove)
            print(
                f"[tracking] fusion doublon occlusion "
                f"{removed_id} -> {keep.global_id} score={score:.3f}"
            )

            self.debug_writer.write(
                frame_id=self.current_frame_id,
                track_id=keep.global_id,
                group_index=-1,
                group_views=sorted(set(keep.current_bbox.keys())),
                app_cost=0.0,
                motion_cost=0.0,
                spatial_cost=0.0,
                total_cost=score,
                decision="accepted",
                reason=f"occlusion_duplicate_merge_removed_{removed_id}",
            )

    def suppress_predicted_ghost_boxes(self) -> None:
        """Masque uniquement les bbox prédites qui tombent sur une vraie bbox
        """
        if not self.suppress_predicted_ghost_enabled:
            return

        for camera_name in self.camera_names:
            real_tracks = [
                track
                for track in self.active_tracks()
                if camera_name in track.current_detections
                and camera_name in track.current_bbox
            ]

            if not real_tracks:
                continue

            predicted_tracks = [
                track
                for track in self.active_tracks()
                if camera_name in track.current_bbox
                and camera_name not in track.current_detections
                and track.time_since_update > 0
            ]

            for predicted_track in predicted_tracks:
                pred_box = predicted_track.current_bbox.get(camera_name)
                if pred_box is None:
                    continue

                best_overlap = None

                for real_track in real_tracks:
                    if real_track.global_id == predicted_track.global_id:
                        continue

                    real_box = real_track.current_bbox[camera_name]
                    iou = bbox_iou(pred_box, real_box)
                    containment = bbox_containment_of_smaller(pred_box, real_box)
                    center_ratio = bbox_center_distance_ratio(pred_box, real_box)

                    should_mask = (
                        iou >= self.predicted_ghost_iou_threshold
                        or (
                            containment >= self.predicted_ghost_containment_threshold
                            and center_ratio <= self.predicted_ghost_center_ratio_threshold
                        )
                    )

                    if not should_mask:
                        continue

                    score = max(iou, containment - center_ratio)
                    if best_overlap is None or score > best_overlap[0]:
                        best_overlap = (score, real_track.global_id, iou, containment, center_ratio)

                if best_overlap is None:
                    continue

                _score, real_id, iou, containment, center_ratio = best_overlap

                predicted_track.current_bbox.pop(camera_name, None)
                predicted_track.current_pose_2d.pop(camera_name, None)
                predicted_track.current_joint_confidences.pop(camera_name, None)

                self.debug_writer.write(
                    frame_id=self.current_frame_id,
                    track_id=predicted_track.global_id,
                    group_index=-1,
                    group_views=[camera_name],
                    app_cost=float(iou),
                    motion_cost=float(containment),
                    spatial_cost=float(center_ratio),
                    total_cost=float(_score),
                    decision="suppressed",
                    reason=f"predicted_ghost_masked_by_{real_id}",
                )

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

                shared_current_camera = set(track_a.current_bbox.keys()) & set(
                    track_b.current_bbox.keys()
                )

                if shared_current_camera:
                    continue

                distance = np.linalg.norm(track_a.mean_3d[:3] - track_b.mean_3d[:3])

                if distance > self.duplicate_merge_distance_mm:
                    continue

                if track_a.feature_gallery and track_b.feature_gallery:
                    feature_a = np.mean(
                        np.stack(list(track_a.feature_gallery)),
                        axis=0,
                    )
                    feature_b = np.mean(
                        np.stack(list(track_b.feature_gallery)),
                        axis=0,
                    )

                    app_distance = float(
                        cdist([feature_a], [feature_b], metric="cosine")[0, 0]
                    )
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
            keep.last_seen_frame_by_camera.update(remove.last_seen_frame_by_camera)
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
        self.current_frame_detections_by_camera = {
            camera_name: list(detections)
            for camera_name, detections in detections_by_camera.items()
        }

        self.predict_tracks()

        groups = self.group_detections(
            detections_by_camera=detections_by_camera,
            cameras_by_camera=cameras_by_camera,
        )

        self.current_frame_group_count = len(groups)
        self.current_frame_multiview_group_count = sum(
            1 for group in groups if group.num_views >= 2
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

        self.fill_missing_camera_predictions_for_matched_tracks(frame_id)

        self.update_occlusion_state()

        self.merge_occlusion_duplicate_tracks()
        self.suppress_predicted_ghost_boxes()

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