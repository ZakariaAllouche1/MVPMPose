from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cameras import CameraData
from src.geometry import project_points
from src.joints import joint_names_18, pelvis_18_index, skeleton_18
from src.tracking import GlobalTrack, TrackState


def color_for_id(person_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=int(person_id) * 1009)
    color = rng.integers(40, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def bgr_to_rgb(color_bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    b, g, r = color_bgr
    return r / 255.0, g / 255.0, b / 255.0


def has_valid_pose18(track: GlobalTrack) -> bool:
    if track.current_pose_3d_18 is None:
        return False

    pose = np.asarray(track.current_pose_3d_18)

    return pose.shape == (18, 3) and np.isfinite(pose).all()


def track_pose_reliable_for_display(
    track: GlobalTrack,
    min_views: int = 2,
    min_score: float = 0.01,
) -> bool:
    """True si la pose 3D courante est assez fiable pour être affichée
    """
    if not has_valid_pose18(track):
        return False

    if len(getattr(track, "learned_views_used", [])) < int(min_views):
        return False

    score = float(getattr(track, "learned_pose_score", 0.0))
    if not np.isfinite(score) or score < float(min_score):
        return False

    return True


def has_valid_unreliable_pose18(track: GlobalTrack) -> bool:
    pose = getattr(track, "current_unreliable_pose_3d_18", None)
    if pose is None:
        return False

    pose = np.asarray(pose)
    return pose.shape == (18, 3) and np.isfinite(pose).all()


def track_unreliable_pose_for_display(
    track: GlobalTrack,
    min_views: int = 2,
    min_score: float = 0.0,
) -> bool:
    """True si une pose low3D courante peut être affichée en debug
    """
    if not has_valid_unreliable_pose18(track):
        return False

    if len(getattr(track, "unreliable_views_used", [])) < int(min_views):
        return False

    score = float(getattr(track, "unreliable_pose_score", 0.0))
    if not np.isfinite(score) or score < float(min_score):
        return False

    return True



def track_reliable_detections_for_3d(track: GlobalTrack) -> Dict[str, object]:
    reliable = {}
    for camera_name, detection in track.current_detections.items():
        if bool(getattr(detection, "reliable_for_3d", False)) and not bool(getattr(detection, "partial_body", False)):
            reliable[camera_name] = detection
    return reliable


def detection_reliability_fields(track: GlobalTrack, camera_name: str) -> tuple[int, str, int, float]:
    detection = track.current_detections.get(camera_name)
    if detection is None:
        return 0, "", 0, float("nan")

    reliable = int(bool(getattr(detection, "reliable_for_3d", False)))
    reason = str(getattr(detection, "unreliable_reason", ""))
    partial = int(bool(getattr(detection, "partial_body", False)))
    completeness = float(getattr(detection, "body_completeness_score", float("nan")))
    return reliable, reason, partial, completeness


def track_visible_in_camera(track: GlobalTrack, camera_name: str) -> bool:
    return camera_name in track.current_bbox


def valid_pose_tracks(tracks: Iterable[GlobalTrack]) -> List[GlobalTrack]:
    return [
        track
        for track in tracks
        if track.state != TrackState.lost and track_pose_reliable_for_display(track)
    ]


class PoseCsvWriter:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.writer.writerow(self.header())
        self.file.flush()

    def header(self) -> List[str]:
        header = [
            "frame_id",
            "person_id",
            "views_used",
            "score",
        ]

        for joint_name in joint_names_18:
            header.extend([
                f"{joint_name}_x",
                f"{joint_name}_y",
                f"{joint_name}_z",
            ])

        return header

    def write_frame(self, frame_id: int, tracks: Iterable[GlobalTrack]) -> None:
        for track in valid_pose_tracks(tracks):
            pose = np.asarray(track.current_pose_3d_18, dtype=np.float32)

            row = [
                int(frame_id),
                int(track.global_id),
                "|".join(track.learned_views_used),
                float(track.learned_pose_score),
            ]

            for joint in pose:
                row.extend([
                    float(joint[0]),
                    float(joint[1]),
                    float(joint[2]),
                ])

            self.writer.writerow(row)

        self.file.flush()

    def close(self) -> None:
        self.file.close()


class UnreliablePoseCsvWriter:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.header())
        self.file.flush()

    def header(self) -> List[str]:
        header = [
            "frame_id",
            "person_id",
            "views_used",
            "score",
            "reason",
        ]

        for joint_name in joint_names_18:
            header.extend([
                f"{joint_name}_x",
                f"{joint_name}_y",
                f"{joint_name}_z",
            ])

        return header

    def write_track(self, frame_id: int, track: GlobalTrack) -> None:
        if not has_valid_unreliable_pose18(track):
            return

        pose = np.asarray(track.current_unreliable_pose_3d_18, dtype=np.float32)
        row = [
            int(frame_id),
            int(track.global_id),
            "|".join(getattr(track, "unreliable_views_used", [])),
            float(getattr(track, "unreliable_pose_score", 0.0)),
            str(getattr(track, "unreliable_pose_reason", "")),
        ]

        for joint in pose:
            row.extend([
                float(joint[0]),
                float(joint[1]),
                float(joint[2]),
            ])

        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class FailureCsvWriter:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(self.output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "frame_id",
            "person_id",
            "reason",
            "num_views",
            "views",
        ])

        self.file.flush()

    def write(
        self,
        frame_id: int,
        person_id: int,
        reason: str,
        views: Optional[List[str]] = None,
    ) -> None:
        views = views or []

        self.writer.writerow([
            int(frame_id),
            int(person_id),
            reason,
            len(views),
            "|".join(views),
        ])

        self.file.flush()

    def close(self) -> None:
        self.file.close()


def draw_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        2,
    )

    cv2.putText(
        image,
        label,
        (int(x1), max(20, int(y1) - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_projected_skeleton(
    image: np.ndarray,
    points_2d: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]

    for point in points_2d:
        x, y = point

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        x_i = int(round(float(x)))
        y_i = int(round(float(y)))

        if 0 <= x_i < width and 0 <= y_i < height:
            cv2.circle(image, (x_i, y_i), 4, color, -1)

    for joint_a, joint_b in skeleton_18:
        pa = points_2d[joint_a]
        pb = points_2d[joint_b]

        if not np.isfinite(pa).all() or not np.isfinite(pb).all():
            continue

        xa, ya = int(round(float(pa[0]))), int(round(float(pa[1])))
        xb, yb = int(round(float(pb[0]))), int(round(float(pb[1])))

        if (
            0 <= xa < width
            and 0 <= ya < height
            and 0 <= xb < width
            and 0 <= yb < height
        ):
            cv2.line(image, (xa, ya), (xb, yb), color, 2)


def draw_reprojection_frame(
    frame_id: int,
    activity_name: str,
    frames_by_camera: Dict[str, np.ndarray],
    cameras_by_camera: Dict[str, CameraData],
    tracks: Iterable[GlobalTrack],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = list(tracks)

    for camera_name, frame in frames_by_camera.items():
        if camera_name not in cameras_by_camera:
            continue

        image = frame.copy()
        camera = cameras_by_camera[camera_name]

        for track in tracks:
            if track.state == TrackState.lost:
                continue

            if not track_visible_in_camera(track, camera_name):
                continue

            color = color_for_id(track.global_id)
            reliable_pose = track_pose_reliable_for_display(track)
            unreliable_pose = (not reliable_pose) and track_unreliable_pose_for_display(track)

            label = f"ID {track.global_id}"

            if track.state == TrackState.occluded:
                label += " [occ]"
            elif track.is_occluded_by is not None:
                label += f" [occ by {track.is_occluded_by}]"

            if unreliable_pose:
                reason = str(getattr(track, "unreliable_pose_reason", "low3D"))
                label += f" [low3D:{reason}]"

            draw_bbox(
                image=image,
                bbox=track.current_bbox[camera_name],
                color=color,
                label=label,
            )

            if reliable_pose:
                pose18 = np.asarray(track.current_pose_3d_18, dtype=np.float32)
            elif unreliable_pose:
                pose18 = np.asarray(track.current_unreliable_pose_3d_18, dtype=np.float32)
            else:
                pose18 = None

            if pose18 is not None:
                try:
                    projected = project_points(pose18, camera.P)
                    draw_projected_skeleton(image, projected, color)
                except Exception:
                    pass

        output_path = (
            output_dir
            / activity_name
            / f"frame_{frame_id:06d}_{camera_name}.jpg"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), image)


_ACTIVITY_AXIS_LIMITS: Dict[str, Dict[str, tuple[float, float]]] = {}


def compute_scene_limits(
    tracks: Iterable[GlobalTrack],
    padding_xy: float = 650.0,
    min_radius_xy: float = 2000.0,
    z_min_fixed: float = 0.0,
    z_padding_top: float = 900.0,
    min_z_max: float = 3000.0,
) -> Optional[Dict[str, tuple[float, float]]]:
    points = []

    for track in valid_pose_tracks(tracks):
        pose = np.asarray(track.current_pose_3d_18, dtype=np.float32)
        points.append(pose)

    if not points:
        return None

    points_array = np.concatenate(points, axis=0)

    x_min, y_min, z_min = np.min(points_array, axis=0)
    x_max, y_max, z_max = np.max(points_array, axis=0)

    center_x = 0.5 * (float(x_min) + float(x_max))
    center_y = 0.5 * (float(y_min) + float(y_max))

    radius_xy = max(
        0.5 * float(x_max - x_min) + padding_xy,
        0.5 * float(y_max - y_min) + padding_xy,
        min_radius_xy,
    )

    z_lower = float(z_min_fixed)
    z_upper = max(float(z_max) + z_padding_top, float(min_z_max))

    return {
        "x": (float(center_x - radius_xy), float(center_x + radius_xy)),
        "y": (float(center_y - radius_xy), float(center_y + radius_xy)),
        "z": (z_lower, z_upper),
    }


def expand_axis_limits(
    old_limits: Dict[str, tuple[float, float]],
    new_limits: Dict[str, tuple[float, float]],
    margin: float = 300.0,
) -> Dict[str, tuple[float, float]]:
    expanded = {}

    for axis in ["x", "y", "z"]:
        old_min, old_max = old_limits[axis]
        new_min, new_max = new_limits[axis]
        expanded[axis] = (
            float(min(old_min, new_min - margin)),
            float(max(old_max, new_max + margin)),
        )

    return expanded


def get_activity_axis_limits(
    activity_name: str,
    tracks: Iterable[GlobalTrack],
    allow_expand: bool = False,
) -> Optional[Dict[str, tuple[float, float]]]:
    """
    Retourne des limites fixes par activité
    """
    tracks = list(tracks)
    cached = _ACTIVITY_AXIS_LIMITS.get(activity_name)
    current = compute_scene_limits(tracks)

    if cached is None:
        if current is None:
            return None
        _ACTIVITY_AXIS_LIMITS[activity_name] = current
        return current

    if allow_expand and current is not None:
        cached = expand_axis_limits(cached, current)
        _ACTIVITY_AXIS_LIMITS[activity_name] = cached

    return cached


def reset_activity_axis_limits(activity_name: Optional[str] = None) -> None:
    """Permet de réinitialiser le cache si besoin entre deux traitements."""
    if activity_name is None:
        _ACTIVITY_AXIS_LIMITS.clear()
        return

    _ACTIVITY_AXIS_LIMITS.pop(activity_name, None)


def draw_ground_reference(
    ax,
    axis_limits: Dict[str, tuple[float, float]],
    grid_step: float = 500.0,
) -> None:
    """Dessine une grille simple au sol dans le plan X-Y
    """
    x_min, x_max = axis_limits["x"]
    y_min, y_max = axis_limits["y"]
    z_min, z_max = axis_limits["z"]

    ground_z = 0.0
    if not (z_min <= ground_z <= z_max):
        ground_z = z_min

    x_values = np.arange(
        np.floor(x_min / grid_step) * grid_step,
        np.ceil(x_max / grid_step) * grid_step + grid_step,
        grid_step,
    )
    y_values = np.arange(
        np.floor(y_min / grid_step) * grid_step,
        np.ceil(y_max / grid_step) * grid_step + grid_step,
        grid_step,
    )

    for x in x_values:
        ax.plot(
            [x, x],
            [y_min, y_max],
            [ground_z, ground_z],
            color="0.80",
            linewidth=0.7,
            alpha=0.75,
        )

    for y in y_values:
        ax.plot(
            [x_min, x_max],
            [y, y],
            [ground_z, ground_z],
            color="0.80",
            linewidth=0.7,
            alpha=0.75,
        )


def draw_3d_scene_frame(
    frame_id: int,
    activity_name: str,
    tracks: Iterable[GlobalTrack],
    output_dir: str | Path,
    axis_limits: Optional[Dict[str, tuple[float, float]]] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = valid_pose_tracks(tracks)

    if axis_limits is None:
        axis_limits = get_activity_axis_limits(
            activity_name=activity_name,
            tracks=tracks,
            allow_expand=False,
        )

    if axis_limits is None:
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    draw_ground_reference(ax, axis_limits)

    if not tracks:
        ax.text(
            0.5 * (axis_limits["x"][0] + axis_limits["x"][1]),
            0.5 * (axis_limits["y"][0] + axis_limits["y"][1]),
            0.5 * (axis_limits["z"][0] + axis_limits["z"][1]),
            "No valid 3D pose",
            color="0.35",
        )

    for track in tracks:
        pose = np.asarray(track.current_pose_3d_18, dtype=np.float32)
        color = bgr_to_rgb(color_for_id(track.global_id))

        ax.scatter(
            pose[:, 0],
            pose[:, 1],
            pose[:, 2],
            s=35,
            color=color,
            depthshade=True,
        )

        for joint_a, joint_b in skeleton_18:
            ax.plot(
                [pose[joint_a, 0], pose[joint_b, 0]],
                [pose[joint_a, 1], pose[joint_b, 1]],
                [pose[joint_a, 2], pose[joint_b, 2]],
                linewidth=2,
                color=color,
            )

        root = pose[pelvis_18_index]

        ax.text(
            float(root[0]),
            float(root[1]),
            float(root[2]),
            f"ID {track.global_id}",
            color=color,
        )

    ax.set_xlim(*axis_limits["x"])
    ax.set_ylim(*axis_limits["y"])
    ax.set_zlim(*axis_limits["z"])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    ax.set_title(f"{activity_name} | frame {frame_id}")

    ax.set_box_aspect((
        axis_limits["x"][1] - axis_limits["x"][0],
        axis_limits["y"][1] - axis_limits["y"][0],
        axis_limits["z"][1] - axis_limits["z"][0],
    ))

    ax.view_init(elev=10, azim=18)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

    output_path = (
        output_dir
        / activity_name
        / f"frame_{frame_id:06d}_3d.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=120)
    plt.close(fig)


class DetectionEvalCsvWriter:
    def __init__(
        self,
        activity_name: str,
        csv_dir: str | Path,
        camera_names: List[str],
    ) -> None:
        self.activity_name = activity_name
        self.csv_dir = Path(csv_dir)
        self.camera_names = list(camera_names)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        self.summary_file = open(
            self.csv_dir / f"{activity_name}_detection_summary.csv",
            "w",
            newline="",
            encoding="utf-8",
        )
        self.summary_writer = csv.writer(self.summary_file)
        self.summary_writer.writerow([
            "frame_id",
            "camera",
            "raw_yolo_boxes",
            "valid_yolo_boxes_after_clip",
            "detections_before_duplicate_suppression",
            "final_detections",
            "num_with_pose",
            "num_with_feature",
            "num_reliable_for_3d",
            "tracklet_ids",
            "confidences",
            "bboxes_xyxy",
            "reliable_for_3d",
            "unreliable_reasons",
            "partial_body",
            "body_completeness",
        ])
        self.summary_file.flush()

        self.raw_files: Dict[str, Any] = {}
        self.raw_writers: Dict[str, csv.writer] = {}

        for camera_name in self.camera_names:
            raw_file = open(
                self.csv_dir / f"{activity_name}_{camera_name}_detections.csv",
                "w",
                newline="",
                encoding="utf-8",
            )
            raw_writer = csv.writer(raw_file)
            raw_writer.writerow([
                "frame",
                "camera",
                "image_id",
                "raw_box_index",
                "tracklet_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "w",
                "h",
                "score",
                "class_id",
                "valid_after_clip",
                "raw_x1",
                "raw_y1",
                "raw_x2",
                "raw_y2",
            ])
            raw_file.flush()
            self.raw_files[camera_name] = raw_file
            self.raw_writers[camera_name] = raw_writer

    def write_frame(
        self,
        frame_id: int,
        detections_by_camera: Dict[str, list],
        raw_yolo_by_camera: Optional[Dict[str, List[Dict[str, object]]]] = None,
        detection_counts_by_camera: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        raw_yolo_by_camera = raw_yolo_by_camera or {}
        detection_counts_by_camera = detection_counts_by_camera or {}

        for camera_name in self.camera_names:
            detections = detections_by_camera.get(camera_name, [])
            raw_items = raw_yolo_by_camera.get(camera_name, [])
            counts = detection_counts_by_camera.get(camera_name, {})

            num_with_pose = sum(d.pose_2d is not None for d in detections)
            num_with_feature = sum(d.feature is not None for d in detections)
            num_reliable_for_3d = sum(
                bool(getattr(d, "reliable_for_3d", False))
                for d in detections
            )

            self.summary_writer.writerow([
                int(frame_id),
                camera_name,
                int(counts.get("raw_yolo", len(raw_items))),
                int(counts.get("valid_after_clip", sum(1 for item in raw_items if item.get("valid_after_clip", False)))),
                int(counts.get("before_suppression", len(detections))),
                int(counts.get("after_suppression", len(detections))),
                int(num_with_pose),
                int(num_with_feature),
                int(num_reliable_for_3d),
                "|".join(str(d.tracklet_id) for d in detections),
                "|".join(f"{float(d.confidence):.4f}" for d in detections),
                "|".join(str(tuple(int(v) for v in d.bbox_xyxy)) for d in detections),
                "|".join("1" if bool(getattr(d, "reliable_for_3d", False)) else "0" for d in detections),
                "|".join(str(getattr(d, "unreliable_reason", "")) for d in detections),
                "|".join("1" if bool(getattr(d, "partial_body", False)) else "0" for d in detections),
                "|".join(f"{float(getattr(d, 'body_completeness_score', 0.0)):.3f}" for d in detections),
            ])

            writer = self.raw_writers.get(camera_name)
            if writer is None:
                continue

            for item in raw_items:
                bbox = item.get("bbox_xyxy")
                raw_bbox = item.get("raw_bbox_xyxy")

                if bbox is None:
                    x1 = y1 = x2 = y2 = np.nan
                    w = h = np.nan
                else:
                    x1, y1, x2, y2 = bbox
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)

                if raw_bbox is None:
                    rx1 = ry1 = rx2 = ry2 = np.nan
                else:
                    rx1, ry1, rx2, ry2 = raw_bbox

                image_id = f"{camera_name}_{int(frame_id):06d}"
                writer.writerow([
                    int(frame_id),
                    camera_name,
                    image_id,
                    int(item.get("raw_index", -1)),
                    -1 if item.get("tracklet_id") is None else int(item.get("tracklet_id")),
                    x1,
                    y1,
                    x2,
                    y2,
                    w,
                    h,
                    float(item.get("confidence", 0.0)),
                    int(item.get("class_id", 0)),
                    int(bool(item.get("valid_after_clip", False))),
                    rx1,
                    ry1,
                    rx2,
                    ry2,
                ])

        self.summary_file.flush()
        for raw_file in self.raw_files.values():
            raw_file.flush()

    def close(self) -> None:
        self.summary_file.close()
        for raw_file in self.raw_files.values():
            raw_file.close()
        self.raw_files.clear()
        self.raw_writers.clear()


class TrackingEvalCsvWriter:
    def __init__(
        self,
        activity_name: str,
        csv_dir: str | Path,
        camera_names: List[str],
    ) -> None:
        self.activity_name = activity_name
        self.csv_dir = Path(csv_dir)
        self.camera_names = list(camera_names)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        self.mot_cam_files: Dict[str, Any] = {}
        for camera_name in self.camera_names:
            self.mot_cam_files[camera_name] = open(
                self.csv_dir / f"{activity_name}_{camera_name}_tracking_mot.txt",
                "w",
                encoding="utf-8",
            )

        self.global_mot_file = open(
            self.csv_dir / f"{activity_name}_global_tracking_mot.txt",
            "w",
            encoding="utf-8",
        )

        self.global_frame_map_file = open(
            self.csv_dir / f"{activity_name}_global_frame_map.csv",
            "w",
            newline="",
            encoding="utf-8",
        )
        self.global_frame_map_writer = csv.writer(self.global_frame_map_file)
        self.global_frame_map_writer.writerow(["global_frame", "frame", "camera"])

        self.global_debug_file = open(
            self.csv_dir / f"{activity_name}_global_tracking_debug.csv",
            "w",
            newline="",
            encoding="utf-8",
        )
        self.global_debug_writer = csv.writer(self.global_debug_file)
        self.global_debug_writer.writerow([
            "global_frame",
            "frame",
            "camera",
            "global_id",
            "x1",
            "y1",
            "w",
            "h",
            "score",
            "track_state",
            "time_since_update",
            "has_real_detection",
            "is_predicted_bbox",
            "is_occluded_by",
            "has_3d",
            "x3d",
            "y3d",
            "z3d",
            "has_tracking_3d",
            "tracking_3d_updated_this_frame",
            "tracking_3d_views",
            "has_current_learned_pose",
            "learned_pose_reliable",
            "learned_views_used",
            "track_real_views_count",
            "track_reliable_views_count",
            "track_reliable_views",
            "detection_reliable_for_3d",
            "detection_unreliable_reason",
            "detection_partial_body",
            "detection_body_completeness",
            "has_current_unreliable_pose",
            "unreliable_pose_score",
            "unreliable_pose_reason",
            "unreliable_views_used",
        ])

        self.training_file = open(
            self.csv_dir / f"{activity_name}_training_data.csv",
            "w",
            newline="",
            encoding="utf-8",
        )
        self.training_writer = csv.writer(self.training_file)
        self.training_writer.writerow([
            "frame_id",
            "person_id",
            "camera",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "root_x",
            "root_y",
            "root_z",
            "root_source",
        ])

    def _track_position(self, track: GlobalTrack) -> tuple[int, float, float, float]:
        if track.mean_3d is not None:
            pos = np.asarray(track.mean_3d[:3], dtype=np.float64)
            if np.isfinite(pos).all():
                return 1, float(pos[0]), float(pos[1]), float(pos[2])
        return 0, float("nan"), float("nan"), float("nan")

    def _training_position(self, track: GlobalTrack) -> tuple[str, float, float, float]:
        if track_pose_reliable_for_display(track):
            root = np.asarray(track.current_pose_3d_18[pelvis_18_index], dtype=np.float64)
            if np.isfinite(root).all():
                return "pose18_pelvis", float(root[0]), float(root[1]), float(root[2])

        if track.mean_3d is not None:
            pos = np.asarray(track.mean_3d[:3], dtype=np.float64)
            if np.isfinite(pos).all():
                return "track_mean_3d", float(pos[0]), float(pos[1]), float(pos[2])

        return "none", float("nan"), float("nan"), float("nan")

    def write_frame(self, frame_id: int, tracks: Iterable[GlobalTrack]) -> None:
        tracks = list(tracks)
        num_cameras = max(1, len(self.camera_names))

        for camera_rank, camera_name in enumerate(self.camera_names):
            global_frame = (int(frame_id) - 1) * num_cameras + camera_rank + 1
            self.global_frame_map_writer.writerow([
                int(global_frame),
                int(frame_id),
                camera_name,
            ])

        for track in tracks:
            if track.state == TrackState.lost:
                continue

            has_3d, x3d, y3d, z3d = self._track_position(track)
            root_source, root_x, root_y, root_z = self._training_position(track)
            real_views = sorted(track.current_detections.keys())
            reliable_views = sorted(track_reliable_detections_for_3d(track).keys())
            has_current_learned_pose = int(has_valid_pose18(track))
            learned_reliable = int(track_pose_reliable_for_display(track))
            tracking_updated = int(bool(getattr(track, "tracking_3d_updated_this_frame", False)))
            tracking_views = sorted(getattr(track, "tracking_3d_views_used", []))

            for camera_rank, camera_name in enumerate(self.camera_names):
                if camera_name not in track.current_bbox:
                    continue

                x1, y1, x2, y2 = map(float, track.current_bbox[camera_name])
                w = x2 - x1
                h = y2 - y1

                if w <= 1 or h <= 1:
                    continue

                global_frame = (int(frame_id) - 1) * num_cameras + camera_rank + 1
                has_real_detection = int(camera_name in track.current_detections)
                is_predicted_bbox = int(not has_real_detection)
                score = 1.0 if has_real_detection else 0.5

                # MOT: seulement les tracks confirmées avec une vraie détection.
                if (
                    track.state in {TrackState.confirmed_2d, TrackState.confirmed_3d}
                    and has_real_detection
                ):
                    line = (
                        f"{int(frame_id)},{int(track.global_id)},{x1:.2f},{y1:.2f},"
                        f"{w:.2f},{h:.2f},{score:.1f},-1,-1,-1\n"
                    )
                    self.mot_cam_files[camera_name].write(line)

                    global_line = (
                        f"{int(global_frame)},{int(track.global_id)},{x1:.2f},{y1:.2f},"
                        f"{w:.2f},{h:.2f},{score:.1f},-1,-1,-1\n"
                    )
                    self.global_mot_file.write(global_line)

                self.global_debug_writer.writerow([
                    int(global_frame),
                    int(frame_id),
                    camera_name,
                    int(track.global_id),
                    x1,
                    y1,
                    w,
                    h,
                    score,
                    track.state.name,
                    int(track.time_since_update),
                    has_real_detection,
                    is_predicted_bbox,
                    "" if track.is_occluded_by is None else int(track.is_occluded_by),
                    has_3d,
                    x3d,
                    y3d,
                    z3d,
                    has_3d,
                    tracking_updated,
                    "|".join(tracking_views),
                    has_current_learned_pose,
                    learned_reliable,
                    "|".join(getattr(track, "learned_views_used", [])),
                    len(real_views),
                    len(reliable_views),
                    "|".join(reliable_views),
                    detection_reliability_fields(track, camera_name)[0],
                    detection_reliability_fields(track, camera_name)[1],
                    detection_reliability_fields(track, camera_name)[2],
                    detection_reliability_fields(track, camera_name)[3],
                    int(has_valid_unreliable_pose18(track)),
                    float(getattr(track, "unreliable_pose_score", 0.0)),
                    str(getattr(track, "unreliable_pose_reason", "")),
                    "|".join(getattr(track, "unreliable_views_used", [])),
                ])

                if has_real_detection and root_source != "none":
                    self.training_writer.writerow([
                        int(frame_id),
                        int(track.global_id),
                        camera_name,
                        x1,
                        y1,
                        x2,
                        y2,
                        root_x,
                        root_y,
                        root_z,
                        root_source,
                    ])

        for file in self.mot_cam_files.values():
            file.flush()
        self.global_mot_file.flush()
        self.global_frame_map_file.flush()
        self.global_debug_file.flush()
        self.training_file.flush()

    def close(self) -> None:
        for file in self.mot_cam_files.values():
            file.close()
        self.mot_cam_files.clear()
        self.global_mot_file.close()
        self.global_frame_map_file.close()
        self.global_debug_file.close()
        self.training_file.close()


class ActivityOutputs:
    def __init__(
        self,
        activity_name: str,
        csv_dir: str | Path,
        debug_dir: str | Path,
        save_csv: bool = True,
        save_failures: bool = True,
        camera_names: Optional[List[str]] = None,
        save_debug_exports: bool = True,
        save_unreliable_pose_csv: bool = False,
    ) -> None:
        self.activity_name = activity_name
        self.csv_dir = Path(csv_dir)
        self.debug_dir = Path(debug_dir)

        self.pose_writer: Optional[PoseCsvWriter] = None
        self.unreliable_pose_writer: Optional[UnreliablePoseCsvWriter] = None
        self.failure_writer: Optional[FailureCsvWriter] = None
        self.detection_eval_writer: Optional[DetectionEvalCsvWriter] = None
        self.tracking_eval_writer: Optional[TrackingEvalCsvWriter] = None

        if save_csv:
            self.pose_writer = PoseCsvWriter(
                self.csv_dir / f"{activity_name}_keypoints3d_multiperson.csv"
            )

        if save_unreliable_pose_csv:
            self.unreliable_pose_writer = UnreliablePoseCsvWriter(
                self.debug_dir / f"{activity_name}_unreliable_poses_low3d.csv"
            )

        if save_failures:
            self.failure_writer = FailureCsvWriter(
                self.debug_dir / f"{activity_name}_failed_poses.csv"
            )

        if save_debug_exports and camera_names is not None:
            self.detection_eval_writer = DetectionEvalCsvWriter(
                activity_name=activity_name,
                csv_dir=self.csv_dir,
                camera_names=list(camera_names),
            )
            self.tracking_eval_writer = TrackingEvalCsvWriter(
                activity_name=activity_name,
                csv_dir=self.csv_dir,
                camera_names=list(camera_names),
            )

    def write_pose_frame(
        self,
        frame_id: int,
        tracks: Iterable[GlobalTrack],
    ) -> None:
        if self.pose_writer is None:
            return

        self.pose_writer.write_frame(frame_id, tracks)

    def write_unreliable_pose(
        self,
        frame_id: int,
        track: GlobalTrack,
    ) -> None:
        if self.unreliable_pose_writer is None:
            return

        self.unreliable_pose_writer.write_track(frame_id=frame_id, track=track)

    def write_failure(
        self,
        frame_id: int,
        person_id: int,
        reason: str,
        views: Optional[List[str]] = None,
    ) -> None:
        if self.failure_writer is None:
            return

        self.failure_writer.write(
            frame_id=frame_id,
            person_id=person_id,
            reason=reason,
            views=views,
        )

    def write_detection_frame(
        self,
        frame_id: int,
        detections_by_camera: Dict[str, list],
        raw_yolo_by_camera: Optional[Dict[str, List[Dict[str, object]]]] = None,
        detection_counts_by_camera: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        if self.detection_eval_writer is None:
            return

        self.detection_eval_writer.write_frame(
            frame_id=frame_id,
            detections_by_camera=detections_by_camera,
            raw_yolo_by_camera=raw_yolo_by_camera,
            detection_counts_by_camera=detection_counts_by_camera,
        )

    def write_tracking_frame(
        self,
        frame_id: int,
        tracks: Iterable[GlobalTrack],
    ) -> None:
        if self.tracking_eval_writer is None:
            return

        self.tracking_eval_writer.write_frame(frame_id=frame_id, tracks=tracks)

    def close(self) -> None:
        if self.pose_writer is not None:
            self.pose_writer.close()

        if self.unreliable_pose_writer is not None:
            self.unreliable_pose_writer.close()

        if self.failure_writer is not None:
            self.failure_writer.close()

        if self.detection_eval_writer is not None:
            self.detection_eval_writer.close()

        if self.tracking_eval_writer is not None:
            self.tracking_eval_writer.close()


def draw_annotated_video_frame(
    frame_id: int,
    camera_name: str,
    frame: np.ndarray,
    camera: CameraData,
    tracks: Iterable[GlobalTrack],
) -> np.ndarray:
    image = frame.copy()

    cv2.putText(
        image,
        f"Frame {frame_id}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for track in tracks:
        if track.state == TrackState.lost:
            continue

        if not track_visible_in_camera(track, camera_name):
            continue

        color = color_for_id(track.global_id)
        reliable_pose = track_pose_reliable_for_display(track)
        unreliable_pose = (not reliable_pose) and track_unreliable_pose_for_display(track)

        label = f"ID {track.global_id}"

        if track.state == TrackState.occluded:
            label += " [occ]"
        elif track.is_occluded_by is not None:
            label += f" [occ by {track.is_occluded_by}]"

        if unreliable_pose:
            label += " [low3D]"

        draw_bbox(
            image=image,
            bbox=track.current_bbox[camera_name],
            color=color,
            label=label,
        )

        if reliable_pose:
            pose18 = np.asarray(track.current_pose_3d_18, dtype=np.float32)
        elif unreliable_pose:
            pose18 = np.asarray(track.current_unreliable_pose_3d_18, dtype=np.float32)
        else:
            pose18 = None

        if pose18 is not None:
            try:
                projected = project_points(pose18, camera.P)
                draw_projected_skeleton(
                    image=image,
                    points_2d=projected,
                    color=(255, 0, 0),
                )
            except Exception:
                pass

    return image


def make_video_writer(
    output_path: str | Path,
    fps: float,
    frame_size: tuple[int, int],
    codec: str = "mp4v",
) -> tuple[cv2.VideoWriter, Path]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )

    if writer.isOpened():
        return writer, output_path

    alt_path = output_path.with_name(output_path.stem + "_avc1.mp4")
    writer = cv2.VideoWriter(
        str(alt_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        frame_size,
    )

    if writer.isOpened():
        return writer, alt_path

    alt_path = output_path.with_name(output_path.stem + "_mjpg.avi")
    writer = cv2.VideoWriter(
        str(alt_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        frame_size,
    )

    if writer.isOpened():
        return writer, alt_path

    raise RuntimeError(f"Impossible de créer VideoWriter pour {output_path}")


class ActivityVideoWriters:
    def __init__(
        self,
        activity_name: str,
        camera_names: list[str],
        video_output_dir: str | Path,
        fps: float,
        frame_size_by_camera: dict[str, tuple[int, int]],
        output_size: tuple[int, int] | None = None,
        codec: str = "mp4v",
    ) -> None:
        self.activity_name = activity_name
        self.camera_names = list(camera_names)
        self.video_output_dir = Path(video_output_dir)
        self.output_size = output_size
        self.codec = codec

        self.writers: dict[str, cv2.VideoWriter] = {}
        self.paths: dict[str, Path] = {}

        for camera_name in self.camera_names:
            if output_size is None:
                frame_size = frame_size_by_camera[camera_name]
            else:
                frame_size = output_size

            output_path = (
                self.video_output_dir
                / f"{activity_name}_{camera_name}_tracked.mp4"
            )

            writer, actual_path = make_video_writer(
                output_path=output_path,
                fps=fps,
                frame_size=frame_size,
                codec=codec,
            )

            self.writers[camera_name] = writer
            self.paths[camera_name] = actual_path

            print(f"[video] écriture {camera_name}: {actual_path}")

    def write_frame(
        self,
        frame_id: int,
        frames_by_camera: dict[str, np.ndarray],
        cameras_by_camera: dict[str, CameraData],
        tracks: Iterable[GlobalTrack],
    ) -> None:
        tracks = list(tracks)

        for camera_name in self.camera_names:
            if camera_name not in frames_by_camera:
                continue

            if camera_name not in cameras_by_camera:
                continue

            image = draw_annotated_video_frame(
                frame_id=frame_id,
                camera_name=camera_name,
                frame=frames_by_camera[camera_name],
                camera=cameras_by_camera[camera_name],
                tracks=tracks,
            )

            if self.output_size is not None:
                image = cv2.resize(
                    image,
                    self.output_size,
                    interpolation=cv2.INTER_AREA,
                )

            self.writers[camera_name].write(image)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()

        self.writers.clear()