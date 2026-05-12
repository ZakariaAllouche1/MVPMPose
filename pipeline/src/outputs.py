from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cameras import CameraData
from src.geometry import project_points
from src.joints import joint_names_15, pelvis_15_index, skeleton_15
from src.tracking import GlobalTrack, TrackState


def color_for_id(person_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=int(person_id) * 1009)
    color = rng.integers(40, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def bgr_to_rgb(color_bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    b, g, r = color_bgr
    return r / 255.0, g / 255.0, b / 255.0


def has_valid_pose15(track: GlobalTrack) -> bool:
    if track.current_pose_3d_15 is None:
        return False

    pose = np.asarray(track.current_pose_3d_15)

    return pose.shape == (15, 3) and np.isfinite(pose).all()


def valid_pose_tracks(tracks: Iterable[GlobalTrack]) -> List[GlobalTrack]:
    return [
        track
        for track in tracks
        if track.state != TrackState.lost and has_valid_pose15(track)
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

        for joint_name in joint_names_15:
            header.extend([
                f"{joint_name}_x",
                f"{joint_name}_y",
                f"{joint_name}_z",
            ])

        return header

    def write_frame(self, frame_id: int, tracks: Iterable[GlobalTrack]) -> None:
        for track in valid_pose_tracks(tracks):
            pose = np.asarray(track.current_pose_3d_15, dtype=np.float32)

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

    for joint_a, joint_b in skeleton_15:
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

            color = color_for_id(track.global_id)
            label = f"ID {track.global_id}"

            if camera_name in track.current_bbox:
                draw_bbox(
                    image=image,
                    bbox=track.current_bbox[camera_name],
                    color=color,
                    label=label,
                )

            if has_valid_pose15(track):
                pose15 = np.asarray(track.current_pose_3d_15, dtype=np.float32)

                try:
                    projected = project_points(pose15, camera.P)
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


def compute_scene_limits(
    tracks: Iterable[GlobalTrack],
    padding: float = 500.0,
    min_radius: float = 1000.0,
) -> Optional[Dict[str, tuple[float, float]]]:
    points = []

    for track in valid_pose_tracks(tracks):
        pose = np.asarray(track.current_pose_3d_15, dtype=np.float32)
        points.append(pose)

    if not points:
        return None

    points_array = np.concatenate(points, axis=0)

    x_min, y_min, z_min = np.min(points_array, axis=0)
    x_max, y_max, z_max = np.max(points_array, axis=0)

    center = np.array([
        0.5 * (x_min + x_max),
        0.5 * (y_min + y_max),
        0.5 * (z_min + z_max),
    ])

    radius = max(
        float(x_max - x_min),
        float(y_max - y_min),
        float(z_max - z_min),
    ) / 2.0

    radius = max(radius + padding, min_radius)

    return {
        "x": (float(center[0] - radius), float(center[0] + radius)),
        "y": (float(center[1] - radius), float(center[1] + radius)),
        "z": (float(center[2] - radius), float(center[2] + radius)),
    }


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

    if not tracks:
        return

    if axis_limits is None:
        axis_limits = compute_scene_limits(tracks)

    if axis_limits is None:
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for track in tracks:
        pose = np.asarray(track.current_pose_3d_15, dtype=np.float32)
        color = bgr_to_rgb(color_for_id(track.global_id))

        ax.scatter(
            pose[:, 0],
            pose[:, 1],
            pose[:, 2],
            s=35,
            color=color,
        )

        for joint_a, joint_b in skeleton_15:
            ax.plot(
                [pose[joint_a, 0], pose[joint_b, 0]],
                [pose[joint_a, 1], pose[joint_b, 1]],
                [pose[joint_a, 2], pose[joint_b, 2]],
                linewidth=2,
                color=color,
            )

        root = pose[pelvis_15_index]

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

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(f"{activity_name} | frame {frame_id}")

    ax.set_box_aspect((
        axis_limits["x"][1] - axis_limits["x"][0],
        axis_limits["y"][1] - axis_limits["y"][0],
        axis_limits["z"][1] - axis_limits["z"][0],
    ))

    ax.view_init(elev=10, azim=-90)

    output_path = (
        output_dir
        / activity_name
        / f"frame_{frame_id:06d}_3d.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


class ActivityOutputs:
    def __init__(
        self,
        activity_name: str,
        csv_dir: str | Path,
        debug_dir: str | Path,
        save_csv: bool = True,
        save_failures: bool = True,
    ) -> None:
        self.activity_name = activity_name
        self.csv_dir = Path(csv_dir)
        self.debug_dir = Path(debug_dir)

        self.pose_writer: Optional[PoseCsvWriter] = None
        self.failure_writer: Optional[FailureCsvWriter] = None

        if save_csv:
            self.pose_writer = PoseCsvWriter(
                self.csv_dir / f"{activity_name}_keypoints3d_multiperson.csv"
            )

        if save_failures:
            self.failure_writer = FailureCsvWriter(
                self.debug_dir / f"{activity_name}_failed_poses.csv"
            )

    def write_pose_frame(
        self,
        frame_id: int,
        tracks: Iterable[GlobalTrack],
    ) -> None:
        if self.pose_writer is None:
            return

        self.pose_writer.write_frame(frame_id, tracks)

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

    def close(self) -> None:
        if self.pose_writer is not None:
            self.pose_writer.close()

        if self.failure_writer is not None:
            self.failure_writer.close()


def draw_predicted_3d_position(
    image: np.ndarray,
    track: GlobalTrack,
    camera: CameraData,
    color: tuple[int, int, int],
) -> None:
    if track.mean_3d is None:
        return

    if track.time_since_update > 30:
        return

    try:
        point_3d = track.mean_3d[:3].reshape(1, 3)
        point_2d = project_points(point_3d, camera.P)[0]
    except Exception:
        return

    x, y = point_2d

    if not np.isfinite(x) or not np.isfinite(y):
        return

    height, width = image.shape[:2]

    x_i = int(round(float(x)))
    y_i = int(round(float(y)))

    if 0 <= x_i < width and 0 <= y_i < height:
        cv2.circle(image, (x_i, y_i), 12, color, -1)
        cv2.putText(
            image,
            f"ID {track.global_id}?",
            (x_i + 15, y_i + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


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

        color = color_for_id(track.global_id)
        label = f"ID {track.global_id}"

        if track.state == TrackState.occluded:
            label += " [occ]"
        elif track.is_occluded_by is not None:
            label += f" [occ by {track.is_occluded_by}]"

        if camera_name in track.current_bbox:
            draw_bbox(
                image=image,
                bbox=track.current_bbox[camera_name],
                color=color,
                label=label,
            )

        # Résultat principal : pose Iskakov / volumetric triangulation reprojetée.
        if has_valid_pose15(track):
            pose15 = np.asarray(track.current_pose_3d_15, dtype=np.float32)

            try:
                projected = project_points(pose15, camera.P)
                draw_projected_skeleton(
                    image=image,
                    points_2d=projected,
                    color=(255, 0, 0),
                )
            except Exception:
                pass

        elif track.state == TrackState.occluded and track.has_3d:
            draw_predicted_3d_position(
                image=image,
                track=track,
                camera=camera,
                color=(255, 100, 100),
            )

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