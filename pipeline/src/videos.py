from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np


Frame = np.ndarray


@dataclass(frozen=True)
class VideoInfo:
    camera_name: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class ActivityInput:
    name: str
    xcp_path: Path
    video_paths_by_camera: Dict[str, Path]


def find_xcp_files(base_dir: str | Path) -> List[Path]:
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Dossier vidéo introuvable: {base_dir}")

    direct_matches = sorted(base_dir.glob("*.xcp"))

    if direct_matches:
        return direct_matches

    return sorted(base_dir.rglob("*.xcp"))


def find_video_for_camera(
    base_dir: str | Path,
    activity_name: str,
    device_id: str,
    video_extensions: List[str],
) -> Optional[Path]:
    base_dir = Path(base_dir)

    patterns = []

    for extension in video_extensions:
        patterns.append(f"{activity_name}.{device_id}*{extension}")

    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))
        if matches:
            return matches[0]

    for pattern in patterns:
        matches = sorted(base_dir.rglob(pattern))
        if matches:
            return matches[0]

    return None


def discover_activities(
    base_dir: str | Path,
    camera_names: List[str],
    device_id_by_camera: Dict[str, str],
    video_extensions: List[str],
    activity_mode: str = "all",
    selected_activity_name: Optional[str] = None,
) -> List[ActivityInput]:
    base_dir = Path(base_dir)

    if activity_mode not in {"all", "single"}:
        raise ValueError(
            f"activity_mode doit être 'all' ou 'single', reçu: {activity_mode}"
        )

    xcp_files = find_xcp_files(base_dir)

    if activity_mode == "single":
        if selected_activity_name is None or selected_activity_name.strip() == "":
            raise ValueError(
                "selected_activity_name doit être renseigné quand activity_mode='single'"
            )

        xcp_files = [
            xcp_path for xcp_path in xcp_files
            if xcp_path.stem == selected_activity_name
        ]

        if not xcp_files:
            raise FileNotFoundError(
                f"Aucun XCP trouvé pour l'activité '{selected_activity_name}' "
                f"dans {base_dir}"
            )

    activities: List[ActivityInput] = []

    for xcp_path in xcp_files:
        activity_name = xcp_path.stem
        video_paths_by_camera: Dict[str, Path] = {}
        missing_cameras = []

        for camera_name in camera_names:
            device_id = device_id_by_camera[camera_name]

            video_path = find_video_for_camera(
                base_dir=base_dir,
                activity_name=activity_name,
                device_id=device_id,
                video_extensions=video_extensions,
            )

            if video_path is None:
                missing_cameras.append(camera_name)
            else:
                video_paths_by_camera[camera_name] = video_path

        if missing_cameras:
            print(
                f"[skip] {activity_name}: vidéos manquantes pour "
                f"{missing_cameras}"
            )
            continue

        activities.append(
            ActivityInput(
                name=activity_name,
                xcp_path=xcp_path,
                video_paths_by_camera=video_paths_by_camera,
            )
        )

    return activities


class MultiCameraVideos:
    def __init__(
        self,
        video_paths_by_camera: Dict[str, str | Path],
        camera_names: List[str],
    ) -> None:
        self.video_paths_by_camera = {
            camera_name: Path(path)
            for camera_name, path in video_paths_by_camera.items()
        }

        self.camera_names = list(camera_names)
        self.captures: Dict[str, cv2.VideoCapture] = {}
        self.infos: Dict[str, VideoInfo] = {}

        self.open()

    def open(self) -> None:
        for camera_name in self.camera_names:
            if camera_name not in self.video_paths_by_camera:
                raise KeyError(f"Chemin vidéo manquant pour {camera_name}")

            path = self.video_paths_by_camera[camera_name]

            if not path.exists():
                raise FileNotFoundError(f"Vidéo introuvable pour {camera_name}: {path}")

            capture = cv2.VideoCapture(str(path))

            if not capture.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir la vidéo {camera_name}: {path}")

            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                capture.release()
                raise RuntimeError(
                    f"Taille invalide pour {camera_name}: {width}x{height}"
                )

            if frame_count <= 0:
                capture.release()
                raise RuntimeError(
                    f"Nombre de frames invalide pour {camera_name}: {frame_count}"
                )

            if fps <= 0 or np.isnan(fps):
                fps = 30.0

            self.captures[camera_name] = capture
            self.infos[camera_name] = VideoInfo(
                camera_name=camera_name,
                path=path,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
            )

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()

        self.captures.clear()

    def __enter__(self) -> "MultiCameraVideos":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def min_frame_count(self) -> int:
        return min(info.frame_count for info in self.infos.values())

    @property
    def fps(self) -> float:
        first_camera = self.camera_names[0]
        return self.infos[first_camera].fps

    def print_summary(self) -> None:
        print("")
        print("Résumé des vidéos")

        for camera_name in self.camera_names:
            info = self.infos[camera_name]
            print(
                f"{camera_name}: "
                f"{info.path} | "
                f"{info.width}x{info.height} | "
                f"{info.fps:.2f} fps | "
                f"{info.frame_count} frames"
            )

        print(f"frames utilisables: {self.min_frame_count}")
        print(f"fps de référence: {self.fps:.2f}")
        print("")

    def read_camera(self, camera_name: str, frame_id: int) -> Frame:
        if frame_id < 1:
            raise ValueError(f"frame_id doit être >= 1, reçu: {frame_id}")

        if camera_name not in self.captures:
            raise KeyError(f"Caméra inconnue: {camera_name}")

        info = self.infos[camera_name]

        if frame_id > info.frame_count:
            raise IndexError(
                f"frame_id={frame_id} dépasse la vidéo {camera_name}, "
                f"qui contient {info.frame_count} frames"
            )

        frame_index = frame_id - 1
        capture = self.captures[camera_name]
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = capture.read()

        if not ok or frame is None:
            raise RuntimeError(
                f"Impossible de lire frame_id={frame_id} pour {camera_name}"
            )

        return frame

    def read(self, frame_id: int) -> Dict[str, Frame]:
        if frame_id > self.min_frame_count:
            raise IndexError(
                f"frame_id={frame_id} dépasse le nombre de frames utilisables "
                f"({self.min_frame_count})"
            )

        return {
            camera_name: self.read_camera(camera_name, frame_id)
            for camera_name in self.camera_names
        }

    def iter_frames(
        self,
        start_frame: int = 1,
        end_frame: Optional[int] = None,
        stride: int = 1,
    ) -> Iterator[Tuple[int, Dict[str, Frame]]]:
        if stride <= 0:
            raise ValueError(f"stride doit être > 0, reçu: {stride}")

        if end_frame is None:
            end_frame = self.min_frame_count

        for frame_id in range(start_frame, end_frame + 1, stride):
            yield frame_id, self.read(frame_id)

    def save_debug_grid(
        self,
        frame_id: int,
        output_path: str | Path,
        resize_width: int = 480,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames = self.read(frame_id)
        resized_frames = []

        for camera_name in self.camera_names:
            frame = frames[camera_name]
            height, width = frame.shape[:2]

            scale = resize_width / float(width)
            resized_height = int(round(height * scale))

            resized = cv2.resize(
                frame,
                (resize_width, resized_height),
                interpolation=cv2.INTER_AREA,
            )

            cv2.putText(
                resized,
                f"{camera_name} | frame {frame_id}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            resized_frames.append(resized)

        max_height = max(image.shape[0] for image in resized_frames)

        padded_frames = []
        for image in resized_frames:
            height, width = image.shape[:2]

            if height < max_height:
                padding = np.zeros(
                    (max_height - height, width, 3),
                    dtype=image.dtype,
                )
                image = np.vstack([image, padding])

            padded_frames.append(image)

        grid = np.hstack(padded_frames)

        ok = cv2.imwrite(str(output_path), grid)

        if not ok:
            raise RuntimeError(f"Impossible d'écrire l'image debug: {output_path}")