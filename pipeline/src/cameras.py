from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rotation


@dataclass(frozen=True)
class CameraData:
    name: str
    device_id: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray
    P: np.ndarray


def ensure_column_vector(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)

    if t.ndim == 1:
        return t.reshape(3, 1)

    if t.shape == (1, 3):
        return t.reshape(3, 1)

    return t


def read_vicon_distortion(value: Optional[str]) -> np.ndarray:
    dist = np.zeros(5, dtype=np.float64)

    if value is None:
        return dist

    parts = value.split()

    if len(parts) >= 5:
        try:
            dist[0] = float(parts[-3])
            dist[1] = float(parts[-2])
            dist[4] = float(parts[-1])
        except Exception:
            pass

    return dist


def parse_xcp_variants(xcp_path: str | Path, t_scale: float = 1.0) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    xcp_path = Path(xcp_path)

    if not xcp_path.exists():
        raise FileNotFoundError(f"Fichier XCP introuvable: {xcp_path}")

    tree = ET.parse(xcp_path)
    root = tree.getroot()

    cameras: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for camera_node in root.findall("Camera"):
        if camera_node.get("ISDV") != "1":
            continue

        device_id = camera_node.get("DEVICEID")

        if device_id is None:
            continue

        keyframe = camera_node.find("KeyFrames/KeyFrame")

        if keyframe is None:
            continue

        position_text = keyframe.get("POSITION")
        orientation_text = keyframe.get("ORIENTATION")
        focal_text = keyframe.get("FOCAL_LENGTH")
        principal_text = keyframe.get("PRINCIPAL_POINT")

        if not position_text or not orientation_text or not focal_text or not principal_text:
            continue

        position = np.array(
            [float(value) for value in position_text.split()],
            dtype=np.float64,
        ) * float(t_scale)

        quaternion_raw = [float(value) for value in orientation_text.split()]

        quaternion_variants = [
            [quaternion_raw[1], quaternion_raw[2], quaternion_raw[3], quaternion_raw[0]],
            quaternion_raw,
        ]

        focal_length = float(focal_text)
        cx, cy = [float(value) for value in principal_text.split()]

        K = np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dist = read_vicon_distortion(keyframe.get("VICON_RADIAL2"))
        variants: Dict[str, Dict[str, np.ndarray]] = {}

        for index, quaternion in enumerate(quaternion_variants, start=1):
            try:
                R_world = Rotation.from_quat(quaternion).as_matrix()
            except Exception:
                continue

            if abs(np.linalg.det(R_world) - 1.0) > 0.1:
                continue

            t_world = position.reshape(3, 1)

            R_a = R_world.T
            t_a = -R_world.T @ t_world

            R_b = R_world
            t_b = -R_world @ t_world

            variants[f"{index}A"] = {
                "K": K,
                "R": R_a,
                "t": t_a,
                "dist": dist,
            }

            variants[f"{index}B"] = {
                "K": K,
                "R": R_b,
                "t": t_b,
                "dist": dist,
            }

        cameras[device_id] = variants

    return cameras


def load_cameras_from_xcp(
    xcp_path: str | Path,
    device_id_by_camera: Dict[str, str],
    variant: str = "2B",
    t_scale: float = 1.0,
) -> Dict[str, CameraData]:
    variants_by_device = parse_xcp_variants(xcp_path, t_scale=t_scale)

    cameras: Dict[str, CameraData] = {}

    for camera_name, device_id in device_id_by_camera.items():
        if device_id not in variants_by_device:
            raise KeyError(
                f"Device ID {device_id} introuvable dans le XCP "
                f"pour la caméra {camera_name}"
            )

        if variant not in variants_by_device[device_id]:
            available = sorted(variants_by_device[device_id].keys())
            raise KeyError(
                f"Variante {variant} indisponible pour {camera_name} "
                f"({device_id}). Variantes disponibles: {available}"
            )

        params = variants_by_device[device_id][variant]

        K = np.asarray(params["K"], dtype=np.float64)
        R = np.asarray(params["R"], dtype=np.float64)
        t = ensure_column_vector(params["t"])
        dist = np.asarray(params["dist"], dtype=np.float64).reshape(-1)

        P = K @ np.hstack([R, t])

        cameras[camera_name] = CameraData(
            name=camera_name,
            device_id=device_id,
            K=K,
            R=R,
            t=t,
            dist=dist,
            P=P,
        )

    return cameras


def print_camera_summary(cameras: Dict[str, CameraData]) -> None:
    print("")
    print("Résumé des caméras")

    for camera_name, camera in cameras.items():
        print(f"{camera_name}: device_id={camera.device_id}")
        print(f"  K: {camera.K.shape}")
        print(f"  R: {camera.R.shape}")
        print(f"  t: {camera.t.shape}")
        print(f"  dist: {camera.dist.shape}")
        print(f"  P: {camera.P.shape}")

    print("")