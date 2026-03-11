# xcp_utils.py
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R_scipy


def parse_xcp(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    cameras = {}

    for camera_node in root.findall("Camera"):
        if camera_node.get("ISDV") != "1":
            continue

        device_id = camera_node.get("DEVICEID")
        keyframe = camera_node.find("KeyFrames/KeyFrame")
        if keyframe is None:
            continue

        t_world = np.array(
            [float(p) for p in keyframe.get("POSITION").split()],
            dtype=np.float64
        )

        q = [float(x) for x in keyframe.get("ORIENTATION").split()]
        quat_variants = [
            [q[1], q[2], q[3], q[0]],
            q
        ]

        focal_length = float(keyframe.get("FOCAL_LENGTH"))
        cx, cy = map(float, keyframe.get("PRINCIPAL_POINT").split())

        K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        variants = {}
        for idx, quat in enumerate(quat_variants, 1):
            try:
                Rq = R_scipy.from_quat(quat).as_matrix()
                if abs(np.linalg.det(Rq) - 1) < 0.1:
                    variants[f"{idx}A"] = {
                        "R": Rq.T,
                        "t": -Rq.T @ t_world.reshape(3, 1)
                    }
                    variants[f"{idx}B"] = {
                        "R": Rq,
                        "t": -Rq @ t_world.reshape(3, 1)
                    }
            except Exception:
                pass

        cameras[device_id] = {
            "K": K,
            "variants": variants
        }

    return cameras


def load_cameras_params_from_xcp(
    xcp_path,
    deviceid_by_name,
    camera_names,
    variant="2B",
    t_scale=1.0
):
    parsed = parse_xcp(xcp_path)
    cameras_params = []

    for name in camera_names:
        did = deviceid_by_name[name]

        if did not in parsed:
            raise KeyError(f"DEVICEID {did} absent du XCP")

        if variant not in parsed[did]["variants"]:
            raise KeyError(f"Variant {variant} absente pour {did}")

        extr = parsed[did]["variants"][variant]

        cameras_params.append({
            "name": name,
            "K": parsed[did]["K"].astype(np.float64),
            "R": extr["R"].astype(np.float64),
            "t": (extr["t"] * t_scale).astype(np.float64),
            "dist": np.zeros(5, dtype=np.float64),
        })

    return cameras_params