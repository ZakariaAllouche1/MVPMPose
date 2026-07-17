from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


CUSTOM18_TO_HALPE26: Dict[int, int] = {
    0: 17,   # Head
    2: 19,   # Pelvis (hip center)
    3: 12,   # Hip_R
    4: 11,   # Hip_L
    5: 14,   # Knee_R
    6: 13,   # Knee_L
    7: 16,   # Ankle_R
    8: 15,   # Ankle_L
    9: 6,    # Shoulder_R
    10: 5,   # Shoulder_L
    11: 8,   # Elbow_R
    12: 7,   # Elbow_L
    13: 10,  # Wrist_R
    14: 9,   # Wrist_L
    15: 18,  # Neck
}


def project_points(P: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    n = points_world.shape[0]
    homog = np.hstack([points_world, np.ones((n, 1), dtype=np.float64)])
    proj = homog @ P.T  # N x 3
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return proj[:, :2] / w


def bbox_diagonal(bbox_xyxy) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return float(np.hypot(x2 - x1, y2 - y1))


def reprojection_confidence(
    keypoints18_world: np.ndarray,
    pose_2d_by_camera: Dict[str, np.ndarray],
    conf_by_camera: Dict[str, np.ndarray],
    bbox_by_camera: Dict[str, Tuple[float, float, float, float]],
    P_by_camera: Dict[str, np.ndarray],
    min_joint_conf: float = 0.30,
    max_norm_err: float = 0.10,
    min_joints_per_view: int = 4,
) -> Tuple[float, float, int]:

    keypoints18_world = np.asarray(keypoints18_world, dtype=np.float64)
    if keypoints18_world.shape[-2:] != (18, 3):
        return 0.0, float("inf"), 0

    per_view_err = []

    for camera_name, P in P_by_camera.items():
        pose_2d = pose_2d_by_camera.get(camera_name)
        confs = conf_by_camera.get(camera_name)
        bbox = bbox_by_camera.get(camera_name)
        if pose_2d is None or confs is None or bbox is None:
            continue

        pose_2d = np.asarray(pose_2d, dtype=np.float64)
        confs = np.asarray(confs, dtype=np.float64)
        diag = bbox_diagonal(bbox)
        if diag <= 1.0:
            continue

        custom_idx = [
            c for c, h in CUSTOM18_TO_HALPE26.items()
            if h < pose_2d.shape[0] and confs[h] >= min_joint_conf
        ]
        if len(custom_idx) < min_joints_per_view:
            continue

        halpe_idx = [CUSTOM18_TO_HALPE26[c] for c in custom_idx]
        proj = project_points(P, keypoints18_world[custom_idx, :])  # K x 2
        detected = pose_2d[halpe_idx, :2]

        err_px = np.linalg.norm(proj - detected, axis=1)
        per_view_err.append(float(np.mean(err_px / diag)))

    if not per_view_err:
        return 0.0, float("inf"), 0

    mean_view_err = float(np.mean(per_view_err))
    confidence = float(np.clip(1.0 - mean_view_err / max_norm_err, 0.0, 1.0))
    return confidence, mean_view_err, len(per_view_err)


def view_count_penalty(n_views: int, full_views: int = 3) -> float:
    if n_views <= 0:
        return 0.0
    return float(min(1.0, n_views / float(max(1, full_views))))


def pose_confidence(
    keypoints18_world: np.ndarray,
    pose_2d_by_camera: Dict[str, np.ndarray],
    conf_by_camera: Dict[str, np.ndarray],
    bbox_by_camera: Dict[str, Tuple[float, float, float, float]],
    P_by_camera: Dict[str, np.ndarray],
    n_views: int,
    min_joint_conf: float = 0.30,
    max_norm_err: float = 0.10,
    full_views_for_penalty: int = 3,
) -> Tuple[float, Dict[str, float]]:

    reproj, med_err, n_terms = reprojection_confidence(
        keypoints18_world=keypoints18_world,
        pose_2d_by_camera=pose_2d_by_camera,
        conf_by_camera=conf_by_camera,
        bbox_by_camera=bbox_by_camera,
        P_by_camera=P_by_camera,
        min_joint_conf=min_joint_conf,
        max_norm_err=max_norm_err,
    )
    penalty = view_count_penalty(n_views, full_views=full_views_for_penalty)

    if n_terms == 0:

        score = penalty
    else:
        score = reproj * penalty

    details = {
        "reprojection_confidence": reproj,
        "median_norm_err": med_err,
        "n_terms": float(n_terms),
        "view_penalty": penalty,
        "n_views": float(n_views),
    }
    return float(score), details
