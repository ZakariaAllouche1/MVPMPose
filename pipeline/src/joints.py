from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


joint_names_18: List[str] = [
    "Head",
    "Trunk",
    "Pelvis",
    "Hip_R",
    "Hip_L",
    "Knee_R",
    "Knee_L",
    "Ankle_R",
    "Ankle_L",
    "Shoulder_R",
    "Shoulder_L",
    "Elbow_R",
    "Elbow_L",
    "Wrist_R",
    "Wrist_L",
    "Neck",
    "Foot_R",
    "Foot_L",
]


pelvis_18_index = 2


skeleton_18: List[Tuple[int, int]] = [
    (0, 15),   # Head - Neck
    (15, 1),   # Neck - Trunk
    (1, 2),    # Trunk - Pelvis

    (15, 9),   # Neck - Shoulder_R
    (9, 11),   # Shoulder_R - Elbow_R
    (11, 13),  # Elbow_R - Wrist_R

    (15, 10),  # Neck - Shoulder_L
    (10, 12),  # Shoulder_L - Elbow_L
    (12, 14),  # Elbow_L - Wrist_L

    (2, 3),    # Pelvis - Hip_R
    (3, 5),    # Hip_R - Knee_R
    (5, 7),    # Knee_R - Ankle_R
    (7, 16),   # Ankle_R - Foot_R

    (2, 4),    # Pelvis - Hip_L
    (4, 6),    # Hip_L - Knee_L
    (6, 8),    # Knee_L - Ankle_L
    (8, 17),   # Ankle_L - Foot_L
]


model19_to_18_np = np.array(
    [
        9,   # Head
        8,   # Trunk / Thorax
        6,   # Pelvis
        2,   # Hip_R
        3,   # Hip_L
        1,   # Knee_R
        4,   # Knee_L
        0,   # Ankle_R
        5,   # Ankle_L
        12,  # Shoulder_R
        13,  # Shoulder_L
        11,  # Elbow_R
        14,  # Elbow_L
        10,  # Wrist_R
        15,  # Wrist_L
        16,  # Neck
        17,  # Foot_R
        18,  # Foot_L
    ],
    dtype=np.int64,
)


model19_to_18_torch_indices = torch.tensor(
    model19_to_18_np.tolist(),
    dtype=torch.long,
)


def model19_to_18_numpy(keypoints19: np.ndarray) -> np.ndarray:
    keypoints19 = np.asarray(keypoints19, dtype=np.float32)

    if keypoints19.shape[-2:] != (19, 3):
        raise ValueError(
            f"keypoints19 doit avoir shape [..., 19, 3], reçu: {keypoints19.shape}"
        )

    return keypoints19[..., model19_to_18_np, :].astype(np.float32)


def model19_to_18_torch(keypoints19: torch.Tensor) -> torch.Tensor:
    if keypoints19.shape[-2:] != (19, 3):
        raise ValueError(
            f"keypoints19 doit avoir shape [..., 19, 3], reçu: {tuple(keypoints19.shape)}"
        )

    indices = model19_to_18_torch_indices.to(keypoints19.device)
    return keypoints19.index_select(dim=-2, index=indices)


def get_root_position_18(keypoints18: np.ndarray) -> np.ndarray:
    keypoints18 = np.asarray(keypoints18, dtype=np.float32)

    if keypoints18.shape[-2:] != (18, 3):
        raise ValueError(
            f"keypoints18 doit avoir shape [..., 18, 3], reçu: {keypoints18.shape}"
        )

    return keypoints18[pelvis_18_index].astype(np.float32)

