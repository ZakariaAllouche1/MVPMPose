from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


joint_names_15: List[str] = [
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
]


pelvis_15_index = 2


skeleton_15: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),

    (1, 9),
    (9, 11),
    (11, 13),

    (1, 10),
    (10, 12),
    (12, 14),

    (2, 3),
    (3, 5),
    (5, 7),

    (2, 4),
    (4, 6),
    (6, 8),
]


model17_from_15_np = np.array(
    [
        9,   # Head
        8,   # Trunk
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
    ],
    dtype=np.int64,
)


model17_from_15_torch = torch.tensor(
    model17_from_15_np.tolist(),
    dtype=torch.long,
)


def model17_to_15_numpy(keypoints17: np.ndarray) -> np.ndarray:
    keypoints17 = np.asarray(keypoints17, dtype=np.float32)

    if keypoints17.shape[-2:] != (17, 3):
        raise ValueError(
            f"keypoints17 doit avoir shape [..., 17, 3], reçu: {keypoints17.shape}"
        )

    return keypoints17[..., model17_from_15_np, :].astype(np.float32)


def model17_to_15_torch(keypoints17: torch.Tensor) -> torch.Tensor:
    if keypoints17.shape[-2:] != (17, 3):
        raise ValueError(
            f"keypoints17 doit avoir shape [..., 17, 3], reçu: {tuple(keypoints17.shape)}"
        )

    indices = model17_from_15_torch.to(keypoints17.device)
    return keypoints17.index_select(dim=-2, index=indices)


def get_root_position_15(keypoints15: np.ndarray) -> np.ndarray:
    keypoints15 = np.asarray(keypoints15, dtype=np.float32)

    if keypoints15.shape[-2:] != (15, 3):
        raise ValueError(
            f"keypoints15 doit avoir shape [..., 15, 3], reçu: {keypoints15.shape}"
        )

    return keypoints15[pelvis_15_index].astype(np.float32)