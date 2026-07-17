from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch



# Squelette principal de la pipeline

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



custom19_names: List[str] = [
    "Ankle_R",      # 0
    "Knee_R",       # 1
    "Hip_R",        # 2
    "Hip_L",        # 3
    "Knee_L",       # 4
    "Ankle_L",      # 5
    "Pelvis",       # 6
    "Extra_07",     # 7, non utilisé dans custom18
    "Trunk",        # 8
    "Head",         # 9
    "Wrist_R",      # 10
    "Elbow_R",      # 11
    "Shoulder_R",   # 12
    "Shoulder_L",   # 13
    "Elbow_L",      # 14
    "Wrist_L",      # 15
    "Neck",         # 16
    "Foot_R",       # 17
    "Foot_L",       # 18
]

model19_to_18_np = np.array(
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


# Modèle multi-head

N_CUSTOM19 = 19
N_ECOMOCAP32 = 32
N_METASPORT32 = 32
N_MODEL83 = N_CUSTOM19 + N_ECOMOCAP32 + N_METASPORT32

MULTIHEAD_CUSTOM19_SLICE = slice(0, 19)
MULTIHEAD_ECOMOCAP32_SLICE = slice(19, 51)
MULTIHEAD_METASPORT32_SLICE = slice(51, 83)

multihead_slices: Dict[str, slice] = {
    "custom19": MULTIHEAD_CUSTOM19_SLICE,
    "ecomocap32": MULTIHEAD_ECOMOCAP32_SLICE,
    "metasport32": MULTIHEAD_METASPORT32_SLICE,
}

ecomocap32_names: List[str] = [
    "C7",
    "T10",
    "IJ",
    "STER",
    "OCCD",
    "OCCG",
    "EASD",
    "EASG",
    "EPSD",
    "EPSG",
    "AD",
    "AG",
    "UCOD",
    "RCOD",
    "UCOG",
    "RCOG",
    "UD",
    "RD",
    "UG",
    "RG",
    "MCD",
    "LCD",
    "MCG",
    "LCG",
    "MMD",
    "LMD",
    "MMG",
    "LMG",
    "MT2D",
    "MT5D",
    "MT2G",
    "MT5G",
]

# La head MetaSport correspond aux indices 51:83 de la sortie 83.
metasport32_names: List[str] = [
    "C7",
    "T10",
    "MAN",
    "STRN",
    "ROCC",
    "LOCC",
    "RASI",
    "LASI",
    "RPSI",
    "LPSI",
    "RA",
    "LA",
    "RME",
    "RLE",
    "LME",
    "LLE",
    "RRAD",
    "RULN",
    "LRAD",
    "LULN",
    "RMC",
    "RLC",
    "LMC",
    "LLC",
    "RMM",
    "RLM",
    "LMM",
    "LLM",
    "RMT1",
    "RMT5",
    "LMT1",
    "LMT5",
]

multihead_names: Dict[str, List[str]] = {
    "custom19": custom19_names,
    "ecomocap32": ecomocap32_names,
    "metasport32": metasport32_names,
}


# Validations

def _check_numpy_shape(array: np.ndarray, expected_points: int, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)

    if array.shape[-2:] != (expected_points, 3):
        raise ValueError(
            f"{name} doit avoir shape [..., {expected_points}, 3], reçu: {array.shape}"
        )

    return array


def _check_torch_shape(tensor: torch.Tensor, expected_points: int, name: str) -> torch.Tensor:
    if tensor.shape[-2:] != (expected_points, 3):
        raise ValueError(
            f"{name} doit avoir shape [..., {expected_points}, 3], reçu: {tuple(tensor.shape)}"
        )

    return tensor


def model19_to_18_numpy(keypoints19: np.ndarray) -> np.ndarray:
    keypoints19 = _check_numpy_shape(keypoints19, 19, "keypoints19")
    return keypoints19[..., model19_to_18_np, :].astype(np.float32)


def model19_to_18_torch(keypoints19: torch.Tensor) -> torch.Tensor:
    keypoints19 = _check_torch_shape(keypoints19, 19, "keypoints19")
    indices = model19_to_18_torch_indices.to(keypoints19.device)
    return keypoints19.index_select(dim=-2, index=indices)


custom19_to_18_numpy = model19_to_18_numpy
custom19_to_18_torch = model19_to_18_torch


def split_model83_numpy(keypoints83: np.ndarray) -> Dict[str, np.ndarray]:
    keypoints83 = _check_numpy_shape(keypoints83, N_MODEL83, "keypoints83")

    return {
        "custom19": keypoints83[..., MULTIHEAD_CUSTOM19_SLICE, :].astype(np.float32),
        "ecomocap32": keypoints83[..., MULTIHEAD_ECOMOCAP32_SLICE, :].astype(np.float32),
        "metasport32": keypoints83[..., MULTIHEAD_METASPORT32_SLICE, :].astype(np.float32),
    }


def model83_to_custom19_numpy(keypoints83: np.ndarray) -> np.ndarray:
    return split_model83_numpy(keypoints83)["custom19"]


def model83_to_ecomocap32_numpy(keypoints83: np.ndarray) -> np.ndarray:
    return split_model83_numpy(keypoints83)["ecomocap32"]


def model83_to_metasport32_numpy(keypoints83: np.ndarray) -> np.ndarray:
    return split_model83_numpy(keypoints83)["metasport32"]


def model83_to_18_numpy(keypoints83: np.ndarray) -> np.ndarray:
    custom19 = model83_to_custom19_numpy(keypoints83)
    return custom19_to_18_numpy(custom19)


def split_model83_torch(keypoints83: torch.Tensor) -> Dict[str, torch.Tensor]:
    keypoints83 = _check_torch_shape(keypoints83, N_MODEL83, "keypoints83")

    return {
        "custom19": keypoints83[..., MULTIHEAD_CUSTOM19_SLICE, :],
        "ecomocap32": keypoints83[..., MULTIHEAD_ECOMOCAP32_SLICE, :],
        "metasport32": keypoints83[..., MULTIHEAD_METASPORT32_SLICE, :],
    }


def model83_to_custom19_torch(keypoints83: torch.Tensor) -> torch.Tensor:
    return split_model83_torch(keypoints83)["custom19"]


def model83_to_ecomocap32_torch(keypoints83: torch.Tensor) -> torch.Tensor:
    return split_model83_torch(keypoints83)["ecomocap32"]


def model83_to_metasport32_torch(keypoints83: torch.Tensor) -> torch.Tensor:
    return split_model83_torch(keypoints83)["metasport32"]


def model83_to_18_torch(keypoints83: torch.Tensor) -> torch.Tensor:
    custom19 = model83_to_custom19_torch(keypoints83)
    return custom19_to_18_torch(custom19)


def get_root_position_18(keypoints18: np.ndarray) -> np.ndarray:
    keypoints18 = _check_numpy_shape(keypoints18, 18, "keypoints18")
    return keypoints18[..., pelvis_18_index, :].astype(np.float32)


def get_root_position_model83(keypoints83: np.ndarray) -> np.ndarray:
    keypoints18 = model83_to_18_numpy(keypoints83)
    return get_root_position_18(keypoints18)
