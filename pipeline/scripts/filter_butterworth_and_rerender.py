from __future__ import annotations

"""
Post-process an already completed pipeline run:
1) filter existing 3D CSVs with a Butterworth low-pass filter;
2) regenerate filtered vis3d images and filtered videos from the filtered CSVs.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, List, Tuple
import sys

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    import c3d
except Exception: 
    c3d = None

try:
    from scipy.signal import butter, sosfiltfilt
except ImportError as exc:
    raise SystemExit(
        "scipy is required for Butterworth filtering. Install it with: pip install scipy"
    ) from exc

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from configs import local as cfg
    from src.cameras import load_cameras_from_xcp
    from src.geometry import project_points
    from src.joints import joint_names_18, skeleton_18
    try:
        from src.joints import metasport32_names
    except ImportError:
        metasport32_names = [f"M{i}" for i in range(32)]
    from src.outputs import (
        color_for_id,
        metasport_color_for_id,
        draw_bbox,
        draw_projected_points,
        draw_3d_scene_frame,
        make_video_writer,
        bgr_to_rgb,
    )
    try:
        from src.outputs import draw_ground_reference
    except Exception:
        draw_ground_reference = None
    from src.joints import pelvis_18_index
    from src.tracking import TrackState
    from src.videos import MultiCameraVideos, discover_activities
except Exception as exc:  
    raise SystemExit(
        "Could not import the project modules. Run this script inside your pipeline project, "
        f"Import error: {exc}"
    ) from exc


@dataclass
class ButterworthConfig:
    enabled: bool = True
    order: int = 4
    cutoff_hz: float = 6.0
    min_segment_frames: int = 30
    max_gap_frames: int = 5


def butterworth_config_from_local(cfg_module: Any) -> ButterworthConfig:
    default_cutoff = getattr(
        cfg_module,
        "butterworth_default_cutoff_hz",
        getattr(cfg_module, "butterworth_cutoff_hz", 6.0),
    )
    return ButterworthConfig(
        enabled=bool(getattr(cfg_module, "butterworth_enabled", True)),
        order=int(getattr(cfg_module, "butterworth_order", 4)),
        cutoff_hz=float(default_cutoff),
        min_segment_frames=int(getattr(cfg_module, "butterworth_min_segment_frames", 30)),
        max_gap_frames=int(getattr(cfg_module, "butterworth_max_gap_frames", 5)),
    )


def activity_family_name(activity_name: str) -> str:

    parts = str(activity_name).split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return str(activity_name)


def infer_activity_cutoff_hz(cfg_module: Any, activity_name: str) -> float:
    """Choose the Butterworth cutoff from local.py, using the activity family."""
    mode = str(getattr(cfg_module, "butterworth_cutoff_mode", "activity_profile")).lower()

    if mode in {"fixed", "global", "single"}:
        return float(getattr(
            cfg_module,
            "butterworth_cutoff_hz",
            getattr(cfg_module, "butterworth_default_cutoff_hz", 6.0),
        ))

    default_cutoff = float(getattr(cfg_module, "butterworth_default_cutoff_hz", 6.0))
    mapping = getattr(cfg_module, "butterworth_activity_cutoff_hz", {})
    family = activity_family_name(activity_name)

    if family in mapping:
        return float(mapping[family])

    lower_family = family.lower()
    for key, value in mapping.items():
        key_str = str(key).lower()
        if lower_family == key_str or lower_family.startswith(key_str.lower()):
            return float(value)

    return default_cutoff


def butterworth_config_for_activity(cfg_module: Any, activity_name: str) -> ButterworthConfig:
    bw = butterworth_config_from_local(cfg_module)
    bw.cutoff_hz = infer_activity_cutoff_hz(cfg_module, activity_name)
    return bw


def coordinate_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if col.endswith("_x") or col.endswith("_y") or col.endswith("_z")
    ]


def sos_filter_1d(values: np.ndarray, fps: float, bw: ButterworthConfig) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not bw.enabled:
        return values

    finite_mask = np.isfinite(values)
    if finite_mask.sum() < bw.min_segment_frames:
        return values

    nyquist = 0.5 * float(fps)
    normalized_cutoff = float(bw.cutoff_hz) / nyquist
    normalized_cutoff = min(max(normalized_cutoff, 1e-5), 0.999)

    sos = butter(
        N=int(bw.order),
        Wn=normalized_cutoff,
        btype="lowpass",
        output="sos",
    )

    if len(values) < max(bw.min_segment_frames, int(bw.order) * 6):
        return values

    try:
        return sosfiltfilt(sos, values)
    except ValueError:
        return values


def split_continuous_segments(frame_ids: np.ndarray, max_gap_frames: int) -> list[np.ndarray]:
    if len(frame_ids) == 0:
        return []

    order = np.argsort(frame_ids)
    sorted_frames = frame_ids[order]
    breaks = np.where(np.diff(sorted_frames) > int(max_gap_frames) + 1)[0] + 1
    chunks = np.split(order, breaks)
    return [chunk for chunk in chunks if len(chunk) > 0]


def filter_pose_csv(input_csv: Path, output_csv: Path, fps: float, bw: ButterworthConfig) -> None:
    if not input_csv.exists():
        print(f"[butterworth] missing CSV, skipped: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    if df.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[butterworth] empty CSV copied: {output_csv}")
        return

    coord_cols = coordinate_columns(df)
    filtered = df.copy()

    for person_id, person_df in df.groupby("person_id", sort=False):
        person_df = person_df.sort_values("frame_id")
        original_indices = person_df.index.to_numpy()
        frame_ids = person_df["frame_id"].to_numpy(dtype=int)

        for segment_local_indices in split_continuous_segments(frame_ids, bw.max_gap_frames):
            segment_indices = original_indices[segment_local_indices]
            segment_df = df.loc[segment_indices].sort_values("frame_id")
            segment_indices = segment_df.index.to_numpy()
            segment_frames = segment_df["frame_id"].to_numpy(dtype=int)

            if len(segment_frames) < bw.min_segment_frames:
                continue

            full_frame_index = np.arange(segment_frames.min(), segment_frames.max() + 1)
            reindexed = segment_df.set_index("frame_id").reindex(full_frame_index)

            reindexed[coord_cols] = reindexed[coord_cols].interpolate(
                method="linear",
                limit=bw.max_gap_frames,
                limit_direction="both",
            )

            valid_rows = reindexed[coord_cols].notna().all(axis=1)
            if valid_rows.sum() < bw.min_segment_frames:
                continue

            filtered_reindexed = reindexed.copy()
            for col in coord_cols:
                values = reindexed[col].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    continue
                filtered_reindexed[col] = sos_filter_1d(values, fps=fps, bw=bw)

            for frame in segment_frames:
                if frame in filtered_reindexed.index:
                    row_index = segment_df.index[segment_df["frame_id"].to_numpy(dtype=int) == int(frame)]
                    if len(row_index) > 0:
                        filtered.loc[row_index[0], coord_cols] = filtered_reindexed.loc[frame, coord_cols].to_numpy(dtype=float)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    print(f"[butterworth] wrote: {output_csv}")


def load_tracking_debug_bboxes(csv_dir: Path, activity_name: str) -> Dict[tuple[int, int], Dict[str, tuple[int, int, int, int]]]:
    """Return {(frame_id, person_id): {camera: bbox_xyxy}} if tracking debug exists."""
    path = csv_dir / f"{activity_name}_global_tracking_debug.csv"
    if not path.exists():
        print(f"[rerender] no tracking debug CSV found, videos will draw skeletons without bboxes: {path}")
        return {}

    df = pd.read_csv(path)
    required = {"frame", "camera", "global_id", "x1", "y1", "w", "h"}
    if not required.issubset(df.columns):
        print(f"[rerender] tracking debug CSV has unexpected columns, bboxes skipped: {path}")
        return {}

    bboxes: Dict[tuple[int, int], Dict[str, tuple[int, int, int, int]]] = {}
    for row in df.itertuples(index=False):
        frame_id = int(getattr(row, "frame"))
        person_id = int(getattr(row, "global_id"))
        camera = str(getattr(row, "camera"))
        x1 = float(getattr(row, "x1"))
        y1 = float(getattr(row, "y1"))
        w = float(getattr(row, "w"))
        h = float(getattr(row, "h"))
        if not np.isfinite([x1, y1, w, h]).all():
            continue
        bboxes.setdefault((frame_id, person_id), {})[camera] = (
            int(round(x1)),
            int(round(y1)),
            int(round(x1 + w)),
            int(round(y1 + h)),
        )
    return bboxes


def pose_matrix_from_row(row: pd.Series, names: list[str]) -> Optional[np.ndarray]:
    points = []
    for name in names:
        cols = [f"{name}_x", f"{name}_y", f"{name}_z"]
        if not all(col in row.index for col in cols):
            return None
        points.append([float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])])
    pose = np.asarray(points, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 3 or not np.isfinite(pose).all():
        return None
    return pose


def rows_by_frame_person(df: Optional[pd.DataFrame]) -> Dict[tuple[int, int], pd.Series]:
    if df is None or df.empty:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        mapping[(int(row["frame_id"]), int(row["person_id"]))] = row
    return mapping

METASPORT_MAPPING_NAME = "METASPORT_CUSTOM18_PLUS_MARKERSET32_V1"
METASPORT_GT_ALPHA = 0.7
METASPORT_GT_BETA = 0.5
METASPORT_GT_GAMMA = 0.5

METASPORT_HEAD_POINTS: List[str] = [
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

if len(metasport32_names) == 32 and list(metasport32_names) != METASPORT_HEAD_POINTS:
    print("[GT warning] src.joints.metasport32_names != prepare_dataset METASPORT_HEAD_POINTS. "
          "GT markers will use prepare_dataset order; predicted CSV keeps src.joints order.")


def _unique_paths(paths: List[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def metasport_gt_activity_candidates(activity_name: str) -> List[str]:
    """Return possible file/folder names for the same MetaSport trial."""
    name = str(activity_name)
    candidates = [name]
    candidates.append(name.replace("_Polystyrene", "Polystyrene"))
    candidates.append(name.replace("Polystyrene", "_Polystyrene"))
    candidates.append(name.replace("SprintTapis", "Sprint"))
    candidates.append(name.replace("Sprint", "SprintTapis"))
    return list(dict.fromkeys([c for c in candidates if c]))


def infer_metasport_subject_root(cfg_module: Any) -> Optional[Path]:
    explicit = getattr(cfg_module, "gt_metasport_subject_root", None)
    if explicit is not None:
        p = Path(explicit)
        return p if p.exists() else p

    base_videos_dir = Path(getattr(cfg_module, "base_videos_dir", ""))
    for parent in [base_videos_dir, *base_videos_dir.parents]:
        if parent.name == "Files":
            return parent
    if len(base_videos_dir.parents) >= 2:
        return base_videos_dir.parents[1]
    return None


def find_metasport_gt_points_path(subject_root: Path, activity_name: str) -> Optional[Path]:
    candidates: List[Path] = []
    for name in metasport_gt_activity_candidates(activity_name):
        candidates.extend([
            subject_root / "resultats_cinemat" / name / "points3D_t.mat",
            subject_root / "résultats_cinemat" / name / "points3D_t.mat",
        ])
    for c in candidates:
        if c.exists():
            return c

    for name in metasport_gt_activity_candidates(activity_name):
        matches = sorted(subject_root.rglob(f"{name}/points3D_t.mat"))
        if matches:
            return matches[0]
    return None


def find_metasport_gt_c3d_path(subject_root: Path, activity_name: str) -> Optional[Path]:
    candidates: List[Path] = []
    for name in metasport_gt_activity_candidates(activity_name):
        candidates.extend([
            subject_root / "fichiers_cinemat" / f"{name}.c3d",
            subject_root / "fichiers_cinémat" / f"{name}.c3d",
            subject_root / f"{name}.c3d",
        ])
    for c in candidates:
        if c.exists():
            return c

    for name in metasport_gt_activity_candidates(activity_name):
        matches = sorted(subject_root.rglob(f"{name}.c3d"))
        if matches:
            return matches[0]
    return None


def load_mat_variable(mat_path: Path, variable_name: str):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if variable_name not in mat:
        raise KeyError(f"Variable '{variable_name}' absente dans {mat_path}")
    return mat[variable_name]


def matlab_struct_to_dict(struct_obj) -> Dict[str, np.ndarray]:
    if hasattr(struct_obj, "_fieldnames"):
        return {field: np.asarray(getattr(struct_obj, field)) for field in struct_obj._fieldnames}
    if isinstance(struct_obj, np.ndarray) and struct_obj.dtype.names is not None:
        return {field: np.asarray(struct_obj[field]) for field in struct_obj.dtype.names}
    raise TypeError("Format de struct MATLAB non supporté pour points3D_t")


def to_xyz(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).reshape(3)


def mean_markers(points_dict: Dict[str, np.ndarray], marker_names: List[str], frame_idx: int) -> np.ndarray:
    pts = [to_xyz(points_dict[name][frame_idx]) for name in marker_names]
    return np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32)


def check_required_fields(points_dict: Dict[str, np.ndarray], required_fields: List[str], context: str) -> None:
    missing = [f for f in required_fields if f not in points_dict]
    if missing:
        raise KeyError(f"{context}: champs manquants dans points3D_t: {missing}")


def get_marker_or_none(points_dict: Dict[str, np.ndarray], marker_name: str, frame_idx: int) -> Optional[np.ndarray]:
    if marker_name not in points_dict:
        return None
    arr = points_dict[marker_name]
    if frame_idx < 0 or frame_idx >= len(arr):
        return None
    v = to_xyz(arr[frame_idx])
    if not np.isfinite(v).all():
        return None
    if np.abs(v).sum() <= 0.1:
        return None
    return v.astype(np.float32)


def get_marker_finite_or_none(points_dict: Dict[str, np.ndarray], marker_name: str, frame_idx: int) -> Optional[np.ndarray]:
    if marker_name not in points_dict:
        return None
    arr = points_dict[marker_name]
    if frame_idx < 0 or frame_idx >= len(arr):
        return None
    v = to_xyz(arr[frame_idx])
    if not np.isfinite(v).all():
        return None
    return v.astype(np.float32)


def get_first_marker_or_none(points_dict: Dict[str, np.ndarray], marker_names: List[str], frame_idx: int) -> Optional[np.ndarray]:
    for marker_name in marker_names:
        v = get_marker_or_none(points_dict, marker_name, frame_idx)
        if v is not None:
            return v
    return None


def interpolate_with_fallback(point_a: Optional[np.ndarray], point_b: Optional[np.ndarray], weight_a: float, joint_name: str) -> np.ndarray:
    if point_a is not None and point_b is not None:
        return (weight_a * point_a + (1.0 - weight_a) * point_b).astype(np.float32)
    if point_a is not None:
        return point_a.astype(np.float32)
    if point_b is not None:
        return point_b.astype(np.float32)
    raise KeyError(f"Impossible de calculer {joint_name}: les deux points sont absents")


def compute_trunk_from_t10_and_front(points_dict: Dict[str, np.ndarray], frame_idx: int, front_markers: List[str]) -> np.ndarray:
    if "T10" not in points_dict:
        raise KeyError("T10 absent dans points3D_t")
    t10 = to_xyz(points_dict["T10"][frame_idx])
    front_pts = []
    for marker_name in front_markers:
        if marker_name not in points_dict:
            continue
        v = to_xyz(points_dict[marker_name][frame_idx])
        if np.isfinite(v).all():
            front_pts.append(v.astype(np.float32))
    if not front_pts:
        raise KeyError(f"Impossible de calculer Trunk, aucun marqueur antérieur fini parmi {front_markers}")
    front = np.mean(np.stack(front_pts, axis=0), axis=0).astype(np.float32)
    return (0.5 * (t10 + front)).astype(np.float32)


def compute_neck_from_c7_and_front(points_dict: Dict[str, np.ndarray], frame_idx: int, front_marker: str) -> np.ndarray:
    c7 = get_marker_finite_or_none(points_dict, "C7", frame_idx)
    front = get_marker_finite_or_none(points_dict, front_marker, frame_idx)
    if c7 is None or front is None:
        raise KeyError(f"Impossible de calculer Neck: C7 ou {front_marker} absent")
    return (0.5 * (c7 + front)).astype(np.float32)


def compute_metasport_hip_with_gamma(anterior_pelvis_marker: Optional[np.ndarray], femur_origin: Optional[np.ndarray], old_hjc: Optional[np.ndarray], gamma: float, joint_name: str) -> np.ndarray:
    if anterior_pelvis_marker is not None and femur_origin is not None:
        return (gamma * anterior_pelvis_marker + (1.0 - gamma) * femur_origin).astype(np.float32)
    if femur_origin is not None:
        return femur_origin.astype(np.float32)
    if old_hjc is not None:
        return old_hjc.astype(np.float32)
    if anterior_pelvis_marker is not None:
        return anterior_pelvis_marker.astype(np.float32)
    raise KeyError(f"Impossible de calculer {joint_name}: marqueurs absents")


def extract_metasport_18j(points_dict: Dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    """Exact MetaSport custom18 extraction from prepare_dataset_multihead_markerset32.py."""
    required = [
        "ROCC", "LOCC", "RTEMP", "LTEMP",
        "T10", "C7", "MAN", "STRN",
        "RASI", "LASI", "RPSI", "LPSI",
        "RHJC", "LHJC",
        "Obassin", "OfemurD", "OfemurG",
        "RLC", "RMC", "LLC", "LMC",
        "RLM", "RMM", "LLM", "LMM",
        "RA", "LA",
        "RME", "RLE", "LME", "LLE",
        "RRAD", "RULN", "LRAD", "LULN",
        "RMT1", "RMT5",
        "LMT1", "LMT5",
    ]
    check_required_fields(points_dict, required, METASPORT_MAPPING_NAME)

    joints: List[np.ndarray] = []
    joints.append(mean_markers(points_dict, ["ROCC", "LOCC", "RTEMP", "LTEMP"], frame_idx))
    joints.append(compute_trunk_from_t10_and_front(points_dict, frame_idx, ["MAN", "STRN"]))

    obassin = get_marker_or_none(points_dict, "Obassin", frame_idx)
    ofemur_r = get_marker_or_none(points_dict, "OfemurD", frame_idx)
    ofemur_l = get_marker_or_none(points_dict, "OfemurG", frame_idx)
    center_femurs = None
    if ofemur_r is not None and ofemur_l is not None:
        center_femurs = (0.5 * (ofemur_r + ofemur_l)).astype(np.float32)
    old_pelvis = mean_markers(points_dict, ["RASI", "LASI", "RPSI", "LPSI"], frame_idx)
    if obassin is not None or center_femurs is not None:
        pelvis = interpolate_with_fallback(obassin, center_femurs, METASPORT_GT_BETA, "Pelvis")
    else:
        pelvis = old_pelvis
    joints.append(pelvis)

    rasi = get_marker_or_none(points_dict, "RASI", frame_idx)
    lasi = get_marker_or_none(points_dict, "LASI", frame_idx)
    old_hip_r = get_marker_or_none(points_dict, "RHJC", frame_idx)
    old_hip_l = get_marker_or_none(points_dict, "LHJC", frame_idx)
    joints.append(compute_metasport_hip_with_gamma(rasi, ofemur_r, old_hip_r, METASPORT_GT_GAMMA, "Hip_R"))
    joints.append(compute_metasport_hip_with_gamma(lasi, ofemur_l, old_hip_l, METASPORT_GT_GAMMA, "Hip_L"))

    joints.append(mean_markers(points_dict, ["RLC", "RMC"], frame_idx))
    joints.append(mean_markers(points_dict, ["LLC", "LMC"], frame_idx))
    joints.append(mean_markers(points_dict, ["RLM", "RMM"], frame_idx))
    joints.append(mean_markers(points_dict, ["LLM", "LMM"], frame_idx))

    ra = get_marker_or_none(points_dict, "RA", frame_idx)
    la = get_marker_or_none(points_dict, "LA", frame_idx)
    sjc_r = get_first_marker_or_none(points_dict, ["SJCD", "ObrasD"], frame_idx)
    sjc_l = get_first_marker_or_none(points_dict, ["SJCG", "ObrasG"], frame_idx)
    joints.append(interpolate_with_fallback(ra, sjc_r, METASPORT_GT_ALPHA, "Shoulder_R"))
    joints.append(interpolate_with_fallback(la, sjc_l, METASPORT_GT_ALPHA, "Shoulder_L"))

    joints.append(mean_markers(points_dict, ["RME", "RLE"], frame_idx))
    joints.append(mean_markers(points_dict, ["LME", "LLE"], frame_idx))
    joints.append(mean_markers(points_dict, ["RRAD", "RULN"], frame_idx))
    joints.append(mean_markers(points_dict, ["LRAD", "LULN"], frame_idx))
    joints.append(compute_neck_from_c7_and_front(points_dict, frame_idx, "MAN"))
    joints.append(mean_markers(points_dict, ["RMT1", "RMT5"], frame_idx))
    joints.append(mean_markers(points_dict, ["LMT1", "LMT5"], frame_idx))

    arr = np.asarray(joints, dtype=np.float32)
    if arr.shape != (18, 3) or not np.isfinite(arr).all():
        raise ValueError(f"GT MetaSport custom18 invalide: shape={arr.shape}")
    return arr


def extract_custom_from_metasport_18j(points_dict: Dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    return extract_metasport_18j(points_dict, frame_idx)


def extract_marker_list_head(
    points_dict: Dict[str, np.ndarray],
    frame_idx: int,
    marker_names: List[str],
    context: str,
) -> np.ndarray:
    check_required_fields(points_dict, marker_names, context)
    pts = [to_xyz(points_dict[name][frame_idx]) for name in marker_names]
    arr = np.asarray(pts, dtype=np.float32)
    expected_shape = (len(marker_names), 3)
    if arr.shape != expected_shape:
        raise ValueError(f"{context}: shape inattendue {arr.shape}, attendu {expected_shape}")
    return arr


def extract_metasport_head_18j(points_dict: Dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    return extract_marker_list_head(
        points_dict=points_dict,
        frame_idx=frame_idx,
        marker_names=METASPORT_HEAD_POINTS,
        context="METASPORT_MARKERSET32_HEAD",
    )


def extract_metasport_gt_32(points_dict: Dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    arr = extract_metasport_head_18j(points_dict, frame_idx)
    if arr.shape != (32, 3):
        raise ValueError(f"GT MetaSport32 invalide: shape={arr.shape}")
    return arr


def finite_point_stats(points: Optional[np.ndarray]) -> str:
    if points is None:
        return "None"
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return f"bad shape={arr.shape}"
    valid = arr[np.isfinite(arr).all(axis=1)]
    if len(valid) == 0:
        return "no finite points"
    mn = np.min(valid, axis=0)
    mx = np.max(valid, axis=0)
    span = mx - mn
    center = np.mean(valid, axis=0)
    return (
        f"n={len(valid)} center=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) "
        f"span=({span[0]:.3f},{span[1]:.3f},{span[2]:.3f}) "
        f"min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
        f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})"
    )


def apply_gt_coordinate_transform(points: Optional[np.ndarray], cfg_module: Any) -> Optional[np.ndarray]:
    if points is None:
        return None
    arr = np.asarray(points, dtype=np.float32)
    scale = float(getattr(cfg_module, "gt_coordinate_scale", 1.0))
    offset = np.asarray(
        getattr(cfg_module, "gt_coordinate_offset", (0.0, 0.0, 0.0)),
        dtype=np.float32,
    ).reshape(1, 3)
    return arr * scale + offset

def load_metasport_c3d_timing(c3d_path: Path) -> Dict[str, int]:
    if c3d is None:
        raise ImportError("Le package c3d est requis pour lire le timing C3D")
    with open(c3d_path, "rb") as f:
        reader = c3d.Reader(f)
        first_frame = int(reader.header.first_frame)
        last_frame = int(reader.header.last_frame)
    n_c3d = last_frame - first_frame + 1
    return {"first_frame": first_frame, "last_frame": last_frame, "n_c3d": n_c3d}


def metasport_video_frame_to_mat_idx(frame_id_1based: int, c3d_timing: Dict[str, int], n_vicon_mat: int, vicon_video_ratio: float, manual_extra_offset: int = 0) -> int:
    first_frame = int(c3d_timing["first_frame"])
    n_c3d = int(c3d_timing["n_c3d"])
    mat_offset = n_c3d - int(n_vicon_mat)
    first_frame_0 = first_frame - 1
    vf = int(frame_id_1based) - 1
    vicon_abs = int(vf * float(vicon_video_ratio))
    c3d_idx = vicon_abs - first_frame_0
    mat_idx = c3d_idx - mat_offset
    mat_idx += int(manual_extra_offset)
    return int(mat_idx)


class MetaSportGTProvider:
    def __init__(self, cfg_module: Any, activity_name: str) -> None:
        self.cfg = cfg_module
        self.activity_name = str(activity_name)
        self.enabled = False
        self.points_dict: Dict[str, np.ndarray] = {}
        self.c3d_timing: Dict[str, int] = {}
        self.n_frames_gt = 0
        self.manual_extra_offset = 0
        self.vicon_video_ratio = float(getattr(cfg_module, "gt_vicon_frequency", 200.0)) / float(getattr(cfg_module, "gt_video_frequency", 100.0))
        self._debug_printed = False

        subject_root = infer_metasport_subject_root(cfg_module)
        if subject_root is None:
            print(f"[GT] impossible d'inférer la racine MetaSport pour {activity_name}")
            return

        points_path = find_metasport_gt_points_path(subject_root, self.activity_name)
        c3d_path = find_metasport_gt_c3d_path(subject_root, self.activity_name)

        if points_path is None:
            print(f"[GT] points3D_t introuvable pour {activity_name} sous {subject_root}")
            return
        if c3d_path is None:
            print(f"[GT] C3D introuvable pour {activity_name} sous {subject_root}")
            return

        offsets = getattr(cfg_module, "gt_metasport_frame_offsets", {})
        subject_name = str(getattr(cfg_module, "subject_name", ""))
        self.manual_extra_offset = int(
            offsets.get((subject_name, self.activity_name), offsets.get(self.activity_name, 0))
            if isinstance(offsets, dict) else 0
        )

        points_struct = load_mat_variable(points_path, "points3D_t")
        self.points_dict = matlab_struct_to_dict(points_struct)
        if "Obassin" not in self.points_dict:
            raise KeyError(f"'Obassin' absent dans {points_path}")
        self.n_frames_gt = int(np.asarray(self.points_dict["Obassin"]).shape[0])
        self.c3d_timing = load_metasport_c3d_timing(c3d_path)
        self.enabled = True

        mat_offset = int(self.c3d_timing["n_c3d"]) - int(self.n_frames_gt)
        print(
            f"[GT] {activity_name}: points={points_path} | c3d={c3d_path} | "
            f"n_mat={self.n_frames_gt} n_c3d={self.c3d_timing['n_c3d']} "
            f"mat_offset={mat_offset} manual_extra_offset={self.manual_extra_offset}"
        )

    def _gt_index_for_frame(self, frame_id: int) -> Optional[int]:
        if not self.enabled:
            return None
        gt_idx = metasport_video_frame_to_mat_idx(
            frame_id_1based=int(frame_id),
            c3d_timing=self.c3d_timing,
            n_vicon_mat=self.n_frames_gt,
            vicon_video_ratio=self.vicon_video_ratio,
            manual_extra_offset=self.manual_extra_offset,
        )
        if gt_idx < 0 or gt_idx >= self.n_frames_gt:
            return None
        return int(gt_idx)

    def raw_pose18_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        gt_idx = self._gt_index_for_frame(frame_id)
        if gt_idx is None:
            return None
        try:
            return extract_custom_from_metasport_18j(self.points_dict, gt_idx)
        except Exception as exc:
            print(f"[GT custom18] frame={frame_id} gt_idx={gt_idx} skipped: {exc}")
            return None

    def raw_pose32_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        gt_idx = self._gt_index_for_frame(frame_id)
        if gt_idx is None:
            return None
        try:
            return extract_metasport_gt_32(self.points_dict, gt_idx)
        except Exception as exc:
            print(f"[GT meta32] frame={frame_id} gt_idx={gt_idx} skipped: {exc}")
            return None

    def pose18_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        raw18 = self.raw_pose18_for_frame(frame_id)
        return apply_gt_coordinate_transform(raw18, self.cfg)

    def pose32_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        raw32 = self.raw_pose32_for_frame(frame_id)
        return apply_gt_coordinate_transform(raw32, self.cfg)

    def poses_for_frame(self, frame_id: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        raw18 = self.raw_pose18_for_frame(frame_id)
        raw32 = self.raw_pose32_for_frame(frame_id)
        pose18 = apply_gt_coordinate_transform(raw18, self.cfg)
        pose32 = apply_gt_coordinate_transform(raw32, self.cfg)

        if bool(getattr(self.cfg, "gt_debug_coordinates", False)) and not self._debug_printed:
            gt_idx = self._gt_index_for_frame(frame_id)
            scale = float(getattr(self.cfg, "gt_coordinate_scale", 1.0))
            offset = tuple(getattr(self.cfg, "gt_coordinate_offset", (0.0, 0.0, 0.0)))
            print(f"[GT debug] prepare_dataset extraction | activity={self.activity_name} frame={frame_id} gt_idx={gt_idx}")
            print(f"[GT debug] transform: scale={scale} offset={offset}")
            print(f"[GT debug] raw custom18: {finite_point_stats(raw18)}")
            print(f"[GT debug] raw metasport32: {finite_point_stats(raw32)}")
            print(f"[GT debug] scaled custom18: {finite_point_stats(pose18)}")
            print(f"[GT debug] scaled metasport32: {finite_point_stats(pose32)}")
            self._debug_printed = True

        return pose18, pose32


def build_gt_provider_if_needed(cfg_module: Any, activity_name: str) -> Optional[MetaSportGTProvider]:
    if not bool(getattr(cfg_module, "gt_show_comparison", False)):
        return None
    try:
        provider = MetaSportGTProvider(cfg_module, activity_name)
        return provider if provider.enabled else None
    except Exception as exc:
        print(f"[GT] disabled for {activity_name}: {exc}")
        return None


def _color_tuple_bgr(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        if isinstance(value, str):
            raise TypeError
        vals = tuple(int(v) for v in value)
        if len(vals) == 3:
            return vals
    except Exception:
        pass
    return tuple(int(v) for v in fallback)


def pred_skeleton_color_bgr(cfg_module: Any, person_id: int) -> tuple[int, int, int]:
    mapping = getattr(cfg_module, "comparison_pred_skeleton_colors_bgr", {})
    if isinstance(mapping, dict) and int(person_id) in mapping:
        return _color_tuple_bgr(mapping[int(person_id)], color_for_id(person_id))
    return color_for_id(person_id)


def pred_marker_color_bgr(cfg_module: Any, person_id: int) -> tuple[int, int, int]:
    mapping = getattr(cfg_module, "comparison_pred_marker_colors_bgr", {})
    if isinstance(mapping, dict) and int(person_id) in mapping:
        return _color_tuple_bgr(mapping[int(person_id)], metasport_color_for_id(person_id))
    return metasport_color_for_id(person_id)


def gt_skeleton_color_bgr(cfg_module: Any) -> tuple[int, int, int]:
    return _color_tuple_bgr(
        getattr(cfg_module, "comparison_gt_skeleton_color_bgr", getattr(cfg_module, "gt_color_bgr", (0, 165, 255))),
        (0, 165, 255),
    )


def gt_marker_color_bgr(cfg_module: Any) -> tuple[int, int, int]:
    return _color_tuple_bgr(
        getattr(cfg_module, "comparison_gt_marker_color_bgr", (255, 255, 0)),
        (255, 255, 0),
    )


def _draw_projected_skeleton_lines(image: np.ndarray, points_2d: np.ndarray, color: tuple[int, int, int], dashed: bool = False) -> None:
    height, width = image.shape[:2]
    for joint_a, joint_b in skeleton_18:
        pa = points_2d[joint_a]
        pb = points_2d[joint_b]
        if not np.isfinite(pa).all() or not np.isfinite(pb).all():
            continue
        xa, ya = int(round(float(pa[0]))), int(round(float(pa[1])))
        xb, yb = int(round(float(pb[0]))), int(round(float(pb[1])))
        if not (0 <= xa < width and 0 <= ya < height and 0 <= xb < width and 0 <= yb < height):
            continue

        if not dashed:
            cv2.line(image, (xa, ya), (xb, yb), color, 2)
            continue

        p0 = np.array([xa, ya], dtype=np.float32)
        p1 = np.array([xb, yb], dtype=np.float32)
        length = float(np.linalg.norm(p1 - p0))
        if length < 1.0:
            continue
        n = max(int(length // 12), 1)
        for k in range(n):
            if k % 2 != 0:
                continue
            t0 = k / float(n)
            t1 = min((k + 1) / float(n), 1.0)
            q0 = (1.0 - t0) * p0 + t0 * p1
            q1 = (1.0 - t1) * p0 + t1 * p1
            cv2.line(image, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), color, 2)


def _draw_projected_points(image: np.ndarray, points_2d: np.ndarray, color: tuple[int, int, int], radius: int = 4) -> None:
    height, width = image.shape[:2]
    for point in points_2d:
        if not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), int(radius), color, -1)


def draw_gt_video_overlay(
    image: np.ndarray,
    gt_pose18: Optional[np.ndarray],
    gt_pose32: Optional[np.ndarray],
    camera: Any,
    cfg_module: Any,
) -> None:
    if gt_pose18 is not None and bool(getattr(cfg_module, "comparison_show_gt_custom18", True)):
        try:
            projected18 = project_points(gt_pose18, camera.P)
        except Exception:
            projected18 = None
        if projected18 is not None:
            color18 = gt_skeleton_color_bgr(cfg_module)
            _draw_projected_skeleton_lines(image, projected18, color18, dashed=True)
            finite = projected18[np.isfinite(projected18).all(axis=1)]
            if len(finite) > 0:
                x, y = finite[0]
                cv2.putText(
                    image,
                    str(getattr(cfg_module, "comparison_gt_skeleton_label", "GT skeleton")),
                    (int(round(float(x))), int(round(float(y))) - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color18,
                    2,
                    cv2.LINE_AA,
                )

    if gt_pose32 is not None and bool(getattr(cfg_module, "comparison_show_gt_metasport32", True)):
        try:
            projected32 = project_points(gt_pose32, camera.P)
            _draw_projected_points(
                image,
                projected32,
                gt_marker_color_bgr(cfg_module),
                radius=int(getattr(cfg_module, "comparison_gt_marker_size_video", 4)),
            )
        except Exception:
            pass


def _axis_limits_from_points(points: List[np.ndarray]) -> Optional[Dict[str, tuple[float, float]]]:
    valid = [np.asarray(p, dtype=np.float32).reshape(-1, 3) for p in points if p is not None and np.asarray(p).size > 0]
    valid = [p[np.isfinite(p).all(axis=1)] for p in valid]
    valid = [p for p in valid if len(p) > 0]
    if not valid:
        return None
    pts = np.concatenate(valid, axis=0)
    x_min, y_min, z_min = np.min(pts, axis=0)
    x_max, y_max, z_max = np.max(pts, axis=0)
    center_x = 0.5 * (float(x_min) + float(x_max))
    center_y = 0.5 * (float(y_min) + float(y_max))
    radius_xy = max(0.5 * float(x_max - x_min) + 650.0, 0.5 * float(y_max - y_min) + 650.0, 2000.0)
    return {
        "x": (center_x - radius_xy, center_x + radius_xy),
        "y": (center_y - radius_xy, center_y + radius_xy),
        "z": (0.0, max(float(z_max) + 900.0, 3000.0)),
    }


def draw_3d_scene_frame_with_gt(
    frame_id: int,
    activity_name: str,
    tracks: Iterable[SimpleNamespace],
    gt_pose18: Optional[np.ndarray],
    gt_pose32: Optional[np.ndarray],
    output_dir: str | Path,
    visualization_mode: str,
    cfg_module: Any,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = list(tracks)
    show_custom = str(visualization_mode).lower() in {"custom18", "custom18_metasport", "all"}
    show_meta = str(visualization_mode).lower() in {"metasport32", "custom18_metasport", "all"}

    show_pred_custom = show_custom and bool(getattr(cfg_module, "comparison_show_pred_custom18", True))
    show_pred_meta = show_meta and bool(getattr(cfg_module, "comparison_show_pred_metasport32", True))
    show_gt_custom = bool(getattr(cfg_module, "comparison_show_gt_custom18", True))
    show_gt_meta = bool(getattr(cfg_module, "comparison_show_gt_metasport32", True))

    all_points: List[np.ndarray] = []
    for track in tracks:
        if show_pred_custom and getattr(track, "current_pose_3d_18", None) is not None:
            all_points.append(np.asarray(track.current_pose_3d_18, dtype=np.float32))
        if show_pred_meta and getattr(track, "current_pose_3d_metasport32", None) is not None:
            all_points.append(np.asarray(track.current_pose_3d_metasport32, dtype=np.float32))
    if gt_pose18 is not None and show_gt_custom:
        all_points.append(np.asarray(gt_pose18, dtype=np.float32))
    if gt_pose32 is not None and show_gt_meta:
        all_points.append(np.asarray(gt_pose32, dtype=np.float32))

    axis_limits = _axis_limits_from_points(all_points)
    if axis_limits is None:
        return

    fig = plt.figure(figsize=(13.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    if draw_ground_reference is not None:
        draw_ground_reference(ax, axis_limits)

    handles = []
    for track in tracks:
        pid = int(getattr(track, "global_id", -1))
        skeleton_color_bgr = pred_skeleton_color_bgr(cfg_module, pid)
        marker_color_bgr = pred_marker_color_bgr(cfg_module, pid)
        skeleton_color = bgr_to_rgb(skeleton_color_bgr)
        marker_color = bgr_to_rgb(marker_color_bgr)

        pose = getattr(track, "current_pose_3d_18", None)
        if show_pred_custom and pose is not None:
            pose = np.asarray(pose, dtype=np.float32)
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=26, color=skeleton_color, depthshade=True)
            for a, b in skeleton_18:
                ax.plot(
                    [pose[a, 0], pose[b, 0]],
                    [pose[a, 1], pose[b, 1]],
                    [pose[a, 2], pose[b, 2]],
                    linewidth=2.0,
                    color=skeleton_color,
                )
            handles.append(Line2D([0], [0], color=skeleton_color, marker="o", linestyle="-", label=f"ID {pid} pred skeleton", markersize=5))

        meta = getattr(track, "current_pose_3d_metasport32", None)
        if show_pred_meta and meta is not None:
            meta = np.asarray(meta, dtype=np.float32)
            finite = np.isfinite(meta).all(axis=1)
            if finite.any():
                ax.scatter(
                    meta[finite, 0], meta[finite, 1], meta[finite, 2],
                    s=int(getattr(cfg_module, "comparison_pred_marker_size", 16)),
                    color=marker_color,
                    marker="o",
                    alpha=0.85,
                    depthshade=True,
                )
                handles.append(Line2D([0], [0], color=marker_color, marker="o", linestyle="None", label=f"ID {pid} pred markers32", markersize=5))

    if gt_pose18 is not None and show_gt_custom:
        gt = np.asarray(gt_pose18, dtype=np.float32)
        gt_rgb = bgr_to_rgb(gt_skeleton_color_bgr(cfg_module))
        ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], s=34, color=gt_rgb, marker="o", depthshade=True)
        for a, b in skeleton_18:
            ax.plot(
                [gt[a, 0], gt[b, 0]],
                [gt[a, 1], gt[b, 1]],
                [gt[a, 2], gt[b, 2]],
                linewidth=2.4,
                color=gt_rgb,
                linestyle=str(getattr(cfg_module, "comparison_gt_skeleton_linestyle", "--")),
            )
        handles.append(Line2D([0], [0], color=gt_rgb, marker="o", linestyle="--", label=str(getattr(cfg_module, "comparison_gt_skeleton_label", "GT skeleton")), markersize=5))

    if gt_pose32 is not None and show_gt_meta:
        gt32 = np.asarray(gt_pose32, dtype=np.float32)
        finite = np.isfinite(gt32).all(axis=1)
        if finite.any():
            gt_marker_rgb = bgr_to_rgb(gt_marker_color_bgr(cfg_module))
            ax.scatter(
                gt32[finite, 0], gt32[finite, 1], gt32[finite, 2],
                s=int(getattr(cfg_module, "comparison_gt_marker_size", 18)),
                color=gt_marker_rgb,
                marker="o",
                alpha=0.95,
                depthshade=True,
            )
            handles.append(Line2D([0], [0], color=gt_marker_rgb, marker="o", linestyle="None", label=str(getattr(cfg_module, "comparison_gt_marker_label", "GT markers32")), markersize=5))

    ax.set_xlim(*axis_limits["x"])
    ax.set_ylim(*axis_limits["y"])
    ax.set_zlim(*axis_limits["z"])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"{activity_name} | frame {frame_id} | Butterworth + GT")
    ax.set_box_aspect((
        axis_limits["x"][1] - axis_limits["x"][0],
        axis_limits["y"][1] - axis_limits["y"][0],
        axis_limits["z"][1] - axis_limits["z"][0],
    ))
    ax.view_init(elev=10, azim=18)
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.82, bottom=0.02, top=0.92)

    output_path = output_dir / activity_name / f"frame_{frame_id:06d}_3d.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def build_tracks_for_frame(
    pose_df: pd.DataFrame,
    metasport_rows: Dict[tuple[int, int], pd.Series],
    bboxes: Dict[tuple[int, int], Dict[str, tuple[int, int, int, int]]],
    frame_id: int,
) -> list[SimpleNamespace]:

    frame_id = int(frame_id)
    frame_rows = pose_df[pose_df["frame_id"] == frame_id]
    tracks: list[SimpleNamespace] = []
    ids_with_pose: set[int] = set()

    for _, row in frame_rows.iterrows():
        person_id = int(row["person_id"])
        pose18 = pose_matrix_from_row(row, joint_names_18)
        if pose18 is None:
            continue

        ids_with_pose.add(person_id)

        views_raw = row.get("views_used", "")
        if isinstance(views_raw, float) and np.isnan(views_raw):
            views = []
        else:
            views = [v for v in str(views_raw).split("|") if v]

        try:
            raw_score = row.get("score", 1.0)
            score = float(raw_score) if np.isfinite(float(raw_score)) else 1.0
        except Exception:
            score = 1.0

        current_bbox = bboxes.get((frame_id, person_id), {})

        metasport_pose = None
        meta_row = metasport_rows.get((frame_id, person_id))
        if meta_row is not None:
            metasport_pose = pose_matrix_from_row(meta_row, metasport32_names)

        tracks.append(SimpleNamespace(
            global_id=person_id,
            state=TrackState.confirmed_3d,
            time_since_update=0,
            is_occluded_by=None,
            current_bbox=current_bbox,
            current_pose_3d_18=pose18,
            current_pose_3d_19=None,
            current_pose_3d_custom19=None,
            current_pose_3d_metasport32=metasport_pose,
            current_unreliable_pose_3d_18=None,
            current_unreliable_pose_3d_metasport32=None,
            learned_views_used=views if views else list(current_bbox.keys()),
            learned_pose_score=score,
            unreliable_views_used=[],
            unreliable_pose_score=0.0,
            unreliable_pose_reason="",
            has_filtered_3d_pose=True,
        ))

    bbox_person_ids = sorted(
        int(person_id)
        for (bbox_frame_id, person_id) in bboxes.keys()
        if int(bbox_frame_id) == frame_id
    )
    for person_id in bbox_person_ids:
        if person_id in ids_with_pose:
            continue

        current_bbox = bboxes.get((frame_id, person_id), {})
        if not current_bbox:
            continue

        tracks.append(SimpleNamespace(
            global_id=person_id,
            state=TrackState.confirmed_2d,
            time_since_update=0,
            is_occluded_by=None,
            current_bbox=current_bbox,
            current_pose_3d_18=None,
            current_pose_3d_19=None,
            current_pose_3d_custom19=None,
            current_pose_3d_metasport32=None,
            current_unreliable_pose_3d_18=None,
            current_unreliable_pose_3d_metasport32=None,
            learned_views_used=list(current_bbox.keys()),
            learned_pose_score=0.0,
            unreliable_views_used=[],
            unreliable_pose_score=0.0,
            unreliable_pose_reason="no_filtered_3d_pose_bbox_only",
            has_filtered_3d_pose=False,
        ))

    tracks.sort(key=lambda t: int(t.global_id))
    return tracks

def draw_projected_skeleton_no_bbox(image: np.ndarray, points_2d: np.ndarray, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    for joint_a, joint_b in skeleton_18:
        pa = points_2d[joint_a]
        pb = points_2d[joint_b]
        if not np.isfinite(pa).all() or not np.isfinite(pb).all():
            continue
        xa, ya = int(round(float(pa[0]))), int(round(float(pa[1])))
        xb, yb = int(round(float(pb[0]))), int(round(float(pb[1])))
        if 0 <= xa < width and 0 <= ya < height and 0 <= xb < width and 0 <= yb < height:
            cv2.line(image, (xa, ya), (xb, yb), color, 2)
    for point in points_2d:
        if not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), 4, color, -1)


def draw_filtered_video_frame(
    frame_id: int,
    camera_name: str,
    frame: np.ndarray,
    camera: Any,
    tracks: Iterable[SimpleNamespace],
    visualization_mode: str,
    gt_pose18: Optional[np.ndarray] = None,
    gt_pose32: Optional[np.ndarray] = None,
    cfg_module: Any = None,
) -> np.ndarray:
    image = frame.copy()
    cv2.putText(
        image,
        f"Frame {frame_id} Butterworth",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    show_custom = str(visualization_mode).lower() in {"custom18", "custom18_metasport", "all"}
    show_meta = str(visualization_mode).lower() in {"metasport32", "custom18_metasport", "all"}
    show_pred_custom = show_custom and bool(getattr(cfg_module, "comparison_show_pred_custom18", True))
    show_pred_meta = show_meta and bool(getattr(cfg_module, "comparison_show_pred_metasport32", True))

    for track in tracks:
        skeleton_color = pred_skeleton_color_bgr(cfg_module, int(track.global_id))
        marker_color = pred_marker_color_bgr(cfg_module, int(track.global_id))

        bbox = track.current_bbox.get(camera_name) if getattr(track, "current_bbox", None) else None
        if bbox is not None:
            draw_bbox(image=image, bbox=bbox, color=skeleton_color, label=f"ID {track.global_id} [BW]")

        if show_pred_custom and track.current_pose_3d_18 is not None:
            try:
                projected = project_points(track.current_pose_3d_18, camera.P)
                draw_projected_skeleton_no_bbox(image, projected, skeleton_color)
                if bbox is None:
                    finite = projected[np.isfinite(projected).all(axis=1)]
                    if len(finite) > 0:
                        x, y = finite[0]
                        cv2.putText(
                            image,
                            f"ID {track.global_id} [BW]",
                            (int(round(x)), int(round(y))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            skeleton_color,
                            2,
                            cv2.LINE_AA,
                        )
            except Exception:
                pass

        if show_pred_meta and track.current_pose_3d_metasport32 is not None:
            try:
                projected_meta = project_points(track.current_pose_3d_metasport32, camera.P)
                draw_projected_points(image, projected_meta, marker_color, radius=3)
            except Exception:
                pass

    if cfg_module is not None and bool(getattr(cfg_module, "gt_render_on_filtered_videos", True)):
        draw_gt_video_overlay(
            image=image,
            gt_pose18=gt_pose18,
            gt_pose32=gt_pose32,
            camera=camera,
            cfg_module=cfg_module,
        )

    return image


def read_filtered_data(filtered_csv_dir: Path, activity_name: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    pose_csv = filtered_csv_dir / f"{activity_name}_keypoints3d_multiperson_butterworth.csv"
    metasport_csv = filtered_csv_dir / f"{activity_name}_metasport32_multiperson_butterworth.csv"
    pose_df = pd.read_csv(pose_csv) if pose_csv.exists() else pd.DataFrame()
    metasport_df = pd.read_csv(metasport_csv) if metasport_csv.exists() else None
    return pose_df, metasport_df


def render_filtered_outputs_for_activity(
    cfg_module: Any,
    activity: Any,
    filtered_csv_dir: Path,
    filtered_video_dir: Path,
    filtered_vis3d_dir: Path,
) -> None:
    pose_df, metasport_df = read_filtered_data(filtered_csv_dir, activity.name)
    if pose_df.empty:
        print(f"[rerender] no filtered pose CSV for {activity.name}, skipped")
        return

    metasport_rows = rows_by_frame_person(metasport_df)
    bboxes = load_tracking_debug_bboxes(Path(cfg_module.csv_dir), activity.name)

    cameras = load_cameras_from_xcp(
        xcp_path=activity.xcp_path,
        device_id_by_camera=cfg_module.device_id_by_camera,
        variant=cfg_module.xcp_variant,
        t_scale=cfg_module.xcp_t_scale,
    )

    gt_provider = build_gt_provider_if_needed(cfg_module, activity.name)

    frame_ids_to_render = set(int(x) for x in pose_df["frame_id"].unique())
    render_videos = bool(getattr(
        cfg_module,
        "butterworth_render_filtered_videos",
        getattr(cfg_module, "save_output_videos", True),
    ))
    render_vis3d = bool(getattr(
        cfg_module,
        "butterworth_render_filtered_vis3d",
        getattr(cfg_module, "save_3d_images", True),
    ))

    with MultiCameraVideos(
        video_paths_by_camera=activity.video_paths_by_camera,
        camera_names=cfg_module.camera_names,
    ) as videos:
        frame_size_by_camera = {
            camera_name: (
                videos.infos[camera_name].width,
                videos.infos[camera_name].height,
            )
            for camera_name in cfg_module.camera_names
        }

        writers = {}
        try:
            if render_videos:
                for camera_name in cfg_module.camera_names:
                    frame_size = frame_size_by_camera[camera_name]
                    if getattr(cfg_module, "video_output_size", None) is not None:
                        frame_size = cfg_module.video_output_size
                    output_path = filtered_video_dir / f"{activity.name}_{camera_name}_tracked_butterworth.mp4"
                    writer, actual_path = make_video_writer(
                        output_path=output_path,
                        fps=videos.fps,
                        frame_size=frame_size,
                        codec=getattr(cfg_module, "video_output_codec", "mp4v"),
                    )
                    writers[camera_name] = writer
                    print(f"[rerender] video writer {camera_name}: {actual_path}")

            start_frame = int(getattr(cfg_module, "start_frame", 1))
            end_frame = videos.min_frame_count if cfg_module.end_frame is None else min(int(cfg_module.end_frame), videos.min_frame_count)
            stride = int(getattr(cfg_module, "frame_stride", 1))

            for frame_id, frames in videos.iter_frames(start_frame=start_frame, end_frame=end_frame, stride=stride):
                has_filtered_pose = frame_id in frame_ids_to_render
                has_debug_bboxes = any(int(bbox_frame_id) == int(frame_id) for bbox_frame_id, _ in bboxes.keys())
                tracks = []

                if has_filtered_pose or has_debug_bboxes:
                    tracks = build_tracks_for_frame(
                        pose_df=pose_df,
                        metasport_rows=metasport_rows,
                        bboxes=bboxes,
                        frame_id=frame_id,
                    )

                if gt_provider is not None:
                    gt_pose18, gt_pose32 = gt_provider.poses_for_frame(frame_id)
                else:
                    gt_pose18, gt_pose32 = None, None

                if render_vis3d and has_filtered_pose:
                    every = int(getattr(cfg_module, "save_3d_every", 1))
                    if every > 0 and frame_id % every == 0:
                        if bool(getattr(cfg_module, "gt_render_on_filtered_vis3d", True)):
                            draw_3d_scene_frame_with_gt(
                                frame_id=frame_id,
                                activity_name=activity.name,
                                tracks=tracks,
                                gt_pose18=gt_pose18,
                                gt_pose32=gt_pose32,
                                output_dir=filtered_vis3d_dir,
                                visualization_mode=getattr(cfg_module, "visualization_mode", "custom18_metasport"),
                                cfg_module=cfg_module,
                            )
                        else:
                            draw_3d_scene_frame(
                                frame_id=frame_id,
                                activity_name=activity.name,
                                tracks=tracks,
                                output_dir=filtered_vis3d_dir,
                                visualization_mode=getattr(cfg_module, "visualization_mode", "custom18_metasport"),
                                show_joint_numbers=getattr(cfg_module, "show_vis3d_joint_numbers", False),
                                show_custom18_joint_numbers=getattr(cfg_module, "show_vis3d_custom18_joint_numbers", False),
                                show_metasport_marker_numbers=getattr(cfg_module, "show_vis3d_metasport_marker_numbers", False),
                                show_joint_legend=getattr(cfg_module, "show_vis3d_joint_legend", False),
                                show_person_legend=getattr(cfg_module, "show_vis3d_person_legend", True),
                                show_person_labels_on_scene=getattr(cfg_module, "show_vis3d_person_labels_on_scene", False),
                                joint_label_mode=getattr(cfg_module, "vis3d_joint_label_mode", "index"),
                            )

                for camera_name, writer in writers.items():
                    if camera_name not in frames or camera_name not in cameras:
                        continue
                    image = draw_filtered_video_frame(
                        frame_id=frame_id,
                        camera_name=camera_name,
                        frame=frames[camera_name],
                        camera=cameras[camera_name],
                        tracks=tracks,
                        visualization_mode=getattr(cfg_module, "visualization_mode", "custom18_metasport"),
                        gt_pose18=gt_pose18,
                        gt_pose32=gt_pose32,
                        cfg_module=cfg_module,
                    )
                    if getattr(cfg_module, "video_output_size", None) is not None:
                        image = cv2.resize(image, cfg_module.video_output_size, interpolation=cv2.INTER_AREA)
                    writer.write(image)

                print(f"[rerender] {activity.name} frame={frame_id} tracks={len(tracks)}")
        finally:
            for writer in writers.values():
                writer.release()


def main() -> None:
    base_bw = butterworth_config_from_local(cfg)

    output_root = Path(cfg.output_root)
    raw_csv_dir = Path(cfg.csv_dir)
    filtered_csv_dir = Path(getattr(cfg, "butterworth_csv_dir", output_root / "csv_butterworth"))
    filtered_video_dir = Path(getattr(cfg, "butterworth_video_dir", output_root / "videos_butterworth"))
    filtered_vis3d_dir = Path(getattr(cfg, "butterworth_vis3d_dir", output_root / "vis3d_butterworth"))

    activities = discover_activities(
        base_dir=cfg.base_videos_dir,
        camera_names=cfg.camera_names,
        device_id_by_camera=cfg.device_id_by_camera,
        video_extensions=cfg.video_extensions,
        activity_mode=cfg.activity_mode,
        selected_activity_name=cfg.selected_activity_name,
    )

    if not activities:
        raise SystemExit("No activities found. Check local.py paths.")

    print("[butterworth] base config:", base_bw)
    print(f"[butterworth] raw_csv_dir={raw_csv_dir}")
    print(f"[butterworth] filtered_csv_dir={filtered_csv_dir}")
    print(f"[rerender] filtered_video_dir={filtered_video_dir}")
    print(f"[rerender] filtered_vis3d_dir={filtered_vis3d_dir}")

    for activity in activities:
        bw = butterworth_config_for_activity(cfg, activity.name)
        print(f"\n=== {activity.name} | cutoff={bw.cutoff_hz:.2f} Hz ===")
        with MultiCameraVideos(
            video_paths_by_camera=activity.video_paths_by_camera,
            camera_names=cfg.camera_names,
        ) as videos:
            fps = float(videos.fps)

        raw_pose_csv = raw_csv_dir / f"{activity.name}_keypoints3d_multiperson.csv"
        filtered_pose_csv = filtered_csv_dir / f"{activity.name}_keypoints3d_multiperson_butterworth.csv"
        filter_pose_csv(raw_pose_csv, filtered_pose_csv, fps=fps, bw=bw)

        raw_meta_csv = raw_csv_dir / f"{activity.name}_metasport32_multiperson.csv"
        filtered_meta_csv = filtered_csv_dir / f"{activity.name}_metasport32_multiperson_butterworth.csv"
        if raw_meta_csv.exists():
            filter_pose_csv(raw_meta_csv, filtered_meta_csv, fps=fps, bw=bw)

        if bool(getattr(cfg, "butterworth_render_filtered_outputs", True)):
            render_filtered_outputs_for_activity(
                cfg_module=cfg,
                activity=activity,
                filtered_csv_dir=filtered_csv_dir,
                filtered_video_dir=filtered_video_dir,
                filtered_vis3d_dir=filtered_vis3d_dir,
            )

    print("\nDone. Existing raw outputs were not modified.")


if __name__ == "__main__":
    main()
