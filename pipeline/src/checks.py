from __future__ import annotations

from pathlib import Path
from typing import Any, List


required_config_names = [
    "subject_name",
    "base_videos_dir",
    "output_root",
    "activity_mode",
    "selected_activity_name",
    "video_extensions",
    "camera_names",
    "device_id_by_camera",

    "yolo_model_path",
    "reid_model_path",
    "alg_weights_path",
    "vol_weights_path",

    "yolo_confidence",
    "yolo_person_class",
    "yolo_tracker_config",
    "rtmlib_device",
    "rtmlib_backend",
    "pose_min_joint_confidence",
    "pose_min_valid_joints",
    "bbox_pose_association_max_distance",
    "reid_use_torso_crop",
    "reid_bbox_margin_ratio",

    "track_patience",
    "min_hits_to_confirm",
    "intra_frame_appearance_threshold",
    "association_threshold",
    "min_confidence_for_new_track",
    "min_views_for_new_track",
    "feature_gallery_size",
    "feature_update_weight",
    "tracklet_conflict_max_age",
    "motion_weight",
    "appearance_weight",

    "image_size",
    "min_views_for_learned_pose",
    "n_model_joints",
    "n_output_joints",
    "xcp_variant",
    "xcp_t_scale",
    "cuboid_side",
    "learned_pose_device",
    "learned_pose_bbox_scale",
    "learned_pose_square_crop",

    "start_frame",
    "end_frame",
    "frame_stride",
    "max_frames_per_activity",

    "save_csv_3d",
    "save_reprojection_images",
    "save_3d_images",
    "save_failure_log",
    "save_reprojection_every",
    "save_3d_every",

    "csv_dir",
    "video_output_dir",
    "reprojection_dir",
    "vis3d_dir",
    "debug_dir",
]


def check_required_config_names(cfg: Any) -> List[str]:
    errors = []

    for name in required_config_names:
        if not hasattr(cfg, name):
            errors.append(f"Variable manquante dans configs/local.py: {name}")

    return errors


def check_config_values(cfg: Any) -> List[str]:
    errors = []

    if cfg.activity_mode not in {"all", "single"}:
        errors.append(
            f"activity_mode doit être 'all' ou 'single', reçu: {cfg.activity_mode}"
        )

    if cfg.activity_mode == "single":
        if not isinstance(cfg.selected_activity_name, str) or not cfg.selected_activity_name.strip():
            errors.append(
                "selected_activity_name doit être renseigné quand activity_mode='single'"
            )

    if not isinstance(cfg.camera_names, list) or not cfg.camera_names:
        errors.append("camera_names doit être une liste non vide")

    for camera_name in cfg.camera_names:
        if camera_name not in cfg.device_id_by_camera:
            errors.append(f"device_id_by_camera ne contient pas {camera_name}")

    if cfg.min_views_for_learned_pose < 2:
        errors.append("min_views_for_learned_pose doit être au moins 2")

    if cfg.frame_stride <= 0:
        errors.append("frame_stride doit être > 0")

    if cfg.start_frame < 1:
        errors.append("start_frame doit être >= 1")

    if cfg.end_frame is not None and cfg.end_frame < cfg.start_frame:
        errors.append("end_frame doit être None ou >= start_frame")

    if cfg.save_reprojection_every <= 0 and cfg.save_reprojection_images:
        errors.append("save_reprojection_every doit être > 0 si save_reprojection_images=True")

    if cfg.save_3d_every <= 0 and cfg.save_3d_images:
        errors.append("save_3d_every doit être > 0 si save_3d_images=True")

    return errors


def check_paths(cfg: Any) -> List[str]:
    errors = []

    paths_to_check = [
        ("base_videos_dir", cfg.base_videos_dir),
        ("yolo_model_path", cfg.yolo_model_path),
        ("reid_model_path", cfg.reid_model_path),
        ("alg_weights_path", cfg.alg_weights_path),
        ("vol_weights_path", cfg.vol_weights_path),
    ]

    for name, path in paths_to_check:
        path = Path(path)

        if not path.exists():
            errors.append(f"Chemin introuvable pour {name}: {path}")

    output_dirs = [
        ("csv_dir", cfg.csv_dir),
        ("video_output_dir", cfg.video_output_dir),
        ("reprojection_dir", cfg.reprojection_dir),
        ("vis3d_dir", cfg.vis3d_dir),
        ("debug_dir", cfg.debug_dir),
    ]

    for name, directory in output_dirs:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if not directory.exists():
            errors.append(f"Impossible de créer le dossier {name}: {directory}")

    return errors


def check_imports() -> List[str]:
    errors = []

    modules = [
        "src.videos",
        "src.cameras",
        "src.detection",
        "src.tracking",
        "src.geometry",
        "src.joints",
        "src.learned_pose",
        "src.outputs",
        "src.pipeline",
    ]

    for module_name in modules:
        try:
            __import__(module_name)
        except Exception as exc:
            errors.append(f"Import échoué pour {module_name}: {exc}")

    return errors


def run_project_checks(cfg: Any) -> bool:
    print("")
    print("Vérification du projet")

    all_errors = []

    checks = [
        ("variables de config", check_required_config_names),
        ("valeurs de config", check_config_values),
        ("chemins", check_paths),
    ]

    for title, check_fn in checks:
        print(f"- {title}")
        errors = check_fn(cfg)

        if errors:
            all_errors.extend(errors)
            print("  erreurs")
            for error in errors:
                print(f"  - {error}")
        else:
            print("  ok")

    print("- imports")
    import_errors = check_imports()

    if import_errors:
        all_errors.extend(import_errors)
        print("  erreurs")
        for error in import_errors:
            print(f"  - {error}")
    else:
        print("  ok")

    print("")

    if all_errors:
        print("Vérification terminée avec erreurs.")
        print(f"Nombre d'erreurs: {len(all_errors)}")
        return False

    print("Vérification terminée sans erreur.")
    return True