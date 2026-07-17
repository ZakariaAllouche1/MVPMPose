r"""Run MVPMPose activity by activity, then apply Butterworth + rerender after each activity.

    python .\pipeline\scripts\run_pipeline_full_filtered.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import local as cfg 
from src.pipeline import ( 
    build_learned_pose_estimator,
    print_config_summary,
    run_activity,
)
from src.videos import MultiCameraVideos, discover_activities 


def load_filter_module() -> Any:
    filter_script = SCRIPTS_DIR / "filter_butterworth_and_rerender.py"
    if not filter_script.exists():
        raise FileNotFoundError(f"Missing script: {filter_script}")

    spec = importlib.util.spec_from_file_location(
        "filter_butterworth_and_rerender",
        str(filter_script),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {filter_script}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["filter_butterworth_and_rerender"] = module
    spec.loader.exec_module(module)
    return module


def selected_activities() -> list[Any]:
    activities = discover_activities(
        base_dir=cfg.base_videos_dir,
        camera_names=cfg.camera_names,
        device_id_by_camera=cfg.device_id_by_camera,
        video_extensions=cfg.video_extensions,
        activity_mode=cfg.activity_mode,
        selected_activity_name=cfg.selected_activity_name,
    )
    if not activities:
        raise SystemExit("No activities found. Check local.py paths and activity settings.")
    return activities


def filter_and_rerender_one_activity(filter_module: Any, activity: Any) -> None:
    output_root = Path(cfg.output_root)
    raw_csv_dir = Path(cfg.csv_dir)
    filtered_csv_dir = Path(getattr(cfg, "butterworth_csv_dir", output_root / "csv_butterworth"))
    filtered_video_dir = Path(getattr(cfg, "butterworth_video_dir", output_root / "videos_butterworth"))
    filtered_vis3d_dir = Path(getattr(cfg, "butterworth_vis3d_dir", output_root / "vis3d_butterworth"))

    filtered_csv_dir.mkdir(parents=True, exist_ok=True)
    filtered_video_dir.mkdir(parents=True, exist_ok=True)
    filtered_vis3d_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(filter_module, "butterworth_config_for_activity"):
        bw = filter_module.butterworth_config_for_activity(cfg, activity.name)
    else:
        bw = filter_module.butterworth_config_from_local(cfg)

    print(f"\n===== [Butterworth] {activity.name} | cutoff={float(bw.cutoff_hz):.2f} Hz =====")

    with MultiCameraVideos(
        video_paths_by_camera=activity.video_paths_by_camera,
        camera_names=cfg.camera_names,
    ) as videos:
        fps = float(videos.fps)

    raw_pose_csv = raw_csv_dir / f"{activity.name}_keypoints3d_multiperson.csv"
    filtered_pose_csv = filtered_csv_dir / f"{activity.name}_keypoints3d_multiperson_butterworth.csv"
    filter_module.filter_pose_csv(raw_pose_csv, filtered_pose_csv, fps=fps, bw=bw)

    raw_meta_csv = raw_csv_dir / f"{activity.name}_metasport32_multiperson.csv"
    filtered_meta_csv = filtered_csv_dir / f"{activity.name}_metasport32_multiperson_butterworth.csv"
    if raw_meta_csv.exists():
        filter_module.filter_pose_csv(raw_meta_csv, filtered_meta_csv, fps=fps, bw=bw)

    if bool(getattr(cfg, "butterworth_render_filtered_outputs", True)):
        old_save_output_videos = getattr(cfg, "save_output_videos", True)
        old_save_3d_images = getattr(cfg, "save_3d_images", True)
        try:
            if bool(getattr(cfg, "butterworth_render_filtered_videos", True)):
                cfg.save_output_videos = True
            if bool(getattr(cfg, "butterworth_render_filtered_vis3d", True)):
                cfg.save_3d_images = True

            filter_module.render_filtered_outputs_for_activity(
                cfg_module=cfg,
                activity=activity,
                filtered_csv_dir=filtered_csv_dir,
                filtered_video_dir=filtered_video_dir,
                filtered_vis3d_dir=filtered_vis3d_dir,
            )
        finally:
            cfg.save_output_videos = old_save_output_videos
            cfg.save_3d_images = old_save_3d_images


def main() -> None:
    print("Démarrage de la pipeline complète avec Butterworth par activité")
    print_config_summary(cfg)

    activities = selected_activities()
    print(f"\nActivités à traiter: {len(activities)}")
    for activity in activities:
        print(f"  - {activity.name}")

    filter_module = load_filter_module()
    learned_pose = build_learned_pose_estimator(cfg)

    for index, activity in enumerate(activities, start=1):
        print(f"\n############################################################")
        print(f"# Activité {index}/{len(activities)}: {activity.name}")
        print(f"############################################################")

        print(f"\n===== [1/2] Pipeline brute: {activity.name} =====")
        run_activity(
            cfg=cfg,
            activity=activity,
            learned_pose=learned_pose,
        )

        print(f"\n===== [2/2] Butterworth + rerender filtré: {activity.name} =====")
        filter_and_rerender_one_activity(filter_module, activity)

    print("\nDone. All selected activities have raw CSV + filtered CSV/videos/vis3d.")


if __name__ == "__main__":
    main()
