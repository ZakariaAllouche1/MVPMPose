from __future__ import annotations

from typing import Any

import torch

from src.cameras import load_cameras_from_xcp, print_camera_summary
from src.detection import MultiCameraDetector, print_detection_summary
from src.learned_pose import LearnedPoseEstimator
from src.outputs import (
    ActivityOutputs,
    ActivityVideoWriters,
    draw_3d_scene_frame,
    draw_reprojection_frame,
)
from src.tracking import MultiViewTracker
from src.videos import MultiCameraVideos, discover_activities


def print_config_summary(cfg: Any) -> None:
    print("")
    print("Configuration")
    print(f"subject_name: {cfg.subject_name}")
    print(f"activity_mode: {cfg.activity_mode}")
    print(f"selected_activity_name: {cfg.selected_activity_name}")
    print(f"base_videos_dir: {cfg.base_videos_dir}")
    print(f"video_extensions: {cfg.video_extensions}")
    print(f"start_frame: {cfg.start_frame}")
    print(f"end_frame: {cfg.end_frame}")
    print(f"frame_stride: {cfg.frame_stride}")
    print(f"max_frames_per_activity: {cfg.max_frames_per_activity}")
    print("")


def choose_learned_pose_device(cfg: Any) -> str:
    requested_device = cfg.learned_pose_device

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[learned pose] CUDA indisponible, passage au CPU.")
        return "cpu"

    return requested_device


def build_detector(cfg: Any) -> MultiCameraDetector:
    return MultiCameraDetector(
        camera_names=cfg.camera_names,
        yolo_model_path=cfg.yolo_model_path,
        reid_model_path=cfg.reid_model_path,
        rtmlib_device=cfg.rtmlib_device,
        rtmlib_backend=cfg.rtmlib_backend,
        yolo_confidence=cfg.yolo_confidence,
        yolo_person_class=cfg.yolo_person_class,
        yolo_tracker_config=cfg.yolo_tracker_config,
        pose_min_joint_confidence=cfg.pose_min_joint_confidence,
        pose_min_valid_joints=cfg.pose_min_valid_joints,
        bbox_pose_association_max_distance=cfg.bbox_pose_association_max_distance,
        reid_use_torso_crop=cfg.reid_use_torso_crop,
        reid_bbox_margin_ratio=cfg.reid_bbox_margin_ratio,
    )


def build_tracker(cfg: Any) -> MultiViewTracker:
    return MultiViewTracker(
        camera_names=cfg.camera_names,
        track_patience=cfg.track_patience,
        min_hits_to_confirm=cfg.min_hits_to_confirm,
        intra_frame_appearance_threshold=cfg.intra_frame_appearance_threshold,
        association_threshold=cfg.association_threshold,
        min_confidence_for_new_track=cfg.min_confidence_for_new_track,
        min_views_for_new_track=cfg.min_views_for_new_track,
        feature_gallery_size=cfg.feature_gallery_size,
        feature_update_weight=cfg.feature_update_weight,
        tracklet_conflict_max_age=cfg.tracklet_conflict_max_age,
        motion_weight=cfg.motion_weight,
        appearance_weight=cfg.appearance_weight,
        spatial_weight=getattr(cfg, "spatial_weight", 0.35),
        reprojection_error_threshold=getattr(cfg, "reprojection_error_threshold", 120.0),
        spatial_consistency_threshold_mm=getattr(cfg, "spatial_consistency_threshold_mm", 1500.0),
        duplicate_merge_distance_mm=getattr(cfg, "duplicate_merge_distance_mm", 600.0),
        duplicate_merge_appearance=getattr(cfg, "duplicate_merge_appearance", 0.30),
        group_safety_merge_distance_mm=getattr(cfg, "group_safety_merge_distance_mm", 400.0),
        occlusion_xy_distance_mm=getattr(cfg, "occlusion_xy_distance_mm", 900.0),
        max_occlusion_frames=getattr(cfg, "max_occlusion_frames", 120),
        reactivation_appearance_threshold=getattr(cfg, "reactivation_appearance_threshold", 0.35),
        occlusion_reid_threshold_mm=getattr(cfg, "occlusion_reid_threshold_mm", 1200.0),
        min_visibility_for_new_track=getattr(cfg, "min_visibility_for_new_track", 0.40),
        min_detection_confidence_for_matching=getattr(cfg, "min_detection_confidence_for_matching", 0.10),
        min_detection_confidence_for_grouping=getattr(cfg, "min_detection_confidence_for_grouping", 0.20),
        min_detection_confidence_for_new_track_single_view=getattr(
            cfg,
            "min_detection_confidence_for_new_track_single_view",
            0.70,
        ),
        min_detection_confidence_for_new_track_multiview=getattr(
            cfg,
            "min_detection_confidence_for_new_track_multiview",
            0.55,
        ),
        min_bbox_height_for_new_track=getattr(cfg, "min_bbox_height_for_new_track", 80),
        min_bbox_width_for_new_track=getattr(cfg, "min_bbox_width_for_new_track", 30),
        allow_featureless_detection_for_matching=getattr(
            cfg,
            "allow_featureless_detection_for_matching",
            True,
        ),
        allow_featureless_detection_for_new_track=getattr(
            cfg,
            "allow_featureless_detection_for_new_track",
            False,
        ),
        featureless_pair_cost=getattr(cfg, "featureless_pair_cost", 0.85),
        strong_spatial_score_for_featureless_pair=getattr(
            cfg,
            "strong_spatial_score_for_featureless_pair",
            0.65,
        ),
    )


def build_learned_pose_estimator(cfg: Any) -> LearnedPoseEstimator:
    device = choose_learned_pose_device(cfg)

    return LearnedPoseEstimator(
        alg_weights_path=cfg.alg_weights_path,
        vol_weights_path=cfg.vol_weights_path,
        device=device,
        image_size=cfg.image_size,
        n_model_joints=cfg.n_model_joints,
        cuboid_side=cfg.cuboid_side,
    )


def get_frame_range(cfg: Any, videos: MultiCameraVideos) -> tuple[int, int, int]:
    start_frame = int(cfg.start_frame)

    if cfg.end_frame is None:
        end_frame = videos.min_frame_count
    else:
        end_frame = min(int(cfg.end_frame), videos.min_frame_count)

    stride = int(cfg.frame_stride)

    if start_frame < 1:
        raise ValueError(f"start_frame doit être >= 1, reçu: {start_frame}")

    if end_frame < start_frame:
        raise ValueError(
            f"end_frame doit être >= start_frame, reçu start={start_frame}, end={end_frame}"
        )

    if stride <= 0:
        raise ValueError(f"frame_stride doit être > 0, reçu: {stride}")

    return start_frame, end_frame, stride


def should_stop_after_count(cfg: Any, processed_count: int) -> bool:
    if cfg.max_frames_per_activity is None:
        return False

    return processed_count >= int(cfg.max_frames_per_activity)


def should_save_frame(frame_id: int, every: int) -> bool:
    if every <= 0:
        return False

    return frame_id % every == 0


def run_learned_pose_for_tracks(
    cfg: Any,
    frame_id: int,
    frames_by_camera,
    cameras_by_camera,
    tracker: MultiViewTracker,
    learned_pose: LearnedPoseEstimator,
    outputs: ActivityOutputs,
) -> int:
    num_success = 0

    for track in tracker.active_tracks():
        if track.time_since_update != 0:
            continue

        views = sorted(track.current_bbox.keys())

        if len(track.current_bbox) < cfg.min_views_for_learned_pose:
            outputs.write_failure(
                frame_id=frame_id,
                person_id=track.global_id,
                reason="not_enough_views",
                views=views,
            )
            continue

        result = learned_pose.infer_track(
            person_id=track.global_id,
            frames_by_camera=frames_by_camera,
            boxes_by_camera=track.current_bbox,
            cameras_by_camera=cameras_by_camera,
            min_views=cfg.min_views_for_learned_pose,
            bbox_scale=cfg.learned_pose_bbox_scale,
            square_crop=cfg.learned_pose_square_crop,
        )

        if result is None:
            outputs.write_failure(
                frame_id=frame_id,
                person_id=track.global_id,
                reason="learned_pose_failed",
                views=views,
            )
            continue

        track.update_3d_pose(
            pose_3d_17=result.keypoints17,
            pose_3d_15=result.keypoints15,
            root_position=result.root_position,
            score=result.score,
            views_used=result.views_used,
        )

        num_success += 1

        print(
            f"[learned pose] frame={frame_id} "
            f"id={track.global_id} "
            f"views={result.views_used} "
            f"score={result.score:.3f}"
        )

    return num_success


def run_activity(
    cfg: Any,
    activity,
    learned_pose: LearnedPoseEstimator,
) -> None:
    print("")
    print(f"Activité: {activity.name}")
    print(f"xcp: {activity.xcp_path}")

    for camera_name, video_path in activity.video_paths_by_camera.items():
        print(f"{camera_name}: {video_path}")

    cameras = load_cameras_from_xcp(
        xcp_path=activity.xcp_path,
        device_id_by_camera=cfg.device_id_by_camera,
        variant=cfg.xcp_variant,
        t_scale=cfg.xcp_t_scale,
    )

    print_camera_summary(cameras)

    detector = build_detector(cfg)
    tracker = build_tracker(cfg)

    outputs = ActivityOutputs(
        activity_name=activity.name,
        csv_dir=cfg.csv_dir,
        debug_dir=cfg.debug_dir,
        save_csv=cfg.save_csv_3d,
        save_failures=cfg.save_failure_log,
    )

    video_writers = None

    try:
        with MultiCameraVideos(
            video_paths_by_camera=activity.video_paths_by_camera,
            camera_names=cfg.camera_names,
        ) as videos:
            videos.print_summary()

            if cfg.save_output_videos:
                frame_size_by_camera = {
                    camera_name: (
                        videos.infos[camera_name].width,
                        videos.infos[camera_name].height,
                    )
                    for camera_name in cfg.camera_names
                }

                video_writers = ActivityVideoWriters(
                    activity_name=activity.name,
                    camera_names=cfg.camera_names,
                    video_output_dir=cfg.video_output_dir,
                    fps=videos.fps,
                    frame_size_by_camera=frame_size_by_camera,
                    output_size=cfg.video_output_size,
                    codec=cfg.video_output_codec,
                )

            start_frame, end_frame, stride = get_frame_range(cfg, videos)

            processed_count = 0

            for frame_id, frames in videos.iter_frames(
                start_frame=start_frame,
                end_frame=end_frame,
                stride=stride,
            ):
                print("")
                print(f"Frame {frame_id}")

                if processed_count == 0:
                    debug_grid_path = (
                        cfg.debug_dir / f"{activity.name}_frame_{frame_id:06d}_grid.jpg"
                    )

                    videos.save_debug_grid(
                        frame_id=frame_id,
                        output_path=debug_grid_path,
                    )

                    print(f"Image debug sauvegardée: {debug_grid_path}")

                detections_by_camera = detector.process(frames)
                print_detection_summary(frame_id, detections_by_camera)

                tracker.update(
                    frame_id=frame_id,
                    detections_by_camera=detections_by_camera,
                    cameras_by_camera=cameras,
                )

                tracker.print_summary(frame_id)

                num_learned_poses = run_learned_pose_for_tracks(
                    cfg=cfg,
                    frame_id=frame_id,
                    frames_by_camera=frames,
                    cameras_by_camera=cameras,
                    tracker=tracker,
                    learned_pose=learned_pose,
                    outputs=outputs,
                )

                print(f"[learned pose] poses reconstruites: {num_learned_poses}")

                if cfg.save_csv_3d:
                    outputs.write_pose_frame(
                        frame_id=frame_id,
                        tracks=tracker.active_tracks(),
                    )

                if cfg.save_reprojection_images and should_save_frame(
                    frame_id,
                    cfg.save_reprojection_every,
                ):
                    draw_reprojection_frame(
                        frame_id=frame_id,
                        activity_name=activity.name,
                        frames_by_camera=frames,
                        cameras_by_camera=cameras,
                        tracks=tracker.active_tracks(),
                        output_dir=cfg.reprojection_dir,
                    )

                if cfg.save_3d_images and should_save_frame(
                    frame_id,
                    cfg.save_3d_every,
                ):
                    draw_3d_scene_frame(
                        frame_id=frame_id,
                        activity_name=activity.name,
                        tracks=tracker.active_tracks(),
                        output_dir=cfg.vis3d_dir,
                    )

                if video_writers is not None:
                    video_writers.write_frame(
                        frame_id=frame_id,
                        frames_by_camera=frames,
                        cameras_by_camera=cameras,
                        tracks=tracker.active_tracks(),
                    )

                processed_count += 1

                if should_stop_after_count(cfg, processed_count):
                    break

            print("")
            print(f"Activité terminée: {activity.name}")
            print(f"Frames traitées: {processed_count}")

    finally:
        outputs.close()

        if video_writers is not None:
            video_writers.close()


def run_pipeline(cfg: Any) -> None:
    print("Démarrage de la pipeline")
    print_config_summary(cfg)

    activities = discover_activities(
        base_dir=cfg.base_videos_dir,
        camera_names=cfg.camera_names,
        device_id_by_camera=cfg.device_id_by_camera,
        video_extensions=cfg.video_extensions,
        activity_mode=cfg.activity_mode,
        selected_activity_name=cfg.selected_activity_name,
    )

    if not activities:
        print("Aucune activité exploitable trouvée.")
        return

    print(f"Activités trouvées: {len(activities)}")
    for activity in activities:
        print(f"  - {activity.name}")

    learned_pose = build_learned_pose_estimator(cfg)

    for activity in activities:
        run_activity(
            cfg=cfg,
            activity=activity,
            learned_pose=learned_pose,
        )

    print("")
    print("Pipeline terminée.")