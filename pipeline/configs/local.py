from pathlib import Path


# Racine du projet
project_root = Path(__file__).resolve().parents[1]


# Sujet à traiter
subject_name = "METASPORT12"


# Dossiers d'entrée et de sortie
# base_videos_dir = Path(
#     r"\\intram\Paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT12_undistort\Session1\Files\fichiers_video\Bruts"
# )
base_videos_dir = Path(
    "pipeline_butterworth/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts"
)

output_root = project_root / "outputs_s12_butterworth"


# Choix des activités
# "all" traite toutes les activités trouvées.
# "single" traite seulement selected_activity_name.
activity_mode = "all"
selected_activity_name = "CMJump_03"

video_extensions = [".mp4", ".avi"]

camera_names = ["cam1", "cam2", "cam3", "cam4"]

device_id_by_camera = {
    "cam1": "51433950",
    "cam2": "51434015",
    "cam3": "51470146",
    "cam4": "51470940",
}


# Modèles
models_dir = project_root / "models"

yolo_model_path = models_dir / "yolo" / "best.pt"
reid_model_path = models_dir / "reid" / "osnet_ain_x1_0_M.onnx"

# Modèle learned pose multi-head:
multihead_weights_dir = models_dir / "triangulation" / "multihead_markerset32"

alg_weights_path = multihead_weights_dir / "best_alg_multihead.pth"
vol_weights_path = multihead_weights_dir / "best_vol_multihead_sarandi_full.pth"

# Détection
yolo_confidence = 0.20
yolo_person_class = 0
yolo_tracker_config = "bytetrack.yaml"

rtmlib_device = "cuda"
rtmlib_backend = "onnxruntime"

pose_min_joint_confidence = 0.25
pose_min_valid_joints = 4
bbox_pose_association_max_distance = 200.0

reid_use_torso_crop = True
reid_bbox_margin_ratio = 0.10


# Tracking multi-vues
track_patience = 200
min_hits_to_confirm = 3

intra_frame_appearance_threshold = 0.40
association_threshold = 0.85
association_margin = 0.15

min_confidence_for_new_track = 0.55
min_views_for_new_track = 1

feature_gallery_size = 100
feature_update_weight = 0.05


tracklet_conflict_max_age = 15

process_unmatched_groups_by_views_first = True

occlusion_merge_max_3d_distance_mm = 700.0

appearance_weight = 0.50
motion_weight = 0.30
spatial_weight = 0.20

reprojection_error_threshold = 120.0
spatial_consistency_threshold_mm = 1500.0

duplicate_merge_distance_mm = 600.0
duplicate_merge_appearance = 0.30
group_safety_merge_distance_mm = 400.0

occlusion_xy_distance_mm = 900.0
max_occlusion_frames = 120

reactivation_appearance_threshold = 0.35
occlusion_reid_threshold_mm = 1200.0

min_visibility_for_new_track = 0.40

save_association_debug = True


# Qualité des détections pour le tracking
min_detection_confidence_for_matching = 0.10
min_detection_confidence_for_grouping = 0.20

min_detection_confidence_for_new_track_single_view = 0.70
min_detection_confidence_for_new_track_multiview = 0.55

min_bbox_height_for_new_track = 80
min_bbox_width_for_new_track = 30

allow_featureless_detection_for_matching = True
allow_featureless_detection_for_new_track = False

featureless_pair_cost = 0.85
strong_spatial_score_for_featureless_pair = 0.65


# Réassociation mono-vue
single_view_reassociation_margin = 0.20
single_view_reassociation_max_score = 0.85
single_view_reassociation_max_missing = 80
single_view_motion_threshold_without_feature = 0.35
single_view_appearance_reject_threshold = 0.55


# Triangulation apprenable
image_size = (384, 384)

min_views_for_learned_pose = 2

learned_pose_use_displayed_track_boxes = True

learned_pose_select_best_view_subset = True
learned_pose_max_subset_size = 4
learned_pose_min_subset_reprojection_score = 0.20
learned_pose_prefer_more_views_bonus = 0.02
learned_pose_debug_view_subsets = False
learned_pose_subset_search_only_if_score_below = 0.4
learned_pose_try_pair_subsets = False

learned_pose_reuse_selected_views = True
learned_pose_reuse_selected_views_frames = 10
learned_pose_reselect_if_score_below = 0.25

reprojection_score_enabled = True
reprojection_min_joint_conf = 0.30
reprojection_max_norm_err = 0.10
reprojection_full_views = 3

learned_pose_model_type = "multihead"

n_model_joints = 83
n_output_joints = 18

n_custom_joints = 19
n_ecomocap_markers = 32
n_metasport_markers = 32

# Sortie principale
learned_pose_output_head = "custom18"

# Visualisation :
# - custom18 : squelette final uniquement
# - metasport32 : points MetaSport uniquement
# - custom18_metasport : squelette final + points MetaSport
visualization_mode = "custom18_metasport"

show_metasport_head = True
show_ecomocap_head = False

# Options vis3d:
show_vis3d_joint_numbers = False
show_vis3d_custom18_joint_numbers = False
show_vis3d_metasport_marker_numbers = False

show_vis3d_joint_legend = False
show_vis3d_person_legend = True
show_vis3d_person_labels_on_scene = False
vis3d_joint_label_mode = "index"

save_metasport_eval_csv = True

save_metasport_debug_csv = True
save_ecomocap_debug_csv = False

xcp_variant = "2B"
xcp_t_scale = 1.0

cuboid_side = 2500.0
learned_pose_volume_size = 96

learned_pose_device = "cuda"

learned_pose_bbox_scale = 1.25
learned_pose_square_crop = False

learned_pose_min_reliable_joints = 6
learned_pose_min_reliable_joint_ratio = 0.30
learned_pose_min_pose_confidence = 0.15
learned_pose_reject_edge_truncated = False
learned_pose_min_bbox_height_ratio = 0.08
learned_pose_min_bbox_width_ratio = 0.025

learned_pose_reject_large_root_jump = False
learned_pose_max_root_jump_mm = 1000.0

show_unreliable_reprojection = True
save_unreliable_pose_csv = True


# Boucle vidéo
start_frame = 1
end_frame = None
frame_stride = 1

max_frames_per_activity = None

butterworth_enabled = True
butterworth_order = 4
butterworth_cutoff_mode = "activity_profile"
butterworth_default_cutoff_hz = 6.0

butterworth_activity_cutoff_hz = {
    "CMJump": 9.0,
    "SquatJump": 9.0,

    "Deadlift": 4.0,
    "Deadlift_Polystyrene": 4.0,
    "DeadliftPolystyrene": 4.0,

    "EpauleDebout": 5.0,
    "EpauleDebout_Polystyrene": 5.0,
    "EpauleDeboutPolystyrene": 5.0,

    "SledPush": 7.0,

    "SprintTapis": 10.0,
    "Sprint": 10.0,
}

butterworth_min_segment_frames = 30
butterworth_max_gap_frames = 5

butterworth_render_filtered_outputs = True
butterworth_render_filtered_videos = True
butterworth_render_filtered_vis3d = True
butterworth_csv_dir = output_root / "csv_butterworth"
butterworth_video_dir = output_root / "videos_butterworth"
butterworth_vis3d_dir = output_root / "vis3d_butterworth"

gt_show_comparison = True
gt_render_on_filtered_videos = True
gt_render_on_filtered_vis3d = True

# Afficher toutes les sorties sur la même visu :
# - prédiction custom18
# - prédiction MetaSport32
# - GT custom18
# - GT MetaSport32
comparison_show_pred_custom18 = True
comparison_show_pred_metasport32 = True
comparison_show_gt_custom18 = True
comparison_show_gt_metasport32 = True

comparison_pred_skeleton_colors = {
    1: "orchid",
    2: "darkgreen",
    3: "hotpink",
}
comparison_pred_skeleton_colors_bgr = {
    1: (214, 112, 218), 
    2: (0, 100, 0),      
    3: (180, 105, 255), 
}

comparison_pred_marker_colors = {
    1: "yellow",
    2: "red",
    3: "blue",
}
comparison_pred_marker_colors_bgr = {
    1: (0, 255, 255), 
    2: (0, 0, 255),    
    3: (255, 0, 0),    
}

comparison_gt_skeleton_color = "orange"
comparison_gt_marker_color = "cyan"
comparison_gt_skeleton_color_bgr = (0, 165, 255) 
comparison_gt_marker_color_bgr = (255, 255, 0)   

comparison_gt_skeleton_label = "GT skeleton"
comparison_gt_marker_label = "GT Meta32"
comparison_gt_skeleton_linestyle = "--"

comparison_pred_marker_style = "o"
comparison_gt_marker_style = "o"
comparison_pred_marker_size = 16
comparison_gt_marker_size = 18
comparison_gt_marker_size_video = 4

gt_label = comparison_gt_skeleton_label
gt_color_bgr = comparison_gt_skeleton_color_bgr

gt_metasport_subject_root = base_videos_dir.parents[1]

gt_vicon_frequency = 200.0
gt_video_frequency = 100.0

gt_metasport_frame_offsets = {}

# Conversion GT avant projection/rendu.
# Les joints/markers GT issus de points3D_t.mat sont en mètres,
# alors que la projection XCP et la pipeline utilisent des coordonnées en millimètres.
gt_coordinate_scale = 1000.0
gt_coordinate_offset = (0.0, 0.0, 0.0)

gt_debug_coordinates = True

# Sauvegarde des sorties
save_csv_3d = True
save_reprojection_images = False
save_3d_images = False
save_failure_log = True

# Exports
save_debug_exports = True

save_reprojection_every = 1
save_3d_every = 1

# Sauvegarde vidéo finale
save_output_videos = False

# None garde la taille originale des vidéos
video_output_size = None

video_output_codec = "mp4v"


# Dossiers de sortie
csv_dir = output_root / "csv"
video_output_dir = output_root / "videos"
reprojection_dir = output_root / "reprojection"
vis3d_dir = output_root / "vis3d"
debug_dir = output_root / "debug"

for directory in [
    output_root,
    csv_dir,
    video_output_dir,
    reprojection_dir,
    vis3d_dir,
    debug_dir,
    butterworth_csv_dir,
    butterworth_video_dir,
    butterworth_vis3d_dir,
]:
    directory.mkdir(parents=True, exist_ok=True)
