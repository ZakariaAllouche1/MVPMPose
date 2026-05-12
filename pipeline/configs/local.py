from pathlib import Path


# Racine du projet
project_root = Path(__file__).resolve().parents[1]


# Sujet à traiter
subject_name = "METASPORT12"


# Dossiers d'entrée et de sortie
base_videos_dir = Path(
    r"\\intram\Paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT12_undistort\Session1\Files\fichiers_video\Bruts"
)

output_root = project_root / "outputs"


# Choix des activités
# "all" traite toutes les activités trouvées.
# "single" traite seulement selected_activity_name.
activity_mode = "all"
selected_activity_name = "A_REMPLACER_PAR_TON_ACTIVITE"


# Les vidéos peuvent être en mp4 ou en avi
video_extensions = [".mp4", ".avi"]


# Caméras utilisées dans la pipeline
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

alg_weights_path = models_dir / "triangulation" / "alg_weights.pth"
vol_weights_path = models_dir / "triangulation" / "best_vol_17_masked.pth"


# Détection
# On garde un seuil plutôt bas pour ne pas casser les tracks pendant les frames difficiles.
# La création d'une nouvelle track reste filtrée par min_confidence_for_new_track.
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

min_confidence_for_new_track = 0.55
min_views_for_new_track = 1

feature_gallery_size = 100
feature_update_weight = 0.05

tracklet_conflict_max_age = 15

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


# Triangulation apprenable
image_size = (384, 384)

min_views_for_learned_pose = 2

n_model_joints = 17
n_output_joints = 15

xcp_variant = "2B"
xcp_t_scale = 1.0

cuboid_side = 2500.0

learned_pose_device = "cuda"

learned_pose_bbox_scale = 1.25
learned_pose_square_crop = True


# Boucle vidéo
start_frame = 1
end_frame = None
frame_stride = 1

# None traite toutes les frames disponibles.
max_frames_per_activity = None


# Sauvegarde des sorties
save_csv_3d = True
save_reprojection_images = True
save_3d_images = True
save_failure_log = True

save_reprojection_every = 10
save_3d_every = 10


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
]:
    directory.mkdir(parents=True, exist_ok=True)