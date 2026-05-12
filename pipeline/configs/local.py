from pathlib import Path


# Racine du projet
project_root = Path(__file__).resolve().parents[1]


# Sujet à traiter
subject_name = "METASPORT12"


# Dossiers d'entrée et de sortie
base_videos_dir = (
    project_root
    / "metasport"
    / "DATA"
    / f"{subject_name}_undistort"
    / "Session1"
    / "Files"
    / "fichiers_video"
    / "Bruts"
    / f"{subject_name}_undistorted"
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
yolo_model_path = project_root / "runs" / "detect" / "train" / "weights" / "best.pt"
reid_model_path = project_root / "pretrained_models" / "osnet_ain_x1_0_M.onnx"

alg_weights_path = (
    project_root
    / "data"
    / "pretrained"
    / "human36m"
    / "human36m_alg_10-04-2019"
    / "checkpoints"
    / "0060"
    / "weights.pth"
)

vol_weights_path = (
    project_root
    / "output_vol_17_masked"
    / "checkpoints"
    / "best_vol_17_masked.pth"
)


# Détection
yolo_confidence = 0.35
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
track_patience = 150
min_hits_to_confirm = 3

intra_frame_appearance_threshold = 0.35
association_threshold = 0.75

min_confidence_for_new_track = 0.45
min_views_for_new_track = 1

feature_gallery_size = 100
feature_update_weight = 0.05

tracklet_conflict_max_age = 10

motion_weight = 0.30
appearance_weight = 0.70


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