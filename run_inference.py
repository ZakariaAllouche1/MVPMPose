# run_inference.py
import os
import sys
import json
import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, SCRIPT_DIR)

from easydict import EasyDict
import torchvision.transforms as transforms

from mvn.models.triangulation import VolumetricTriangulationNet, AlgebraicTriangulationNet
from mvn.utils.multiview import Camera
from camera_config import load_cameras_params


# =========================
# CONFIG
# =========================
BASE_DIR = os.path.join(SCRIPT_DIR, "001_ENSAMPOSE_WALK")
SUBJECT = "S01"
ACTION = "walk"
IMSEQ = "imgSeq_walk_01"

CAMERAS_PARAMS = load_cameras_params(BASE_DIR, SUBJECT)

SUBJECT_DIR = os.path.join(BASE_DIR, SUBJECT, "processed", "action", ACTION, IMSEQ)

WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT,
    "data/pretrained/human36m/",
    "human36m_vol_softmax_10-08-2019/",
    "checkpoints/0040/weights.pth"
)

ALG_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "data/pretrained/human36m/",
    "human36m_alg_10-04-2019/",
    "checkpoints/0060/weights.pth"
)

OUTPUT_DIR = "output_inference_vol_debug"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (384, 384)
N_JOINTS = 17

CAMERA_DIRS = ["camera1L", "camera1R", "camera2L", "camera2R"]

BBOX_SCALE = 1.0
CUBOID_SIDE = 2500.0

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

SKELETON = [
    (0, 1),
    (1, 2),
    (2, 6),
    (3, 4),
    (4, 5),
    (3, 6),
    (6, 7),
    (7, 8),
    (8, 16),
    (16, 9),
    (7, 12),
    (12, 11),
    (11, 10),
    (7, 13),
    (13, 14),
    (14, 15)
]


# =========================
# UTILS
# =========================
def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def clip_bbox(bbox, img_shape):
    top, left, bottom, right = bbox
    h, w = img_shape[:2]

    top = max(0, min(h - 1, int(top)))
    left = max(0, min(w - 1, int(left)))
    bottom = max(top + 1, min(h, int(bottom)))
    right = max(left + 1, min(w, int(right)))

    return [top, left, bottom, right]


def expand_square_bbox(bbox, img_shape, scale=1.0):
    bbox = clip_bbox(bbox, img_shape)
    if scale == 1.0:
        return bbox

    top, left, bottom, right = bbox

    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    bw = right - left
    bh = bottom - top
    side = max(bw, bh) * scale

    new_left = int(round(cx - side / 2))
    new_right = int(round(cx + side / 2))
    new_top = int(round(cy - side / 2))
    new_bottom = int(round(cy + side / 2))

    return clip_bbox([new_top, new_left, new_bottom, new_right], img_shape)


def spread_from_pelvis(kps, pelvis_idx=6):
    return np.linalg.norm(kps - kps[pelvis_idx], axis=1)


# =========================
# BBOXES
# =========================
def load_bboxes():
    json_path = os.path.join(SUBJECT_DIR, "annot.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["bboxes"]


BBOXES = load_bboxes()


def get_bbox(cam_name, frame_idx_zero_based):
    try:
        b = BBOXES[cam_name]
        top = int(b["top"][frame_idx_zero_based])
        left = int(b["left"][frame_idx_zero_based])
        bottom = int(b["bottom"][frame_idx_zero_based])
        right = int(b["right"][frame_idx_zero_based])

        if bottom > top and right > left:
            return [top, left, bottom, right]
    except (KeyError, IndexError):
        pass
    return None


# =========================
# PROJECTION
# =========================
def build_proj_matrix(cam_params, bbox=None, orig_size=None):
    K = cam_params["K"].copy()
    R = cam_params["R"]
    t = cam_params["t"]

    if bbox is not None and orig_size is not None:
        top, left, bottom, right = bbox
        crop_h = bottom - top
        crop_w = right - left

        K[0, 2] -= left
        K[1, 2] -= top

        K[0, 0] *= IMAGE_SIZE[1] / crop_w
        K[0, 2] *= IMAGE_SIZE[1] / crop_w
        K[1, 1] *= IMAGE_SIZE[0] / crop_h
        K[1, 2] *= IMAGE_SIZE[0] / crop_h

    Rt = np.hstack([R, t])
    return K @ Rt


# =========================
# MODELS
# =========================
def load_model():
    config = EasyDict({
        "model": {
            "name": "vol",
            "kind": "mpii",   # IMPORTANT
            "backbone": {
                "name": "pose_resnet",
                "style": "simple",
                "num_layers": 152,
                "num_deconv_layers": 3,
                "num_deconv_filters": [256, 256, 256],
                "num_deconv_kernels": [4, 4, 4],
                "num_joints": N_JOINTS,
                "image_size": [384, 384],
                "heatmap_size": [96, 96],
                "init_weights": False,
            },
            "num_joints": N_JOINTS,
            "volume_size": 64,
            "volume_multiplier": 1.0,
            "volume_softmax": True,
            "volume_aggregation_method": "softmax",
            "triangulation_softmax": True,
            "use_gt_pelvis": False,
            "pelvis_type": "pred",
            "cuboid_side": CUBOID_SIDE,
            "heatmap_softmax": True,
            "heatmap_multiplier": 100.0,
        }
    })

    print(f"Chargement du modèle volumétrique sur {DEVICE}...")
    model = VolumetricTriangulationNet(config, device=DEVICE)

    checkpoint = safe_torch_load(WEIGHTS_PATH, map_location=DEVICE)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = strip_module_prefix(state_dict)

    load_result = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)

    model.eval()
    model.to(DEVICE)
    print("Modèle volumétrique chargé")
    return model


def load_alg_model():
    config = EasyDict({
        "model": {
            "name": "alg",
            "use_confidences": True,
            "heatmap_softmax": True,
            "heatmap_multiplier": 100.0,
            "backbone": {
                "name": "pose_resnet",
                "style": "simple",
                "num_layers": 152,
                "num_deconv_layers": 3,
                "num_deconv_filters": [256, 256, 256],
                "num_deconv_kernels": [4, 4, 4],
                "num_joints": N_JOINTS,
                "image_size": [384, 384],
                "heatmap_size": [96, 96],
                "init_weights": False,
                "alg_confidences": True,
                "vol_confidences": False,
            },
        }
    })

    print(f"Chargement du modèle algébrique sur {DEVICE}...")
    model = AlgebraicTriangulationNet(config, device=DEVICE)

    checkpoint = safe_torch_load(ALG_WEIGHTS, map_location=DEVICE)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = strip_module_prefix(state_dict)

    load_result = model.load_state_dict(state_dict, strict=False)
    print("ALG missing keys:", load_result.missing_keys)
    print("ALG unexpected keys:", load_result.unexpected_keys)

    model.eval()
    model.to(DEVICE)
    print("Modèle algébrique chargé")
    return model


# =========================
# INFERENCE
# =========================
def run_frame_alg_vol(alg_model, vol_model, frame_idx):
    images_list = []
    proj_list = []
    cameras_batch = []

    for cam_idx, cam_name in enumerate(CAMERA_DIRS):
        cam_params = CAMERAS_PARAMS[cam_idx]
        img_path = os.path.join(SUBJECT_DIR, cam_name, f"img_{frame_idx:04d}.jpg")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image illisible : {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        bbox = get_bbox(cam_name, frame_idx - 1)
        if bbox is None:
            bbox = [0, 0, orig_h, orig_w]
            print(f"Pas de bbox pour {cam_name} frame {frame_idx} -> image complète")

        bbox = expand_square_bbox(bbox, img.shape, scale=BBOX_SCALE)

        top, left, bottom, right = bbox
        img_crop = img[top:bottom, left:right]

        if img_crop.size == 0:
            raise ValueError(f"Crop vide pour {cam_name} frame {frame_idx}, bbox={bbox}")

        img_resized = cv2.resize(img_crop, IMAGE_SIZE)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        tensor = NORMALIZE(tensor)

        P = build_proj_matrix(cam_params, bbox, (orig_h, orig_w))

        dist = cam_params.get("dist", None)
        dist = dist.copy() if dist is not None else None

        cam = Camera(
            cam_params["R"].copy(),
            cam_params["t"].copy(),
            cam_params["K"].copy(),
            dist,
            cam_params["name"]
        )

        cam.update_after_crop([left, top, right, bottom])
        crop_h = bottom - top
        crop_w = right - left
        cam.update_after_resize((crop_h, crop_w), IMAGE_SIZE)

        cameras_batch.append([cam])   # [n_views][batch_size]
        images_list.append(tensor)
        proj_list.append(torch.from_numpy(P).float())

    images = torch.stack(images_list).unsqueeze(0).to(DEVICE)
    proj_matrices = torch.stack(proj_list).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        alg_output = alg_model(images, proj_matrices, None)
        alg_kps = alg_output[0][0].cpu().numpy().astype(np.float32)

        pred_kps_4d = np.zeros((N_JOINTS, 4), dtype=np.float32)
        pred_kps_4d[:, :3] = alg_kps
        pred_kps_4d[:, 3] = 1.0

        fake_batch = {
            "cameras": cameras_batch,
            "pred_keypoints_3d": [pred_kps_4d]
        }

        vol_output = vol_model(images, proj_matrices, fake_batch)
        vol_kps = vol_output[0][0].cpu().numpy()
        vol_base_point = vol_output[6][0].cpu().numpy()

    return alg_kps, vol_kps, vol_base_point


# =========================
# VISUALISATION
# =========================
def visualize_3d(kps, frame_idx, tag):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], c="red", s=60, zorder=5)

    for i, j in SKELETON:
        ax.plot(
            [kps[i, 0], kps[j, 0]],
            [kps[i, 1], kps[j, 1]],
            [kps[i, 2], kps[j, 2]],
            "b-",
            linewidth=2
        )

    mins = kps.min(axis=0)
    maxs = kps.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-70)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(f"Pose 3D — frame {frame_idx} — {tag}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"pose3d_{frame_idx:06d}_{tag}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_reprojection(kps, frame_idx, tag):
    for cam_idx, cam_name in enumerate(CAMERA_DIRS):
        cam_params = CAMERAS_PARAMS[cam_idx]
        img_path = os.path.join(SUBJECT_DIR, cam_name, f"img_{frame_idx:04d}.jpg")

        img = cv2.imread(img_path)
        if img is None:
            continue

        P = build_proj_matrix(cam_params)

        for j in range(N_JOINTS):
            pt = np.append(kps[j], 1.0)
            proj = P @ pt

            if proj[2] <= 1e-6:
                continue

            x = int(proj[0] / proj[2])
            y = int(proj[1] / proj[2])

            h, w = img.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(
                    img, str(j), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1, cv2.LINE_AA
                )

        for i, j in SKELETON:
            pt_i = P @ np.append(kps[i], 1.0)
            pt_j = P @ np.append(kps[j], 1.0)

            if pt_i[2] <= 1e-6 or pt_j[2] <= 1e-6:
                continue

            xi = int(pt_i[0] / pt_i[2])
            yi = int(pt_i[1] / pt_i[2])
            xj = int(pt_j[0] / pt_j[2])
            yj = int(pt_j[1] / pt_j[2])

            h, w = img.shape[:2]
            if 0 <= xi < w and 0 <= yi < h and 0 <= xj < w and 0 <= yj < h:
                cv2.line(img, (xi, yi), (xj, yj), (255, 0, 0), 2)

        out = os.path.join(OUTPUT_DIR, f"reproj_frame{frame_idx:06d}_{cam_name}_{tag}.jpg")
        cv2.imwrite(out, img)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    alg_model = load_alg_model()
    vol_model = load_model()

    for frame_idx in range(1, 266):
        print(f"\n── Frame {frame_idx} ──")
        try:
            alg_kps, vol_kps, vol_base_point = run_frame_alg_vol(alg_model, vol_model, frame_idx)

            print("VOL spread from pelvis:", spread_from_pelvis(vol_kps).round(1))
            print("ALG spread from pelvis:", spread_from_pelvis(alg_kps).round(1))

            np.save(os.path.join(OUTPUT_DIR, f"kps3d_{frame_idx:06d}_alg.npy"), alg_kps)
            np.save(os.path.join(OUTPUT_DIR, f"kps3d_{frame_idx:06d}_vol.npy"), vol_kps)

            alg_pelvis = alg_kps[6]
            vol_pelvis = vol_kps[6]

            print(f"ALG pelvis : {alg_pelvis.round(1)} mm")
            print(f"VOL pelvis : {vol_pelvis.round(1)} mm")
            print(f"VOL base point : {vol_base_point.round(1)} mm")
            print(f"||base_point - alg_pelvis|| = {np.linalg.norm(vol_base_point - alg_pelvis):.2f} mm")
            print(f"mean joint diff ALG/VOL = {np.mean(np.linalg.norm(vol_kps - alg_kps, axis=1)):.2f} mm")

            visualize_3d(alg_kps, frame_idx, "alg")
            visualize_reprojection(alg_kps, frame_idx, "alg")

            visualize_3d(vol_kps, frame_idx, "vol")
            visualize_reprojection(vol_kps, frame_idx, "vol")

        except Exception as e:
            print(f"Erreur : {e}")
            import traceback
            traceback.print_exc()

    print(f"\nTerminé → résultats dans '{OUTPUT_DIR}/'")