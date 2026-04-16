# run_xcp_inference_vol.py
import os
import sys
import csv
import glob
import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from easydict import EasyDict
import torchvision.transforms as transforms

from mvn.models.triangulation import (
    AlgebraicTriangulationNet,
    VolumetricTriangulationNet,
)
from mvn.utils.multiview import Camera
from xcp_utils import load_cameras_params_from_xcp


# =========================
# CONFIG
# =========================
OUTPUT_ROOT = "output_xcp_alltests_alg_vol"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (384, 384)
N_JOINTS = 17

SUBJECT_NAME = "METASPORT12"

BASE_DIR = f"./metasport/DATA/{SUBJECT_NAME}_undistort/Session1/Files/fichiers_video"
BRUTS_DIR = os.path.join(BASE_DIR, "Bruts", f"{SUBJECT_NAME}_undistorted")
TRACKED_DIR = os.path.join(BASE_DIR, "Traites", "RtmLib", f"{SUBJECT_NAME}_tracked")

CAMERA_NAMES = ["camera1L", "camera1R", "camera2L", "camera2R"]

DEVICEID_BY_NAME = {
    "camera1L": "51433950",
    "camera1R": "51434015",
    "camera2L": "51470146",
    "camera2R": "51470940",
}

CSV_CAMERA_KEY_BY_NAME = {
    "camera1L": "0",
    "camera1R": "1",
    "camera2L": "2",
    "camera2R": "3",
}

CSV_PERSON_ID = None

XCP_VARIANT = "2B"
T_SCALE = 1.0
BBOX_SCALE = 1.0
CUBOID_SIDE = 2500.0

ALG_WEIGHTS = (
    "./data/pretrained/human36m/"
    "human36m_alg_10-04-2019/"
    "checkpoints/0060/weights.pth"
)

VOL_WEIGHTS = (
    "./data/pretrained/human36m/"
    "human36m_vol_softmax_10-08-2019/"
    "checkpoints/0040/weights.pth"
)

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

SKELETON = [
    (0, 1), (1, 2), (2, 6),
    (3, 4), (4, 5), (3, 6),
    (6, 7), (7, 8), (8, 16), (16, 9),
    (7, 12), (12, 11), (11, 10),
    (7, 13), (13, 14), (14, 15)
]


# =========================
# HELPERS
# =========================
def infer_cam_name_from_path(path):
    basename = os.path.basename(path)
    for cam_name, device_id in DEVICEID_BY_NAME.items():
        if device_id in basename:
            return cam_name
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def sanitize_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def mean_joint_diff(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def spread_from_pelvis(kps, pelvis_idx=6):
    return np.linalg.norm(kps - kps[pelvis_idx], axis=1)


def discover_tests():
    if not os.path.isdir(BRUTS_DIR):
        raise FileNotFoundError(f"Dossier introuvable: {BRUTS_DIR}")

    xcp_paths = sorted(glob.glob(os.path.join(BRUTS_DIR, "*.xcp")))
    if not xcp_paths:
        raise FileNotFoundError(f"Aucun fichier .xcp trouvé dans {BRUTS_DIR}")

    tests = []

    for xcp_path in xcp_paths:
        test_name = os.path.splitext(os.path.basename(xcp_path))[0]

        video_by_name = {}
        missing_videos = []

        for cam_name in CAMERA_NAMES:
            device_id = DEVICEID_BY_NAME[cam_name]
            matches = sorted(glob.glob(os.path.join(BRUTS_DIR, f"{test_name}.{device_id}*.mp4")))
            if not matches:
                missing_videos.append(cam_name)
            else:
                video_by_name[cam_name] = matches[0]

        if missing_videos:
            print(f"[SKIP] {test_name}: vidéos manquantes pour {missing_videos}")
            continue

        bbox_csv_paths = sorted(glob.glob(os.path.join(TRACKED_DIR, f"{test_name}*_training_data.csv")))
        if not bbox_csv_paths:
            print(f"[SKIP] {test_name}: aucun CSV bbox trouvé dans {TRACKED_DIR}")
            continue

        tests.append({
            "test_name": test_name,
            "xcp_path": xcp_path,
            "video_by_name": video_by_name,
            "bbox_csv_paths": bbox_csv_paths,
        })

    return tests


# =========================
# CAMERAS
# =========================
def load_cameras_for_test(xcp_path):
    return load_cameras_params_from_xcp(
        xcp_path=xcp_path,
        deviceid_by_name=DEVICEID_BY_NAME,
        camera_names=CAMERA_NAMES,
        variant=XCP_VARIANT,
        t_scale=T_SCALE
    )


def build_proj_matrix(cam_params, bbox=None):
    K = cam_params["K"].copy()
    R = cam_params["R"]
    t = cam_params["t"]

    if bbox is not None:
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
# CSV BBOX PROVIDER
# =========================
def normalize_camera_key(value):
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


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
    h, w = img_shape[:2]

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


def find_col(fieldnames, candidates, required=True):
    lower_map = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise KeyError(
            f"Colonne introuvable. Candidats testés: {candidates}. Colonnes disponibles: {fieldnames}"
        )
    return None


class CSVBBoxProvider:
    def __init__(
        self,
        csv_paths,
        deviceid_by_name,
        bbox_scale=1.0,
        person_id=None,
        csv_camera_key_by_name=None,
    ):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]

        self.csv_paths = csv_paths
        self.deviceid_by_name = {
            k: normalize_camera_key(v) for k, v in deviceid_by_name.items()
        }
        self.csv_camera_key_by_name = {
            k: normalize_camera_key(v) for k, v in (csv_camera_key_by_name or {}).items()
        }
        self.bbox_scale = bbox_scale
        self.person_id = person_id
        self.data = {}
        self._load_all()

    def _infer_default_cam_key(self, csv_path):
        cam_name = infer_cam_name_from_path(csv_path)
        if cam_name is None:
            return None

        if cam_name in self.csv_camera_key_by_name:
            return self.csv_camera_key_by_name[cam_name]
        return self.deviceid_by_name.get(cam_name)

    def _load_all(self):
        for csv_path in self.csv_paths:
            self._load_one(csv_path)

        print(f"\nCSV bbox total fusionné: {len(self.data)} entrées uniques")

    def _load_one(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV bbox introuvable: {csv_path}")

        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if not rows:
            raise ValueError(f"CSV bbox vide: {csv_path}")

        print("\n=== CSV BBOX ===")
        print("Fichier :", csv_path)
        print("Colonnes détectées :", fieldnames)

        cam_col = find_col(
            fieldnames,
            ["camera", "cam_name", "camera_name", "cam_id", "camera_id", "device_id"],
            required=False
        )
        frame_col = find_col(
            fieldnames,
            ["frame", "frame_idx", "frame_id", "image_id"]
        )
        person_col = find_col(
            fieldnames,
            ["person_id", "person"],
            required=False
        )

        x1_col = find_col(fieldnames, ["bbox_x1", "x1", "xmin", "left"], required=False)
        y1_col = find_col(fieldnames, ["bbox_y1", "y1", "ymin", "top"], required=False)
        x2_col = find_col(fieldnames, ["bbox_x2", "x2", "xmax", "right"], required=False)
        y2_col = find_col(fieldnames, ["bbox_y2", "y2", "ymax", "bottom"], required=False)

        x_col = find_col(fieldnames, ["x"], required=False)
        y_col = find_col(fieldnames, ["y"], required=False)
        w_col = find_col(fieldnames, ["w", "width"], required=False)
        h_col = find_col(fieldnames, ["h", "height"], required=False)

        default_cam_key = self._infer_default_cam_key(csv_path)
        if cam_col is None and default_cam_key is None:
            raise KeyError(
                f"Aucune colonne caméra trouvée dans {csv_path} et impossible d'inférer la caméra depuis le nom du fichier."
            )

        frame_values = []
        for row in rows:
            try:
                frame_values.append(int(round(float(row[frame_col]))))
            except Exception:
                pass

        if not frame_values:
            raise ValueError(f"Impossible de lire les indices de frame dans le CSV: {csv_path}")

        min_frame = min(frame_values)
        frame_offset = 0 if min_frame == 1 else 1

        duplicate_count = 0
        kept_count = 0
        unique_before = len(self.data)

        for row in rows:
            try:
                if self.person_id is not None and person_col is not None:
                    try:
                        pid = int(float(row[person_col]))
                        if pid != self.person_id:
                            continue
                    except Exception:
                        continue

                cam_key = normalize_camera_key(row[cam_col]) if cam_col is not None else default_cam_key
                frame_raw = int(round(float(row[frame_col])))
                frame_idx_1based = frame_raw + frame_offset

                if None not in [x1_col, y1_col, x2_col, y2_col]:
                    x1 = float(row[x1_col])
                    y1 = float(row[y1_col])
                    x2 = float(row[x2_col])
                    y2 = float(row[y2_col])
                elif None not in [x_col, y_col, w_col, h_col]:
                    x1 = float(row[x_col])
                    y1 = float(row[y_col])
                    x2 = x1 + float(row[w_col])
                    y2 = y1 + float(row[h_col])
                else:
                    raise KeyError(
                        "Colonnes bbox non trouvées. Il faut bbox_x1,bbox_y1,bbox_x2,bbox_y2 "
                        "ou x1,y1,x2,y2 ou left,top,right,bottom ou x,y,w,h."
                    )

                bbox = [int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2))]
                key = (cam_key, frame_idx_1based)

                if key in self.data:
                    duplicate_count += 1

                self.data[key] = bbox
                kept_count += 1

            except Exception:
                continue

        print(f"CSV bbox chargé: +{len(self.data) - unique_before} nouvelles entrées uniques")
        print(f"Lignes gardées: {kept_count}")
        print(f"Décalage de frame appliqué: {frame_offset}")
        if duplicate_count > 0:
            print(f"Attention: {duplicate_count} doublons cam/frame trouvés, la dernière bbox a été gardée.")

        camera_keys_in_csv = sorted({k[0] for k in self.data.keys()})
        print("Clés caméra actuellement disponibles :", camera_keys_in_csv)

    def get_bbox(self, cam_name, frame_idx_1based, img_shape):
        keys_to_try = [cam_name]

        if cam_name in self.deviceid_by_name:
            keys_to_try.append(self.deviceid_by_name[cam_name])

        if cam_name in self.csv_camera_key_by_name:
            keys_to_try.append(self.csv_camera_key_by_name[cam_name])

        seen = set()
        keys_to_try = [k for k in keys_to_try if not (k in seen or seen.add(k))]

        for cam_key in keys_to_try:
            bbox = self.data.get((cam_key, frame_idx_1based))
            if bbox is not None:
                return expand_square_bbox(bbox, img_shape, scale=self.bbox_scale)

        raise RuntimeError(
            f"Aucune bbox CSV pour cam={cam_name} "
            f"(clés testées: {keys_to_try}) frame={frame_idx_1based}"
        )


# =========================
# MODELS
# =========================
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
    state_dict = sanitize_state_dict(state_dict)

    load_result = model.load_state_dict(state_dict, strict=False)
    print("ALG missing keys:", load_result.missing_keys)
    print("ALG unexpected keys:", load_result.unexpected_keys)

    model.eval()
    model.to(DEVICE)
    print("Modèle algébrique chargé")
    return model


def load_vol_model():
    config = EasyDict({
        "model": {
            "name": "vol",
            "kind": "mpii",
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

    checkpoint = safe_torch_load(VOL_WEIGHTS, map_location=DEVICE)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = sanitize_state_dict(state_dict)

    load_result = model.load_state_dict(state_dict, strict=False)
    print("VOL missing keys:", load_result.missing_keys)
    print("VOL unexpected keys:", load_result.unexpected_keys)

    model.eval()
    model.to(DEVICE)
    print("Modèle volumétrique chargé")
    return model


# =========================
# VIDEO I/O
# =========================
def open_video_caps(video_by_name):
    caps = {}
    frame_counts = {}

    for cam_name in CAMERA_NAMES:
        path = video_by_name[cam_name]
        if not os.path.exists(path):
            raise RuntimeError(f"Chemin introuvable pour {cam_name}: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir {path}")

        caps[cam_name] = cap
        frame_counts[cam_name] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return caps, frame_counts


# =========================
# INFERENCE
# =========================
def run_frame_alg_vol(frames_bgr, frame_idx, bbox_provider, cameras_params, alg_model, vol_model):
    images_list = []
    proj_list = []
    cameras_batch = []

    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        cam_params = cameras_params[cam_idx]
        img_bgr = frames_bgr[cam_name]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        bbox = bbox_provider.get_bbox(cam_name, frame_idx, img_bgr.shape)
        top, left, bottom, right = bbox

        img_crop = img_rgb[top:bottom, left:right]
        if img_crop.size == 0:
            raise RuntimeError(f"Crop vide pour {cam_name} frame {frame_idx}, bbox={bbox}")

        img_resized = cv2.resize(img_crop, IMAGE_SIZE)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        tensor = NORMALIZE(tensor)

        P = build_proj_matrix(cam_params, bbox=bbox)

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

        cameras_batch.append([cam])
        images_list.append(tensor)
        proj_list.append(torch.from_numpy(P).float())

    images = torch.stack(images_list).unsqueeze(0).to(DEVICE)
    proj_matrices = torch.stack(proj_list).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        alg_output = alg_model(images, proj_matrices, None)
        alg_kps = alg_output[0][0].cpu().numpy().astype(np.float32)

        fake_batch = {
            "cameras": cameras_batch,
            "pred_keypoints_3d": [alg_kps],
        }

        vol_output = vol_model(images, proj_matrices, fake_batch)
        vol_kps = vol_output[0][0].cpu().numpy()
        vol_base_point = vol_output[6][0].cpu().numpy()

    return alg_kps, vol_kps, vol_base_point


# =========================
# CSV OUTPUT
# =========================
def keypoints3d_header():
    header = ["frame"]
    for j in range(N_JOINTS):
        header.extend([f"joint{j}_x", f"joint{j}_y", f"joint{j}_z"])
    return header


def append_keypoints3d_csv(csv_path, frame_idx, kps):
    ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(keypoints3d_header())

        row = [frame_idx]
        for j in range(kps.shape[0]):
            row.extend([float(kps[j, 0]), float(kps[j, 1]), float(kps[j, 2])])

        writer.writerow(row)


def append_global_keypoints3d_csv(csv_path, test_name, frame_idx, kps):
    ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["test_name"] + keypoints3d_header())

        row = [test_name, frame_idx]
        for j in range(kps.shape[0]):
            row.extend([float(kps[j, 0]), float(kps[j, 1]), float(kps[j, 2])])

        writer.writerow(row)


# =========================
# VISU
# =========================
def visualize_3d(kps, frame_idx, output_dir, title_prefix="Pose 3D"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], c="red", s=60)

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
    ax.set_title(f"{title_prefix} — frame {frame_idx}")

    ensure_dir(output_dir)
    out = os.path.join(output_dir, f"pose3d_{frame_idx:06d}.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_reprojection(kps, frame_idx, frames_bgr, cameras_params, output_dir):
    ensure_dir(output_dir)

    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        img = frames_bgr[cam_name].copy()
        cam_params = cameras_params[cam_idx]
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
                    img,
                    str(j),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )

        for i, j in SKELETON:
            pi = P @ np.append(kps[i], 1.0)
            pj = P @ np.append(kps[j], 1.0)

            if pi[2] <= 1e-6 or pj[2] <= 1e-6:
                continue

            xi = int(pi[0] / pi[2])
            yi = int(pi[1] / pi[2])
            xj = int(pj[0] / pj[2])
            yj = int(pj[1] / pj[2])

            h, w = img.shape[:2]
            if 0 <= xi < w and 0 <= yi < h and 0 <= xj < w and 0 <= yj < h:
                cv2.line(img, (xi, yi), (xj, yj), (255, 0, 0), 2)

        out = os.path.join(output_dir, f"reproj_frame{frame_idx:06d}_{cam_name}.jpg")
        cv2.imwrite(out, img)


# =========================
# PROCESS ONE TEST
# =========================
def process_test(test_info, alg_model, vol_model, global_alg_csv_path, global_vol_csv_path):
    output_dir = os.path.join(OUTPUT_ROOT, test_info["test_name"])

    alg_csv_path = os.path.join(output_dir, "keypoints3d_alg.csv")
    vol_csv_path = os.path.join(output_dir, "keypoints3d_vol.csv")

    vis3d_alg_dir = os.path.join(output_dir, "vis3d_alg")
    vis3d_vol_dir = os.path.join(output_dir, "vis3d_vol")

    reproj_alg_dir = os.path.join(output_dir, "reprojection_alg")
    reproj_vol_dir = os.path.join(output_dir, "reprojection_vol")

    ensure_dir(output_dir)
    ensure_dir(vis3d_alg_dir)
    ensure_dir(vis3d_vol_dir)
    ensure_dir(reproj_alg_dir)
    ensure_dir(reproj_vol_dir)

    cameras_params = load_cameras_for_test(test_info["xcp_path"])

    bbox_provider = CSVBBoxProvider(
        csv_paths=test_info["bbox_csv_paths"],
        deviceid_by_name=DEVICEID_BY_NAME,
        bbox_scale=BBOX_SCALE,
        person_id=CSV_PERSON_ID,
        csv_camera_key_by_name=CSV_CAMERA_KEY_BY_NAME,
    )

    caps, frame_counts = open_video_caps(test_info["video_by_name"])
    max_frames = min(frame_counts.values())

    print("\n==============================")
    print(f"TEST: {test_info['test_name']}")
    print(f"XCP: {test_info['xcp_path']}")
    print(f"CSV bbox: {test_info['bbox_csv_paths']}")
    print(f"Frames traitées: {max_frames}")

    ok_count = 0
    err_count = 0

    for frame_idx in range(1, max_frames + 1):
        print(f"\n── {test_info['test_name']} | Frame {frame_idx}/{max_frames} ──")

        frames_bgr = {}
        for cam_name in CAMERA_NAMES:
            ok, frame_bgr = caps[cam_name].read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Frame manquante pour {cam_name} à {frame_idx}")
            frames_bgr[cam_name] = frame_bgr

        try:
            alg_kps, vol_kps, vol_base_point = run_frame_alg_vol(
                frames_bgr=frames_bgr,
                frame_idx=frame_idx,
                bbox_provider=bbox_provider,
                cameras_params=cameras_params,
                alg_model=alg_model,
                vol_model=vol_model,
            )

            append_keypoints3d_csv(alg_csv_path, frame_idx, alg_kps)
            append_keypoints3d_csv(vol_csv_path, frame_idx, vol_kps)

            append_global_keypoints3d_csv(global_alg_csv_path, test_info["test_name"], frame_idx, alg_kps)
            append_global_keypoints3d_csv(global_vol_csv_path, test_info["test_name"], frame_idx, vol_kps)

            np.save(os.path.join(output_dir, f"kps3d_{frame_idx:06d}_alg.npy"), alg_kps)
            np.save(os.path.join(output_dir, f"kps3d_{frame_idx:06d}_vol.npy"), vol_kps)

            visualize_3d(alg_kps, frame_idx, vis3d_alg_dir, title_prefix="Pose 3D ALG")
            visualize_3d(vol_kps, frame_idx, vis3d_vol_dir, title_prefix="Pose 3D VOL")

            visualize_reprojection(alg_kps, frame_idx, frames_bgr, cameras_params, reproj_alg_dir)
            visualize_reprojection(vol_kps, frame_idx, frames_bgr, cameras_params, reproj_vol_dir)

            diff = mean_joint_diff(alg_kps, vol_kps)
            alg_spread = spread_from_pelvis(alg_kps)
            vol_spread = spread_from_pelvis(vol_kps)

            print("ALG pelvis:", alg_kps[6].round(1))
            print("VOL pelvis:", vol_kps[6].round(1))
            print("VOL base point:", vol_base_point.round(1))
            print(f"||base_point - alg_pelvis|| = {np.linalg.norm(vol_base_point - alg_kps[6]):.2f} mm")
            print(f"mean joint diff ALG/VOL = {diff:.2f} mm")
            print("ALG spread from pelvis:", alg_spread.round(1))
            print("VOL spread from pelvis:", vol_spread.round(1))

            ok_count += 1

        except Exception as e:
            print(f"Erreur frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            err_count += 1

    for cap in caps.values():
        cap.release()

    print(
        f"\nTerminé {test_info['test_name']} -> "
        f"{ok_count} frames OK, {err_count} erreurs, résultats dans '{output_dir}'"
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    ensure_dir(OUTPUT_ROOT)

    tests = discover_tests()
    if not tests:
        raise RuntimeError("Aucun test exploitable trouvé.")

    print("\nTests détectés :")
    for test in tests:
        print(
            f"- {test['test_name']} | "
            f"{len(test['bbox_csv_paths'])} CSV bbox | "
            f"{len(test['video_by_name'])} vidéos"
        )

    alg_model = load_alg_model()
    vol_model = load_vol_model()

    global_alg_csv_path = os.path.join(OUTPUT_ROOT, "all_tests_keypoints3d_alg.csv")
    global_vol_csv_path = os.path.join(OUTPUT_ROOT, "all_tests_keypoints3d_vol.csv")

    for test_info in tests:
        process_test(
            test_info=test_info,
            alg_model=alg_model,
            vol_model=vol_model,
            global_alg_csv_path=global_alg_csv_path,
            global_vol_csv_path=global_vol_csv_path,
        )

    print(f"\nTous les tests sont terminés -> résultats dans '{OUTPUT_ROOT}'")