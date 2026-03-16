# run_xcp_inference.py
import os
import sys
import csv
import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from easydict import EasyDict
import torchvision.transforms as transforms

from mvn.models.triangulation import AlgebraicTriangulationNet
from xcp_utils import load_cameras_params_from_xcp


# =========================
# CONFIG
# =========================
OUTPUT_DIR = "output_xcp_allframes"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (384, 384)
N_JOINTS = 17

CAMERA_NAMES = ["camera1L", "camera1R", "camera2L", "camera2R"]

VIDEO_BY_NAME = {
    "camera1L": "./metasport/DATA/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts/METASPORT12_undistorted/CMJump_01.51433950.20251218140238.mp4",
    "camera1R": "./metasport/DATA/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts/METASPORT12_undistorted/CMJump_01.51434015.20251218140238.mp4",
    "camera2L": "./metasport/DATA/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts/METASPORT12_undistorted/CMJump_01.51470146.20251218140238.mp4",
    "camera2R": "./metasport/DATA/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts/METASPORT12_undistorted/CMJump_01.51470940.20251218140238.mp4",
}

XCP_PATH = "./metasport/DATA/METASPORT12_undistort/Session1/Files/fichiers_video/Bruts/METASPORT12_undistorted/CMJump_01.xcp"

DEVICEID_BY_NAME = {
    "camera1L": "51433950",
    "camera1R": "51434015",
    "camera2L": "51470146",
    "camera2R": "51470940",
}

# Dans le CSV, camera_id = 0,1,2,3
CSV_CAMERA_KEY_BY_NAME = {
    "camera1L": "0",
    "camera1R": "1",
    "camera2L": "2",
    "camera2R": "3",
}

CSV_BBOX_PATH = "./metasport/DATA/METASPORT12_undistort/Session1/Files/fichiers_video/Traites/RtmLib/METASPORT12_tracked/CMJump_01_51433950_training_data.csv"

# Si le CSV contient plusieurs personnes, mets l'ID voulu ici, sinon None
CSV_PERSON_ID = None

XCP_VARIANT = "2B"
T_SCALE = 1.0
BBOX_SCALE = 1.25  # important : augmenter la bbox pour que le modèle alg puisse voir tous les joints

ALG_WEIGHTS = (
    "./data/pretrained/human36m/"
    "human36m_alg_10-04-2019/"
    "checkpoints/0060/weights.pth"
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
# CAMERAS
# =========================
CAMERAS_PARAMS = load_cameras_params_from_xcp(
    xcp_path=XCP_PATH,
    deviceid_by_name=DEVICEID_BY_NAME,
    camera_names=CAMERA_NAMES,
    variant=XCP_VARIANT,
    t_scale=T_SCALE
)


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
        csv_path,
        deviceid_by_name,
        bbox_scale=1.0,
        person_id=None,
        csv_camera_key_by_name=None,
    ):
        self.csv_path = csv_path
        self.deviceid_by_name = {
            k: normalize_camera_key(v) for k, v in deviceid_by_name.items()
        }
        self.csv_camera_key_by_name = {
            k: normalize_camera_key(v) for k, v in (csv_camera_key_by_name or {}).items()
        }
        self.bbox_scale = bbox_scale
        self.person_id = person_id
        self.data = {}  # (cam_key, frame_idx_1based) -> bbox [top,left,bottom,right]
        self.frame_offset = 0
        self._load()

    def _load(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV bbox introuvable: {self.csv_path}")

        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if not rows:
            raise ValueError(f"CSV bbox vide: {self.csv_path}")

        print("\n=== CSV BBOX ===")
        print("Colonnes détectées :", fieldnames)

        cam_col = find_col(
            fieldnames,
            ["camera", "cam_name", "camera_name", "cam_id", "camera_id", "device_id"]
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

        frame_values = []
        for row in rows:
            try:
                frame_values.append(int(round(float(row[frame_col]))))
            except Exception:
                pass

        if not frame_values:
            raise ValueError("Impossible de lire les indices de frame dans le CSV.")

        min_frame = min(frame_values)
        self.frame_offset = 0 if min_frame == 1 else 1

        duplicate_count = 0
        kept_count = 0

        for row in rows:
            try:
                if self.person_id is not None and person_col is not None:
                    try:
                        pid = int(float(row[person_col]))
                        if pid != self.person_id:
                            continue
                    except Exception:
                        continue

                cam_key = normalize_camera_key(row[cam_col])
                frame_raw = int(round(float(row[frame_col])))
                frame_idx_1based = frame_raw + self.frame_offset

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

        print(f"CSV bbox chargé: {len(self.data)} entrées uniques")
        print(f"Lignes gardées: {kept_count}")
        print(f"Décalage de frame appliqué: {self.frame_offset}")
        if duplicate_count > 0:
            print(f"Attention: {duplicate_count} doublons cam/frame trouvés, la dernière bbox a été gardée.")

        camera_keys_in_csv = sorted({k[0] for k in self.data.keys()})
        print("Clés caméra trouvées dans le CSV :", camera_keys_in_csv)

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
def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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

    model = AlgebraicTriangulationNet(config, device=DEVICE)
    checkpoint = safe_torch_load(ALG_WEIGHTS, map_location=DEVICE)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    return model


# =========================
# VIDEO IO
# =========================
def open_video_caps():
    caps = {}
    frame_counts = {}

    for cam_name in CAMERA_NAMES:
        path = VIDEO_BY_NAME[cam_name]
        if not os.path.exists(path):
            raise RuntimeError(f"Chemin introuvable pour {cam_name}: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir {path}")

        caps[cam_name] = cap
        frame_counts[cam_name] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return caps, frame_counts


def print_video_and_camera_debug():
    print("\n=== VIDEOS ===")
    for cam_name in CAMERA_NAMES:
        cap = cv2.VideoCapture(VIDEO_BY_NAME[cam_name])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"{cam_name}: {w}x{h}, {n} frames")

    print("\n=== CAMERAS ===")
    for cam in CAMERAS_PARAMS:
        print(f"\n{cam['name']}")
        print("K =")
        print(cam["K"])
        print("det(R) =", np.linalg.det(cam["R"]))
        print("t =", cam["t"].reshape(-1))


# =========================
# INFERENCE
# =========================
def run_frame_alg_only(alg_model, frames_bgr, frame_idx, bbox_provider):
    images_list = []
    proj_list = []
    bboxes_used = {}

    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        cam_params = CAMERAS_PARAMS[cam_idx]
        img_bgr = frames_bgr[cam_name]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        bbox = bbox_provider.get_bbox(cam_name, frame_idx, img_bgr.shape)
        bboxes_used[cam_name] = bbox

        top, left, bottom, right = bbox
        img_crop = img_rgb[top:bottom, left:right]

        if img_crop.size == 0:
            raise RuntimeError(f"Crop vide pour {cam_name} frame {frame_idx}, bbox={bbox}")

        img_resized = cv2.resize(img_crop, IMAGE_SIZE)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        tensor = NORMALIZE(tensor)

        P = build_proj_matrix(cam_params, bbox, (orig_h, orig_w))

        images_list.append(tensor)
        proj_list.append(torch.from_numpy(P).float())

    images = torch.stack(images_list).unsqueeze(0).to(DEVICE)
    proj_matrices = torch.stack(proj_list).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        alg_output = alg_model(images, proj_matrices, None)
        alg_kps = alg_output[0][0].cpu().numpy()

    return alg_kps, bboxes_used


# =========================
# VISU
# =========================
def visualize_3d(kps, frame_idx):
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
    ax.set_title(f"Pose 3D — frame {frame_idx}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"pose3d_{frame_idx:06d}.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()


def visualize_reprojection(kps, frame_idx, frames_bgr):
    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        img = frames_bgr[cam_name].copy()
        cam_params = CAMERAS_PARAMS[cam_idx]
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

        out = os.path.join(OUTPUT_DIR, f"reproj_frame{frame_idx:06d}_{cam_name}.jpg")
        cv2.imwrite(out, img)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_video_and_camera_debug()

    alg_model = load_alg_model()
    bbox_provider = CSVBBoxProvider(
        csv_path=CSV_BBOX_PATH,
        deviceid_by_name=DEVICEID_BY_NAME,
        bbox_scale=BBOX_SCALE,
        person_id=CSV_PERSON_ID,
        csv_camera_key_by_name=CSV_CAMERA_KEY_BY_NAME,
    )
    caps, frame_counts = open_video_caps()

    max_frames = min(frame_counts.values())
    print("\nNombre de frames traitées :", max_frames)

    for frame_idx in range(1, max_frames + 1):
        print(f"\n── Frame {frame_idx}/{max_frames} ──")

        frames_bgr = {}
        for cam_name in CAMERA_NAMES:
            ok, frame_bgr = caps[cam_name].read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Frame manquante pour {cam_name} à {frame_idx}")
            frames_bgr[cam_name] = frame_bgr

        try:
            kps, _ = run_frame_alg_only(
                alg_model=alg_model,
                frames_bgr=frames_bgr,
                frame_idx=frame_idx,
                bbox_provider=bbox_provider,
            )

            np.save(os.path.join(OUTPUT_DIR, f"kps3d_{frame_idx:06d}.npy"), kps)
            visualize_3d(kps, frame_idx)
            visualize_reprojection(kps, frame_idx, frames_bgr)

            print("shape:", kps.shape, "| pelvis:", kps[6].round(1))

        except Exception as e:
            print(f"Erreur frame {frame_idx}: {e}")

    for cap in caps.values():
        cap.release()

    print(f"\nTerminé -> résultats dans '{OUTPUT_DIR}'")