# run_xcp_inference.py
import os
import sys
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from easydict import EasyDict
import torchvision.transforms as transforms
from ultralytics import YOLO

from mvn.models.triangulation import VolumetricTriangulationNet, AlgebraicTriangulationNet
from mvn.utils.multiview import Camera
from xcp_utils import load_cameras_params_from_xcp


# =========================
# CONFIG
# =========================
OUTPUT_DIR = "output_xcp_allframes_yolo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (384, 384)
N_JOINTS = 17

CAMERA_NAMES = ["camera1L", "camera1R", "camera2L", "camera2R"]

VIDEO_BY_NAME = {
    "camera1L": r"\\intram\paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT12_undistort\Session1\Files\fichiers_video\Bruts\METASPORT12_undistorted\CMJump_01.51433950.20251218140238.mp4",
    "camera1R": r"\\intram\paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT12_undistort\Session1\Files\fichiers_video\Bruts\METASPORT12_undistorted\CMJump_01.51434015.20251218140238.mp4",
    "camera2L": r"\\intram\paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT12_undistort\Session1\Files\fichiers_video\Bruts\METASPORT12_undistorted\CMJump_01.51470146.20251218140238.mp4",
    "camera2R": r"\\intram\paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT12_undistort\Session1\Files\fichiers_video\Bruts\METASPORT12_undistorted\CMJump_01.51470940.20251218140238.mp4",
}

XCP_PATH = r"\\intram\paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT12_undistort\Session1\Files\fichiers_video\Bruts\METASPORT12_undistorted\CMJump_01.xcp"

DEVICEID_BY_NAME = {
    "camera1L": "CAM03",
    "camera1R": "CAM04",
    "camera2L": "CAM05",
    "camera2R": "CAM06",
}

XCP_VARIANT = "2B"
T_SCALE = 1.0

YOLO_WEIGHTS = "./runs/detect/train/weights/best.pt"
YOLO_TRACKER = "bytetrack.yaml"
YOLO_CONF = 0.20
YOLO_IOU = 0.50
YOLO_CLASS = 0
BBOX_SCALE = 1.25
SMOOTH_ALPHA = 0.7

WEIGHTS_PATH = (
    "data/pretrained/human36m/"
    "human36m_vol_softmax_10-08-2019/"
    "checkpoints/0040/weights.pth"
)

ALG_WEIGHTS = (
    "data/pretrained/human36m/"
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
# YOLO BBOX TRACKER
# =========================
def clip_bbox(bbox, img_shape):
    top, left, bottom, right = bbox
    h, w = img_shape[:2]
    top = max(0, min(h - 1, int(top)))
    left = max(0, min(w - 1, int(left)))
    bottom = max(top + 1, min(h, int(bottom)))
    right = max(left + 1, min(w, int(right)))
    return [top, left, bottom, right]


def expand_square_bbox(bbox, img_shape, scale=1.25):
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


def smooth_bbox(prev_bbox, curr_bbox, alpha=0.7):
    if prev_bbox is None:
        return curr_bbox
    return [
        int(round(alpha * p + (1.0 - alpha) * c))
        for p, c in zip(prev_bbox, curr_bbox)
    ]


class YoloTrackBBoxProvider:
    def __init__(self, camera_names):
        self.models = {cam: YOLO(YOLO_WEIGHTS) for cam in camera_names}
        self.last_bbox = {cam: None for cam in camera_names}
        self.last_track_id = {cam: None for cam in camera_names}

    def _choose_detection(self, cam_name, boxes_xyxy, confs, track_ids, clss):
        if len(boxes_xyxy) == 0:
            return None

        keep = np.ones(len(boxes_xyxy), dtype=bool)
        if clss is not None:
            keep = (clss == YOLO_CLASS)

        idxs = np.where(keep)[0]
        if len(idxs) == 0:
            return None

        prev_id = self.last_track_id[cam_name]
        if prev_id is not None and track_ids is not None:
            same = [i for i in idxs if track_ids[i] == prev_id]
            if len(same) > 0:
                return max(same, key=lambda i: confs[i])

        return max(idxs, key=lambda i: confs[i])

    def get_bbox(self, cam_name, img_bgr):
        model = self.models[cam_name]

        results = model.track(
            source=img_bgr,
            persist=True,
            tracker=YOLO_TRACKER,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            classes=[YOLO_CLASS],
            verbose=False
        )

        if results is None or len(results) == 0:
            return self.last_bbox[cam_name]

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return self.last_bbox[cam_name]

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else None
        track_ids = result.boxes.id.int().cpu().numpy() if result.boxes.id is not None else None

        best_idx = self._choose_detection(cam_name, boxes_xyxy, confs, track_ids, clss)
        if best_idx is None:
            return self.last_bbox[cam_name]

        x1, y1, x2, y2 = boxes_xyxy[best_idx]
        bbox = [int(y1), int(x1), int(y2), int(x2)]
        bbox = expand_square_bbox(bbox, img_bgr.shape, scale=BBOX_SCALE)
        bbox = smooth_bbox(self.last_bbox[cam_name], bbox, alpha=SMOOTH_ALPHA)
        bbox = clip_bbox(bbox, img_bgr.shape)

        self.last_bbox[cam_name] = bbox
        if track_ids is not None:
            self.last_track_id[cam_name] = int(track_ids[best_idx])

        return bbox


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

    model = AlgebraicTriangulationNet(config, device=DEVICE)
    checkpoint = torch.load(ALG_WEIGHTS, map_location=DEVICE)
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
        cap = cv2.VideoCapture(VIDEO_BY_NAME[cam_name])
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir {VIDEO_BY_NAME[cam_name]}")
        caps[cam_name] = cap
        frame_counts[cam_name] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return caps, frame_counts


# =========================
# INFERENCE
# =========================
def run_frame_alg_only(alg_model, frames_bgr, frame_idx, bbox_provider):
    images_list = []
    proj_list = []

    for cam_idx, cam_name in enumerate(CAMERA_NAMES):
        cam_params = CAMERAS_PARAMS[cam_idx]
        img_bgr = frames_bgr[cam_name]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        bbox = bbox_provider.get_bbox(cam_name, img_bgr)
        if bbox is None:
            raise RuntimeError(f"Aucune bbox YOLO pour {cam_name} frame {frame_idx}")

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

    return alg_kps


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
            "b-", linewidth=2
        )

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
                cv2.putText(img, str(j), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1, cv2.LINE_AA)

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

    alg_model = load_alg_model()
    bbox_provider = YoloTrackBBoxProvider(CAMERA_NAMES)
    caps, frame_counts = open_video_caps()

    max_frames = min(frame_counts.values())
    print("Nombre de frames traité :", max_frames)

    for frame_idx in range(1, max_frames + 1):
        print(f"\n── Frame {frame_idx}/{max_frames} ──")

        frames_bgr = {}
        for cam_name in CAMERA_NAMES:
            ok, frame_bgr = caps[cam_name].read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Frame manquante pour {cam_name} à {frame_idx}")
            frames_bgr[cam_name] = frame_bgr

        try:
            kps = run_frame_alg_only(alg_model, frames_bgr, frame_idx, bbox_provider)
            np.save(os.path.join(OUTPUT_DIR, f"kps3d_{frame_idx:06d}.npy"), kps)
            visualize_3d(kps, frame_idx)
            visualize_reprojection(kps, frame_idx, frames_bgr)
            print("shape:", kps.shape, "| pelvis:", kps[6].round(1))
        except Exception as e:
            print(f"Erreur frame {frame_idx}: {e}")

    for cap in caps.values():
        cap.release()

    print(f"\nTerminé -> résultats dans '{OUTPUT_DIR}'")