# run_inference.py
import os
import sys
import torch
import numpy as np
import cv2
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mvn.models.triangulation import VolumetricTriangulationNet
from easydict import EasyDict
import torchvision.transforms as transforms

from camera_config import CAMERAS_PARAMS

# CONFIG — adapte ces chemins
BASE_DIR     = r'\\intram\paris\Labo\IBHGC\PROJETS\ecomocap\3_Data\001_ENSAMPOSE_WALK'
SUBJECT      = 'S01'
ACTION       = 'walk'
IMSEQ        = 'imgSeq_walk01'

SUBJECT_DIR  = os.path.join(BASE_DIR, SUBJECT, 'processed', 'action', ACTION, IMSEQ)
WEIGHTS_PATH = 'data/pretrained/human36m/' \
               'human36m_vol_softmax_10-08-2019/' \
               'checkpoints/0040/weights.pth'
OUTPUT_DIR   = 'output_inference'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE   = (384, 384)
N_JOINTS     = 17

CAMERA_DIRS  = ['camera1L', 'camera1R', 'camera2L', 'camera2R']

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std= [0.229, 0.224, 0.225]
)

SKELETON = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),
    (8,14),(14,15),(15,16)
]

# CHARGER LES BBOXES DU JSON
def load_bboxes():
    json_path = os.path.join(SUBJECT_DIR, 'annot.json')
    with open(json_path) as f:
        data = json.load(f)
    return data['bboxes']

BBOXES = load_bboxes()

def get_bbox(cam_name, frame_idx):
    try:
        b      = BBOXES[cam_name]
        top    = int(b['top']   [frame_idx])
        left   = int(b['left']  [frame_idx])
        bottom = int(b['bottom'][frame_idx])
        right  = int(b['right'] [frame_idx])
        if bottom > top and right > left:
            return [top, left, bottom, right]
    except (KeyError, IndexError):
        pass
    return None

# CONSTRUIRE LA MATRICE P
def build_proj_matrix(cam_params, bbox=None, orig_size=None):
    K = cam_params['K'].copy()
    R = cam_params['R']
    t = cam_params['t']

    if bbox is not None and orig_size is not None:
        top, left, bottom, right = bbox
        orig_h, orig_w = orig_size
        crop_h = bottom - top
        crop_w = right  - left

        # Mettre à jour K pour le crop
        K[0, 2] -= left
        K[1, 2] -= top

        # Mettre à jour K pour le resize
        K[0, 0] *= IMAGE_SIZE[1] / crop_w   # fx
        K[0, 2] *= IMAGE_SIZE[1] / crop_w   # cx
        K[1, 1] *= IMAGE_SIZE[0] / crop_h   # fy
        K[1, 2] *= IMAGE_SIZE[0] / crop_h   # cy

    Rt = np.hstack([R, t])  # (3,4)
    P  = K @ Rt              # (3,4)
    return P

# CHARGER LE MODÈLE
def load_model():
    config = EasyDict({
        'model': {
            'name': 'vol',
            'backbone': {
                'name':               'pose_resnet',
                'num_layers':         152,
                'num_deconv_layers':  3,
                'num_deconv_filters': [256, 256, 256],
                'num_deconv_kernels': [4, 4, 4],
                'num_joints':         N_JOINTS,
                'image_size':         [384, 384],
                'heatmap_size':       [96, 96],
                'init_weights':       False,
            },
            'num_joints':            N_JOINTS,
            'volume_size':           64,
            'volume_multiplier':     1.0,
            'volume_softmax':        True,
            'triangulation_softmax': True,
            'use_gt_pelvis':         False,
            'pelvis_type':           'pred',
            'cuboid_side':           2500.0,
        }
    })

    print(f"Chargement du modèle sur {DEVICE}...")
    model = VolumetricTriangulationNet(config, device=DEVICE)

    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    print(" Modèle chargé")
    return model

# INFÉRENCE SUR UNE FRAME
def run_frame(model, frame_idx):
    images_list = []
    proj_list   = []

    for cam_idx, cam_name in enumerate(CAMERA_DIRS):
        cam_params = CAMERAS_PARAMS[cam_idx]
        img_path =os.path.join(SUBJECT_DIR, cam_name, f'img_{frame_idx:04d}.jpg')
        img      = cv2.imread(img_path)
        if img is None:
            raise ValueError(f" Image illisible : {img_path}")
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Récupérer la bbox
        bbox = get_bbox(cam_name, frame_idx)
        if bbox is None:
            # Pas de bbox -> image complète
            bbox = [0, 0, orig_h, orig_w]
            print(f"  Pas de bbox pour {cam_name} "
                  f"frame {frame_idx} → image complète")

        # Crop
        top, left, bottom, right = bbox
        img_crop = img[top:bottom, left:right]

        # Resize + normalize
        img_resized = cv2.resize(img_crop, IMAGE_SIZE)
        tensor = torch.from_numpy(
            img_resized
        ).permute(2, 0, 1).float() / 255.0
        tensor = NORMALIZE(tensor)

        # Matrice de projection
        P = build_proj_matrix(cam_params, bbox, (orig_h, orig_w))

        images_list.append(tensor)
        proj_list.append(torch.from_numpy(P).float())

    # Assembler le batch (taille 1)
    images        = torch.stack(images_list).unsqueeze(0).to(DEVICE)
    proj_matrices = torch.stack(proj_list).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(images, proj_matrices, None)

    keypoints_3d = output[0] if isinstance(output, tuple) else output
    return keypoints_3d[0].cpu().numpy()  # [17, 3]

# VISUALISATION 3D
def visualize_3d(kps, frame_idx):
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(kps[:,0], kps[:,1], kps[:,2],
               c='red', s=60, zorder=5)

    for i, j in SKELETON:
        ax.plot(
            [kps[i,0], kps[j,0]],
            [kps[i,1], kps[j,1]],
            [kps[i,2], kps[j,2]],
            'b-', linewidth=2
        )

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Pose 3D — frame {frame_idx}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f'pose3d_{frame_idx:06d}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  3D sauvegardé : {path}")

# REPROJECTION 2D SUR LES IMAGES
def visualize_reprojection(kps, frame_idx):
    for cam_idx, cam_name in enumerate(CAMERA_DIRS):
        cam_params = CAMERAS_PARAMS[cam_idx]

        try:
            img_path =os.path.join(SUBJECT_DIR, cam_name, f'img_{frame_idx:04d}.jpg')
        except FileNotFoundError:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Projection sur image originale (sans crop)
        P = build_proj_matrix(cam_params)

        # Dessiner les joints
        for j in range(N_JOINTS):
            pt   = np.append(kps[j], 1.0)
            proj = P @ pt
            x    = int(proj[0] / proj[2])
            y    = int(proj[1] / proj[2])
            h, w = img.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

        # Dessiner le squelette
        for i, j in SKELETON:
            pt_i = P @ np.append(kps[i], 1.0)
            pt_j = P @ np.append(kps[j], 1.0)
            xi = int(pt_i[0] / pt_i[2])
            yi = int(pt_i[1] / pt_i[2])
            xj = int(pt_j[0] / pt_j[2])
            yj = int(pt_j[1] / pt_j[2])
            h, w = img.shape[:2]
            if (0<=xi<w and 0<=yi<h and 0<=xj<w and 0<=yj<h):
                cv2.line(img, (xi,yi), (xj,yj), (255,0,0), 2)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(
            OUTPUT_DIR,
            f'reproj_frame{frame_idx:06d}_{cam_name}.jpg'
        )
        cv2.imwrite(out, img)
        print(f" Reprojection : {out}")

#main
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()

    # Frames à tester  indices à adapter 
    frame_indices = [0, 10, 20, 30, 40]

    print(f"\nTraitement de {len(frame_indices)} frames...")

    for frame_idx in frame_indices:
        print(f"\n── Frame {frame_idx} ──")
        try:
            kps = run_frame(model, frame_idx)
            print(f"  Keypoints shape : {kps.shape}")
            print(f"  Pelvis (joint 0): {kps[0].round(1)} mm")
            visualize_3d(kps, frame_idx)
            visualize_reprojection(kps, frame_idx)
        except Exception as e:
            print(f"  Erreur : {e}")
            import traceback
            traceback.print_exc()

    print(f"\n Terminé → résultats dans '{OUTPUT_DIR}/'")
