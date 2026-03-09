# camera_config.py
import scipy.io as sio
import numpy as np


CALIB_PATH = r'\\intram\paris\Labo\IBHGC\PROJETS\ecomocap\3_Data\001_ENSAMPOSE_WALK\S01\processed\calibration\camera_params.mat'

mat = sio.loadmat(CALIB_PATH)

def get_cam(key_K, key_R, key_t, key_dist, name):
    K    = mat[key_K].astype(np.float64)          # (3,3)
    R    = mat[key_R].astype(np.float64)          # (3,3)
    t    = mat[key_t].astype(np.float64)          # (3,1)
    dist = mat[key_dist].astype(np.float64).flatten()  # (5,)
    
    return {
        'name': name,
        'K':    K,
        'R':    R,
        't':    t,
        'dist': dist
    }

CAMERAS_PARAMS = [
    get_cam('K1l',  'R1l',  't1l',  'dist1l',  'camera1L'),
    get_cam('K1r',  'R1r',  't1r',  'dist1r',  'camera1R'),
    get_cam('K2l',  'R2l',  't2l',  'dist2l',  'camera2L'),
    get_cam('K2r',  'R2r',  't2r',  'dist2r',  'camera2R'),
]

# Vérification
if __name__ == '__main__':
    for cam in CAMERAS_PARAMS:
        det = np.linalg.det(cam['R'])
        print(f"\n{cam['name']}")
        print(f"  K  shape : {cam['K'].shape}")
        print(f"  R  shape : {cam['R'].shape}, det={det:.4f}")
        print(f"  t  shape : {cam['t'].shape}, vals={cam['t'].flatten()}")
        print(f"  dist     : {cam['dist']}")