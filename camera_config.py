import os
import scipy.io as sio
import numpy as np


def load_cameras_params(base_dir, subject):
    calib_path = os.path.join(
        base_dir,
        subject,
        'processed',
        'calibration',
        'camera_params.mat'
    )

    mat = sio.loadmat(calib_path)

    def get_cam(key_K, key_R, key_t, key_dist, name):
        K = mat[key_K].astype(np.float64)
        R = mat[key_R].astype(np.float64)
        t = mat[key_t].astype(np.float64)
        dist = mat[key_dist].astype(np.float64).flatten()

        return {
            'name': name,
            'K': K,
            'R': R,
            't': t,
            'dist': dist
        }

    cameras_params = [
        get_cam('K1l', 'R1l', 't1l', 'dist1l', 'camera1L'),
        get_cam('K1r', 'R1r', 't1r', 'dist1r', 'camera1R'),
        get_cam('K2l', 'R2l', 't2l', 'dist2l', 'camera2L'),
        get_cam('K2r', 'R2r', 't2r', 'dist2r', 'camera2R'),
    ]

    return cameras_params


if __name__ == '__main__':
    BASE_DIR = r'\\intram\paris\Labo\IBHGC\PROJETS\ecomocap\3_Data\001_ENSAMPOSE_WALK'
    SUBJECT = 'S01'

    cameras = load_cameras_params(BASE_DIR, SUBJECT)

    for cam in cameras:
        det = np.linalg.det(cam['R'])
        print(f"\n{cam['name']}")
        print(f"  K  shape : {cam['K'].shape}")
        print(f"  R  shape : {cam['R'].shape}, det={det:.4f}")
        print(f"  t  shape : {cam['t'].shape}, vals={cam['t'].flatten()}")
        print(f"  dist     : {cam['dist']}")