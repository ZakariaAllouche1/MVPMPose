#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tracker multi-personnes — Contrôle de Cohérence Spatio-Temporelle
  Après l'association, le système vérifie pour chaque ID si toutes
  ses instances (détections actuelles + positions prédites) sont spatialement
  cohérentes (en  3D). Si un ID est détecté à deux
  endroits trop éloignés simultanément (un "split ID"), la correspondance la moins
  fiable est annulée. "SPATIAL_CONSISTENCY_THRESH_MM" a été
  ajouté pour définir la distance maximale tolérée entre deux instances d'un même ID.

"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import scipy.linalg
from ultralytics import YOLO
from collections import deque, defaultdict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as R_scipy
from tqdm import tqdm
import onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET
import csv
from datetime import datetime
from enum import Enum
from rtmlib import BodyWithFeet
import matplotlib.pyplot as plt
from pathlib import Path

print("NOTE: TensorFlow n'est plus une dépendance.")

# Constantes
chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}
INFTY_COST = 1e+5


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def _make_spd(M: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """Rend M symétrique définie positive en relevant les vp négatives."""
    M = _symmetrize(M)
    try:
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, min_eig)
        return (V * w) @ V.T
    except np.linalg.LinAlgError:
        eps = max(min_eig, 1e-6)
        return _symmetrize(M + eps * np.eye(M.shape[0], dtype=M.dtype))

class TrackState(Enum):
    Tentative = 1
    Confirmed2D = 2
    Confirmed3D = 3
    Occluded = 4
    Lost = 5

class KalmanFilter(object):
    """ Filtre de Kalman simple pour bbox [cx,cy,a,h | vx,vy,va,vh] """
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 80

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3], 1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3], 1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3], 1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3], 1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        covariance = _make_spd(covariance, 1e-9)
        return mean, covariance

    def project(self, mean, covariance):
        covariance = _make_spd(covariance, 1e-9)
        std = [
            self._std_weight_position * max(1e-3, mean[3]),
            self._std_weight_position * max(1e-3, mean[3]),
            1e-1,
            self._std_weight_position * max(1e-3, mean[3])
        ]
        innovation_cov = np.diag(np.square(std))
        innovation_cov = _make_spd(innovation_cov, 1e-12)

        mean = np.dot(self._update_mat, mean)
        cov_proj = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        cov_proj = _make_spd(cov_proj, 1e-12)
        return mean, _make_spd(cov_proj + innovation_cov, 1e-12)

    def update(self, mean, covariance, measurement):
        if not np.all(np.isfinite(measurement)):
            return mean, covariance
        projected_mean, projected_cov = self.project(mean, covariance)
        projected_cov = _make_spd(projected_cov, 1e-9)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, 
            check_finite=False).T
            
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T))
            
        return new_mean, _make_spd(new_covariance, 1e-9)

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        covariance = _make_spd(covariance, 1e-9)
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        try:
            cholesky_factor = np.linalg.cholesky(covariance)
            d = measurements - mean
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            squared_maha = np.full(measurements.shape[0], INFTY_COST)
        return squared_maha


class KalmanFilter3D:
    """ KF pour position 3D (x,y,z,vx,vy,vz) """
    def __init__(self):
        ndim, dt = 3, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 25
        self._std_weight_velocity = 1. / 150

    def initiate(self, measurement_3d):
        mean_pos = measurement_3d
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [2 * self._std_weight_position * 1000] * 3 + [10 * self._std_weight_velocity * 1000] * 3
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position * 100] * 3
        std_vel = [self._std_weight_velocity * 100] * 3
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        covariance = _make_spd(covariance, 1e-9)
        return mean, covariance

    def project(self, mean, covariance):
        std = [self._std_weight_position * 200] * 3
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        covariance = _make_spd(covariance, 1e-12)
        innovation_cov = _make_spd(innovation_cov, 1e-12)
        return mean, _make_spd(covariance + innovation_cov, 1e-12)

    def update(self, mean, covariance, measurement_3d):
        proj_mean, proj_cov = self.project(mean, covariance)
        proj_cov = _make_spd(proj_cov, 1e-9)
        
        chol, lower = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol, lower), (covariance @ self._update_mat.T).T, check_finite=False).T
        innov = measurement_3d - proj_mean
        new_mean = mean + K @ innov
        new_cov = _make_spd(covariance - K @ proj_cov @ K.T, 1e-9)
        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements_3d):
        covariance = _make_spd(covariance, 1e-9)
        mean, covariance = self.project(mean, covariance)
        try:
            cholesky_factor = np.linalg.cholesky(covariance)
            d = measurements_3d - mean
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            squared_maha = np.full(measurements_3d.shape[0], INFTY_COST)
        return squared_maha


def bbox_to_kalman_format(bbox):
    x1, y1, x2, y2 = map(float, bbox)
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / h
    return np.array([cx, cy, a, h], dtype=float)

def kalman_to_bbox_format(kalman_mean):
    cx, cy, a, h = kalman_mean[:4]
    w = a * h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=float)


@dataclass
class GlobalTrack:
    global_id: int
    kf_mean: Optional[np.ndarray] = None
    kf_covariance: Optional[np.ndarray] = None
    kf_3d: KalmanFilter3D = field(default_factory=KalmanFilter3D)
    mean_3d: Optional[np.ndarray] = None
    covariance_3d: Optional[np.ndarray] = None
    last_seen_frame: int = 0
    first_seen_frame: int = 0
    feature_gallery: deque = field(default_factory=lambda: deque(maxlen=100))
    color: np.ndarray = field(default_factory=lambda: np.random.randint(0, 255, (3,), dtype=int))
    current_bbox: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    current_mask: Dict[int, np.ndarray] = field(default_factory=dict)
    current_pose_2d: Dict[int, np.ndarray] = field(default_factory=dict)
    pose_2d_confidence: Dict[int, float] = field(default_factory=dict)
    current_pose_2d_joint_confidences: Dict[int, np.ndarray] = field(default_factory=dict)
    current_pose_3d: Optional[np.ndarray] = None
    pose_3d_confidence: float = 0.0
    state: TrackState = TrackState.Tentative
    time_since_update: int = 0
    hits: int = 0
    age: int = 0
    last_known_3d_pos: Optional[np.ndarray] = None
    high_motion_timer: int = 0
    HIGH_MOTION_THRESHOLD_MM: float = 200.0
    HIGH_MOTION_DURATION: int = 10
    last_near_edge: bool = False
    jump_timer: int = 0
    occlusion_timer: int = 0
    visible_joints_streak: int = 0
    last_visible_ratio: float = 1.0
    aspect_ratio_history: deque = field(default_factory=lambda: deque(maxlen=5))
    is_occluded_by: Optional[int] = None
    position_3d_history: deque = field(default_factory=lambda: deque(maxlen=50))
    position_variance_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_triangulated_frame: int = -1
    single_view_streak: int = 0  # Compteur de frames consécutives avec une seule vue
    multi_view_history: deque = field(default_factory=lambda: deque(maxlen=30))  # Historique vue unique/multiple
    
    def update_3d_position_with_validation(self, new_pos_3d: np.ndarray):
        """Mise à jour 3D avec validation de cohérence"""
        if len(self.position_3d_history) > 3:
            recent_positions = np.array(list(self.position_3d_history)[-5:])
            predicted_pos = recent_positions[-1] + np.mean(np.diff(recent_positions, axis=0), axis=0)
            
            # Vérifier la cohérence avec la prédiction
            prediction_error = np.linalg.norm(new_pos_3d - predicted_pos)
            if prediction_error > 2000.0:  # Seuil de cohérence
                print(f" Track {self.global_id}: Position 3D incohérente (err={prediction_error:.0f}mm)")
                # Utiliser une position interpolée
                new_pos_3d = 0.7 * predicted_pos + 0.3 * new_pos_3d
        
        self.position_3d_history.append(new_pos_3d.copy())

    def update_features(self, new_features: np.ndarray, weight: float = 0.05):
        new_features = new_features.copy()
        n = np.linalg.norm(new_features)
        if n == 0: return
        new_features /= n
        
        if not self.feature_gallery:
            self.feature_gallery.append(new_features)
            return

        distances = cdist(np.array([new_features]), np.array(list(self.feature_gallery)), metric='cosine')[0]
        min_dist = np.min(distances)
        min_idx = np.argmin(distances)

        if min_dist < 0.15:
            self.feature_gallery[min_idx] = (1 - weight) * self.feature_gallery[min_idx] + weight * new_features
            self.feature_gallery[min_idx] /= np.linalg.norm(self.feature_gallery[min_idx])
        elif 0.15 <= min_dist < 0.5:
            self.feature_gallery.append(new_features)


        # Nettoyage périodique plus agressif
        if len(self.feature_gallery) > 40 and self.age % 200 == 0:  # Tous les ~7 secondes
            gallery_array = np.array(list(self.feature_gallery))
            
            # Calculer la cohérence interne
            distances = cdist(gallery_array, gallery_array, metric='cosine')
            mean_distances = np.mean(distances, axis=1)
            
            # Garder les 20 features les plus cohérentes + 5 récentes
            coherent_indices = np.argsort(mean_distances)[:20]
            recent_indices = list(range(max(0, len(self.feature_gallery)-5), len(self.feature_gallery)))
            
            keep_indices = set(list(coherent_indices) + recent_indices)
            new_gallery = deque(maxlen=100)
            for idx in sorted(keep_indices):
                new_gallery.append(self.feature_gallery[idx])
            self.feature_gallery = new_gallery
            
            print(f" Track {self.global_id}: Feature gallery cleaned ({len(keep_indices)} kept)")

    @property
    def is_in_high_motion(self) -> bool:
        return self.high_motion_timer > 0

    @property
    def is_in_high_uncertainty(self) -> bool:
        if self.is_in_high_motion or self.is_occluded_by is not None or self.state == TrackState.Occluded:
            return True
        if len(self.aspect_ratio_history) < 5:
            return False
        std_dev = np.std(list(self.aspect_ratio_history))
        mean_ar = np.mean(list(self.aspect_ratio_history))
        if mean_ar > 0 and (std_dev / mean_ar) > 0.15:
            return True
        return False

    def mark_missed(self, max_patience: int):
        self.age += 1
        self.time_since_update += 1
        if self.state not in [TrackState.Lost, TrackState.Occluded] and self.time_since_update > max_patience:
            self.state = TrackState.Lost

    def mark_hit(self, frame_idx):
        self.time_since_update = 0
        self.last_seen_frame = frame_idx
        self.hits += 1
        if self.state == TrackState.Tentative and self.hits >= 3:
            self.state = TrackState.Confirmed2D
        elif self.state == TrackState.Lost or self.state == TrackState.Occluded:
            self.state = TrackState.Confirmed2D if self.mean_3d is None else TrackState.Confirmed3D

    @property
    def has_3d(self) -> bool:
        return self.mean_3d is not None and self.state == TrackState.Confirmed3D


class FeatureExtraction:
    def __init__(self, onnx_path: str):
        self.ort_session = ort.InferenceSession(onnx_path)
        h, w = self.ort_session.get_inputs()[0].shape[2:4]
        self.augment = A.Compose([A.Resize(h, w), A.Normalize(), ToTensorV2()])

    def predict(self, img: np.ndarray) -> Optional[np.ndarray]:
        if img.size == 0:
            return None
        image = self.augment(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"]
        image = np.expand_dims(image, axis=0)
        feat = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: image})[0]
        return feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)


def parse_xcp(filepath):
    "Vicon radial2 parametre de distorition n'est pas utilisé. j'ai testé son utilisation ça n'a rien apporté de plus bien au contraire"
    tree = ET.parse(filepath)
    root = tree.getroot()
    cameras = {}
    for camera_node in root.findall("Camera"):
        if camera_node.get("ISDV") != "1": continue
        device_id = camera_node.get("DEVICEID")
        keyframe = camera_node.find("KeyFrames/KeyFrame")
        if keyframe is None: continue
        pos_str = keyframe.get("POSITION").split()
        t_world = np.array([float(p) for p in pos_str])
        orient_str = keyframe.get("ORIENTATION").split()
        q = [float(x) for x in orient_str]
        quat_variants = [[q[1], q[2], q[3], q[0]], q]
        focal_length = float(keyframe.get("FOCAL_LENGTH"))
        cx, cy = map(float, keyframe.get("PRINCIPAL_POINT").split())
        #if focal_length < 50: focal_length *= 200
        dist_str = keyframe.get("VICON_RADIAL2", "").split()
        dist_coeffs = np.zeros(5)
        vicon_dist = None
        if len(dist_str) >= 6 and dist_str[0].lower().startswith("vicon"):
            try:
                vicon_dist = {'cx_d': float(dist_str[1]),
                          'cy_d': float(dist_str[2]),
                          'k1': float(dist_str[3]),
                          'k2': float(dist_str[4]),
                          'k3': float(dist_str[5]),}
            except Exception:
                pass
        elif len(dist_str) >= 5:
            dist_coeffs[0], dist_coeffs[1], dist_coeffs[4] = float(dist_str[2]), float(dist_str[3]), float(dist_str[4])
        variants = {}
        for idx, quat in enumerate(quat_variants, 1):
            try:
                Rq = R_scipy.from_quat(quat).as_matrix()
                if abs(np.linalg.det(Rq) - 1) < 0.1:
                    variants[f"{idx}A"] = {"R": Rq.T, "t": -Rq.T @ t_world.reshape(3, 1)}
                    variants[f"{idx}B"] = {"R": Rq, "t": -Rq @ t_world.reshape(3, 1)}
            except Exception: pass
        K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float64)
        cameras[device_id] = {"variants": {k: {**rt, "K": K, "dist": dist_coeffs, "vicon_dist": vicon_dist} for k, rt in variants.items()}}
    return cameras


def create_projection_methods(cameras, device_ids):
    all_variants = set()
    for did in device_ids:
        if did in cameras: all_variants.update(cameras[did]["variants"].keys())
    projection_sets = {}
    for var in all_variants:
        projs, valid = [], True
        for did in device_ids:
            if did in cameras and var in cameras[did]["variants"]:
                cam = cameras[did]["variants"][var]
                projs.append(cam["K"] @ np.hstack([cam["R"], cam["t"]]))
            else:
                valid = False
                break
        if valid and len(projs) == len(device_ids): projection_sets[var] = projs
    return projection_sets


def weighted_triangulate_points(proj_matrices: List[np.ndarray], points_2d: List[np.ndarray], weights: List[float]) -> np.ndarray:
    A = np.zeros((2 * len(proj_matrices), 4))
    for i, (P, p2d, w) in enumerate(zip(proj_matrices, points_2d, weights)):
        A[2 * i] = w * (p2d[0] * P[2] - P[0])
        A[2 * i + 1] = w * (p2d[1] * P[2] - P[1])
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    return X[:3] / (X[3] + 1e-12)

def check_cheirality(p3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> bool:
    return (R @ p3d.reshape(3, 1) + t)[2] > 0

def build_undistort_maps(w, h, cx_d, cy_d, k1, k2, k3,
                         max_iter=10, tol=1e-6):
    """
    Calcule les maps de remap pour redresser la video.
    Methode iterative Vicon (confirmee par Vicon Support).
    Identique a project_markers_c3d.py.
    """
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float64),
                                 np.arange(h, dtype=np.float64))
    u_est = u_grid.copy()
    v_est = v_grid.copy()

    for it in range(max_iter):
        du = u_est - cx_d
        dv = v_est - cy_d
        r2 = du * du + dv * dv
        radial = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        u_new = u_grid - du * radial
        v_new = v_grid - dv * radial
        err = max(np.max(np.abs(u_new - u_est)),
                  np.max(np.abs(v_new - v_est)))
        u_est = u_new
        v_est = v_new
        if err < tol:
            break

    return u_est.astype(np.float32), v_est.astype(np.float32)


class RobustMultiViewTracker:
    def __init__(self, model_path: str, feature_model_path: str, rtmlib_device: str = 'cuda', rtmlib_backend: str = 'onnxruntime', save_res: Tuple[int, int] = (1280, 720),debug=True, **kwargs):
        self.params = {
            'confidence_thresh': 0.25, 'intra_frame_thresh': 0.3, 'min_views_for_3d': 2, 'pose_confidence_thresh': 0.3,
            'reprojection_error_thresh': 80, 'min_valid_joints': 6, 'min_confidence_for_new_track': 0.6,
            'track_patience': 200, 'iou_match_thresh': 0.4, 'association_thresh': 0.65,
            'min_joint_confidence_for_triangulation': 0.3, 'edge_margin_pixels': 40,
            'occlusion_patience': 100, 'reactivation_appearance_threshold': 0.35,
            'occlusion_xy_dist_thresh': 1000.0,
            'min_visibility_for_new_track': 0.5, 
            'low_visibility_threshold': 0.5,
            'occlusion_reid_threshold_mm': 1200.0,
            'max_occlusion_frames': 100,
            # NOUVEAU: Seuil pour le contrôle de cohérence
            'SPATIAL_CONSISTENCY_THRESH_MM': 1000.0
        }
        self.debug = debug
        #if self.debug:
            #self.debugger = TrackerDebugger()
        self.params.update(kwargs)
        self.model_path = model_path
        self.feature_extractor = FeatureExtraction(feature_model_path)
        self.pose_estimator = None
        if rtmlib_device == 'cuda' and 'CUDAExecutionProvider' not in ort.get_available_providers():
            print(" CUDA non disponible pour onnxruntime, passage au CPU.")
            self.rtmlib_device = 'cpu'
        else:
            self.rtmlib_device = rtmlib_device
        self.rtmlib_backend = rtmlib_backend
        self.kf = KalmanFilter()
        self.save_resolution = save_res
        self.joint_names, self.joint_edges = None, None
        self.camera_configs: Dict[int, Dict] = {}
        self.projection_matrices: List[np.ndarray] = []
        self.cam_params: List[Dict] = []
        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.next_global_id = 1
        self.frame_count = 0
        self.csv_file, self.csv_writer = None, None
        self.video_fps = 30
        self.yolo_models = {}
        self.csv_file_train, self.csv_writer_train = None, None

        self.P_by_camid = {}              # cam_id -> matrice P (3x4)
        self.cam_params_by_camid = {}     # cam_id -> {R,t,K,dist}
        self.device_id_by_camid = {}      # cam_id -> DEVICEID (ex: 'CAM03')
        self.csv_file_train, self.csv_writer_train = None, None


    def _draw_skeleton_custom(self, img, pts, edges, color, thickness=2):
        if pts is None or edges is None: return
        h, w = img.shape[:2]
        for p in pts:
            if not np.any(np.isnan(p)) and 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(img, tuple(p.astype(int)), 3, color, -1)
        for e in edges:
            if e[0] >= len(pts) or e[1] >= len(pts): continue
            p1, p2 = pts[e[0]], pts[e[1]]
            if not np.any(np.isnan(p1)) and not np.any(np.isnan(p2)):
                if all((0 <= p1[0] < w, 0 <= p1[1] < h)) and all((0 <= p2[0] < w, 0 <= p2[1] < h)):
                    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), color, thickness)

    def add_camera(self, cam_id: int, source: str, name: str):
        self.camera_configs[cam_id] = {"source": source, "name": name}

    def _initialize_yolo_models(self):
        self.yolo_models.clear()
        for cam_id in self.camera_configs:
            self.yolo_models[cam_id] = YOLO(self.model_path)
    
    def reset_for_new_activity(self):
        self.global_tracks.clear()
        self.next_global_id = 1
        self.frame_count = 0
        self.camera_configs.clear()
        self.projection_matrices.clear()
        self.cam_params.clear()
        self.P_by_camid.clear()
        self.cam_params_by_camid.clear()
        self.device_id_by_camid.clear()
        self.undistort_maps = {} 

        if self.csv_file:
            self.csv_file.close()
            self.csv_file, self.csv_writer = None, None
        if self.csv_file_train:
            self.csv_file_train.close()
            self.csv_file_train, self.csv_writer_train = None, None

    def initialize(self, xcp_file: str, force_variant: Optional[str] = None) -> bool:
        print(f" Initialisation du tracker v68.0 avec Cohérence Spatiale...")
        if self.pose_estimator is None:
            try:
                print(f" Initialisation du modèle de pose RTMLib sur device='{self.rtmlib_device}'...")
                self.pose_estimator = BodyWithFeet(
                    det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
                    det_input_size=(640, 640),
                    pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip',
                    pose_input_size=(288, 384),
                    to_openpose=False, backend=self.rtmlib_backend, device=self.rtmlib_device
                )
                self.joint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel']
                self.joint_edges = [[15, 13], [13, 11], [11, 19], [16, 14], [14, 12], [12, 19], [17, 18], [18, 19], [18, 5], [5, 7], [7, 9], [18, 6], [6, 8], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [15, 20], [15, 22], [15, 24], [16, 21], [16, 23], [16, 25]]
                print(f" RTMLib OK — {len(self.joint_names)} joints.")
            except Exception as e:
                print(f" Erreur RTMLib : {e}")
                return False
        #device_ids = [os.path.basename(cfg["source"]).split('.')[1] for cfg in self.camera_configs.values()]
        #  construire des mappings par cam_id -> DEVICEID -> params ---
        vicon_cams = parse_xcp(xcp_file)

        # Associer chaque cam_id au DEVICEID extrait du nom de fichier
        self.device_id_by_camid = {}
        for cam_id, cfg in self.camera_configs.items():
            # ex: "activity.CAM03.avi" -> "CAM03"
            parts = os.path.basename(cfg["source"]).split('.')
            if len(parts) < 3:
                print(f" Nom de fichier inattendu pour la caméra {cam_id}: {cfg['source']}")
                return False
            self.device_id_by_camid[cam_id] = parts[1]

        device_ids = [self.device_id_by_camid[cid] for cid in sorted(self.camera_configs.keys())]

        projection_sets = create_projection_methods(vicon_cams, device_ids)
        if not projection_sets:
            print(f" Aucune variante de calibration valide pour {xcp_file} et IDs {device_ids}.")
            return False

        best_var = force_variant if force_variant and force_variant in projection_sets else list(projection_sets.keys())[0]

        # Construire P et params pour CHAQUE cam_id
        self.P_by_camid.clear()
        self.cam_params_by_camid.clear()

        for cam_id in self.camera_configs.keys():
            did = self.device_id_by_camid[cam_id]
            if did not in vicon_cams or best_var not in vicon_cams[did]["variants"]:
                print(f" Variante '{best_var}' indisponible pour caméra {did}.")
                return False
            var = vicon_cams[did]["variants"][best_var]
            P = var["K"] @ np.hstack([var["R"], var["t"]])
            self.P_by_camid[cam_id] = P
            self.cam_params_by_camid[cam_id] = var

        print(f" Calibration chargée (variante '{best_var}') pour {len(self.P_by_camid)} caméras.")
        self.undistort_maps = {}
        for cam_id in self.camera_configs.keys():
            vd = self.cam_params_by_camid[cam_id].get('vicon_dist')
            if vd is not None:
                src = self.camera_configs[cam_id]["source"]
                cap_tmp = cv2.VideoCapture(src)
                w = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap_tmp.release()
                print(f"   Cam {cam_id}: calcul undistort maps {w}x{h} "
                    f"(k1={vd['k1']:.3e})...")
                map_x, map_y = build_undistort_maps(
                    w, h, vd['cx_d'], vd['cy_d'], vd['k1'], vd['k2'], vd['k3'])
                self.undistort_maps[cam_id] = (map_x, map_y)
                print(f"   Cam {cam_id}: undistort maps OK")
        if self.undistort_maps:
            print(f" Undistort maps calculées pour {len(self.undistort_maps)} caméras.")
        self._initialize_yolo_models()
        return True


    def _extract_mask_patch(self, mask: np.ndarray, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extrait la portion du masque correspondant à la bbox"""
        h, w = frame_shape
        fx1, fy1, fx2, fy2 = bbox
        
        # S'assurer que les coordonnées sont dans les limites
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(w, fx2), min(h, fy2)
        
        if fx1 >= fx2 or fy1 >= fy2:
            return None
        
        # Redimensionner le masque à la taille de l'image si nécessaire
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return mask[fy1:fy2, fx1:fx2]
    
    def _apply_mask_to_patch(self, patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
        """Applique le masque au patch d'image"""
        if mask_patch.shape[:2] != patch.shape[:2]:
            mask_patch = cv2.resize(mask_patch, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convertir le masque en booléen
        mask_bool = mask_patch > 0.5
        
        # Appliquer le masque (mettre l'arrière-plan en gris foncé)
        masked_patch = patch.copy()
        masked_patch[~mask_bool] = [32, 32, 32]  # Gris foncé
        
        return masked_patch

    
    def _get_torso_box(self, keypoints: np.ndarray, confidences: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame_shape
        torso_indices = [5, 6, 11, 12]
        min_confidence = 0.4
        valid_points = [keypoints[idx] for idx in torso_indices if idx < len(confidences) and confidences[idx] > min_confidence and not np.any(np.isnan(keypoints[idx]))]
        if len(valid_points) < 3:
            return None
        valid_points = np.array(valid_points)
        x1, y1 = np.min(valid_points, axis=0)
        x2, y2 = np.max(valid_points, axis=0)
        margin_x, margin_y = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        x1, y1 = int(max(0, x1 - margin_x)), int(max(0, y1 - margin_y))
        x2, y2 = int(min(w, x2 + margin_x)), int(min(h, y2 + margin_y))
        return (x1, y1, x2, y2) if x1 < x2 and y1 < y2 else None

    def _process_single_view(self, frame: np.ndarray, cam_id: int) -> List[Dict]:
        detections = []
        try:
            keypoints_list, scores_list = self.pose_estimator(frame)
        except Exception as e:
            print(f" Erreur RTMLib caméra {cam_id}: {e}")
            return []
        
        all_poses_2d = np.array(keypoints_list) if keypoints_list is not None and len(keypoints_list) > 0 else np.empty((0, 0, 2))
        all_pose_confidences = np.array(scores_list) if scores_list is not None and len(scores_list) > 0 else np.empty((0, 0))
        
        # Utilisation de YOLOv8-seg 
        results = self.yolo_models[cam_id].track(frame, persist=True, conf=self.params['confidence_thresh'], verbose=False, classes=[0], tracker="bytetrack.yaml")
        if not results or not results[0].boxes or results[0].boxes.id is None:
            return []
            
        yolo_boxes = results[0].boxes
        yolo_masks = results[0].masks
        
        if yolo_masks is None:
            yolo_masks = None
        
        yolo_pose_associations = {}
        if len(all_poses_2d) > 0:
            cost_matrix = np.full((len(yolo_boxes), len(all_poses_2d)), INFTY_COST)
            for i, box in enumerate(yolo_boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                for j, pose in enumerate(all_poses_2d):
                    valid_mask = all_pose_confidences[j] > self.params['min_joint_confidence_for_triangulation']
                    if np.sum(valid_mask) < self.params['min_valid_joints']: continue
                    pose_center = np.mean(pose[valid_mask], axis=0)
                    if x1 < pose_center[0] < x2 and y1 < pose_center[1] < y2:
                        cost_matrix[i, j] = np.linalg.norm(box_center - pose_center)
            
            box_indices, pose_indices = linear_sum_assignment(cost_matrix)
            yolo_pose_associations = {b_idx: p_idx for b_idx, p_idx in zip(box_indices, pose_indices) if cost_matrix[b_idx, p_idx] < 200}
    
        for i, box in enumerate(yolo_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            det = {"bbox": (x1, y1, x2, y2), "features": None, "confidence": float(box.conf[0]), "camera_id": cam_id}
            
            if yolo_masks is not None and i < len(yolo_masks.data):
                mask = yolo_masks.data[i].cpu().numpy()
                det["mask"] = mask
            
            feature_patch_box = (x1, y1, x2, y2)
            if i in yolo_pose_associations:
                pose_idx = yolo_pose_associations[i]
                pose_kpts, pose_confs = all_poses_2d[pose_idx], all_pose_confidences[pose_idx]
                det.update({"pose_2d": pose_kpts, "pose_confidence": float(np.mean(pose_confs)), "pose_2d_joint_confidences": pose_confs})
                torso_box = self._get_torso_box(pose_kpts, pose_confs, frame.shape[:2])
                if torso_box:
                    feature_patch_box = torso_box
    
            # Extraction de features avec masque
            fx1, fy1, fx2, fy2 = feature_patch_box
            if fx1 < fx2 and fy1 < fy2:
                patch = frame[fy1:fy2, fx1:fx2]
                
                if "mask" in det and det["mask"] is not None:
                    mask_patch = self._extract_mask_patch(det["mask"], (fx1, fy1, fx2, fy2), frame.shape[:2])
                    if mask_patch is not None:
                        patch = self._apply_mask_to_patch(patch, mask_patch)
                
                feat = self.feature_extractor.predict(patch)
                if feat is not None:
                    det["features"] = feat.flatten()
            
            if det["features"] is not None:
                detections.append(det)
        return detections
    
    def _update_pose_data_in_track(self, track: GlobalTrack, detection: Dict):
        cid = detection['camera_id']
        track.current_bbox[cid] = detection['bbox']
        if 'mask' in detection and detection['mask'] is not None:
            track.current_mask[cid] = detection['mask']
        if 'pose_2d' in detection:
            track.current_pose_2d[cid] = detection['pose_2d']
            track.pose_2d_confidence[cid] = detection.get('pose_confidence', 0.5)
            track.current_pose_2d_joint_confidences[cid] = detection.get('pose_2d_joint_confidences', np.full(len(detection['pose_2d']), 0.5))

    def _triangulate_group(self, group: List[Dict]) -> Optional[Tuple[np.ndarray, float, float]]:
        valid_dets = [d for d in group if 'pose_2d' in d]
        if len(valid_dets) < self.params['min_views_for_3d']:
            return None
        
        centers_2d, weights, proj_matrices = [], [], []
        
        for d in valid_dets:
            joint_confs = d.get('pose_2d_joint_confidences', np.array([]))
            pose_2d = d['pose_2d']
            
            # Stratégie de calcul du centre plus robuste
            center = None
            weight = 0
            
            # Essayer d'abord avec le torse 
            torso_indices = [5, 6, 11, 12]  # épaules, hanches
            if len(joint_confs) > max(torso_indices):
                valid_torso_mask = joint_confs[torso_indices] > self.params['min_joint_confidence_for_triangulation']
                if np.sum(valid_torso_mask) >= 2:
                    center = np.mean(pose_2d[torso_indices][valid_torso_mask], axis=0)
                    weight = np.mean(joint_confs[torso_indices][valid_torso_mask])
            
            # Si pas de torse, essayer avec la tête et le cou
            if center is None:
                head_indices = [0, 17, 18]  # nez, tête, cou
                if len(joint_confs) > max(head_indices):
                    valid_head_mask = joint_confs[head_indices] > self.params['min_joint_confidence_for_triangulation']
                    if np.sum(valid_head_mask) >= 2:
                        center = np.mean(pose_2d[head_indices][valid_head_mask], axis=0)
                        weight = np.mean(joint_confs[head_indices][valid_head_mask]) * 0.8  # Pénalité pour tête seule
            
            # En dernier recours, utiliser tous les joints valides
            if center is None:
                valid_mask = joint_confs > self.params['min_joint_confidence_for_triangulation']
                if np.sum(valid_mask) >= 3:
                    center = np.mean(pose_2d[valid_mask], axis=0)
                    weight = np.mean(joint_confs[valid_mask]) * 0.7
                else:
                    # Utiliser le centre de la bbox
                    bbox = d['bbox']
                    center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
                    weight = 0.2
    
            centers_2d.append(center)
            weights.append(weight)
            #proj_matrices.append(self.projection_matrices[d['camera_id']])
            proj_matrices.append(self.P_by_camid[d['camera_id']])
    
        if len(centers_2d) < self.params['min_views_for_3d']:
            return None
    

        try:
            # Première tentative de triangulation
            center_3d = weighted_triangulate_points(proj_matrices, centers_2d, weights)
            
            # Validation de la cheiralité
            cheirality_checks = [
                check_cheirality(center_3d, self.cam_params_by_camid[d['camera_id']]["R"],
                                 self.cam_params_by_camid[d['camera_id']]["t"])
                for d in valid_dets
            ]

            
            if sum(cheirality_checks) < len(valid_dets):
                # Si échec, essayer avec seulement les vues qui passent la cheiralité
                valid_indices = [i for i, check in enumerate(cheirality_checks) if check]
                if len(valid_indices) >= 2:
                    centers_2d_valid = [centers_2d[i] for i in valid_indices]
                    weights_valid = [weights[i] for i in valid_indices]
                    proj_matrices_valid = [proj_matrices[i] for i in valid_indices]
                    
                    center_3d = weighted_triangulate_points(
                        proj_matrices_valid, centers_2d_valid, weights_valid
                    )
                else:
                    return None
            
            # Calcul d'erreur de reprojection
            errors = []
            for p, c in zip(proj_matrices, centers_2d):
                proj_point = (p @ np.append(center_3d, 1))
                if proj_point[2] > 0:  # Devant la caméra
                    proj_2d = proj_point[:2] / proj_point[2]
                    errors.append(np.linalg.norm(proj_2d - c))
                else:
                    errors.append(1000.0)  # Pénalité élevée si derrière la caméra
            
            mean_error = float(np.mean(errors))


            #  Adapter le seuil pour les mouvements rapides
            adaptive_threshold = self.params['reprojection_error_thresh']
            
            # Vérifier si c'est un mouvement rapide (sledPush)
            if len(valid_dets) >= 2:
                # Calculer la disparité entre les vues
                bbox_sizes = []
                for d in valid_dets:
                    bbox = d['bbox']
                    bbox_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    bbox_sizes.append(bbox_size)
                
                # Si grande variation de taille = mouvement rapide/proche caméra
                size_variance = np.var(bbox_sizes) / np.mean(bbox_sizes)**2
                if size_variance > 0.3:  # Grande variance = mouvement dynamique
                    adaptive_threshold *= 2.0  # Plus tolérant
                    #print(f"🏃 Mouvement rapide détecté, seuil adapté à {adaptive_threshold}")
            
            if np.mean(weights) < 0.5:  # Détections de faible qualité
                adaptive_threshold *= 1.5
            
            
            if mean_error > adaptive_threshold:
                return None
            
            reproj_conf = 1.0 - (mean_error / adaptive_threshold)
            pose_conf = float(np.mean([d.get('pose_confidence', 0.5) for d in valid_dets]))
            
            return center_3d, reproj_conf, pose_conf
            
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f" Erreur triangulation: {e}")
            return None

    def _calculate_spatial_consistency_score(self, group1: List[Dict], group2: List[Dict]) -> float:
        cameras1, cameras2 = {d['camera_id'] for d in group1}, {d['camera_id'] for d in group2}
        if not cameras1.isdisjoint(cameras2): return 0.0
        merged_group = group1 + group2
        triangulation_result = self._triangulate_group(merged_group)
        return float(triangulation_result[1]) if triangulation_result else 0.0

    def _group_detections(self, all_dets: List[Dict]) -> List[List[Dict]]:
        if not all_dets: return []
        valid_dets = [d for d in all_dets if d["features"] is not None]
        if not valid_dets: return []
        n_dets = len(valid_dets)
        appearance_cost_matrix = cdist(np.array([d["features"] for d in valid_dets]), np.array([d["features"] for d in valid_dets]), "cosine")
        clusters = {k: [k] for k in range(n_dets)}
        while True:
            min_combined_cost, best_pair = float('inf'), (-1, -1)
            cluster_keys = list(clusters.keys())
            for i in range(len(cluster_keys)):
                for j in range(i + 1, len(cluster_keys)):
                    c1_key, c2_key = cluster_keys[i], cluster_keys[j]
                    group1, group2 = [valid_dets[idx] for idx in clusters[c1_key]], [valid_dets[idx] for idx in clusters[c2_key]]
                    spatial_score = self._calculate_spatial_consistency_score(group1, group2)
                    if spatial_score == 0.0: continue
                    inter_costs = appearance_cost_matrix[np.ix_(clusters[c1_key], clusters[c2_key])]
                    if inter_costs.size == 0: continue
                    mean_appearance_cost = float(np.mean(inter_costs))
                    w_spatial = 0.7 if spatial_score > 0.5 else 0.3
                    combined_cost = (1.0 - w_spatial) * mean_appearance_cost + w_spatial * (1 - spatial_score)
                    if combined_cost < self.params['intra_frame_thresh'] and combined_cost < min_combined_cost:
                        min_combined_cost, best_pair = combined_cost, (c1_key, c2_key)
            if best_pair == (-1, -1):
                break
            c1_key, c2_key = best_pair
            clusters[c1_key].extend(clusters[c2_key])
            del clusters[c2_key]

                # avant le return
        final_clusters = [[valid_dets[idx] for idx in cluster_indices] for cluster_indices in clusters.values()]
    

        if len(final_clusters) > 1:
            merged_anything = True
            while merged_anything:
                merged_anything = False
                i = 0
                while i < len(final_clusters):
                    j = i + 1
                    while j < len(final_clusters):
                        group1 = final_clusters[i]
                        group2 = final_clusters[j]
    
                        # Ne pas fusionner si les groupes partagent déjà une caméra
                        cams1 = {d['camera_id'] for d in group1}
                        cams2 = {d['camera_id'] for d in group2}
                        if not cams1.isdisjoint(cams2):
                            j += 1
                            continue
    
                        # Trianguler chaque groupe
                        pos_info1 = self._triangulate_group(group1)
                        pos_info2 = self._triangulate_group(group2)
    
                        if pos_info1 and pos_info2:
                            dist_3d = np.linalg.norm(pos_info1[0] - pos_info2[0])
                            # Si deux groupes sont extrêmement proches, c'est presque certainement la même personne
                            if dist_3d < 400.0: # Seuil très strict de 40cm
                                print(f" Fusion de sécurité dans _group_detections (dist={dist_3d:.0f}mm)")
                                # Fusionner group2 dans group1
                                final_clusters[i].extend(group2)
                                # Supprimer group2
                                final_clusters.pop(j)
                                merged_anything = True
                                # Ne pas incrémenter j, car la liste a changé
                            else:
                                j += 1
                        else:
                            j += 1
                    i += 1
                    
        return final_clusters
        

    def _update_occlusion_state(self, tracks: List[GlobalTrack]):
        """ Met à jour l'état d'occlusion en déterminant qui cache qui. """
        for t in tracks:
            t.is_occluded_by = None

        confirmed_3d_tracks = [t for t in tracks if t.has_3d]
        
        for i in range(len(confirmed_3d_tracks)):
            for j in range(i + 1, len(confirmed_3d_tracks)):
                t1 = confirmed_3d_tracks[i]
                t2 = confirmed_3d_tracks[j]

                dist_xy = np.linalg.norm(t1.mean_3d[:2] - t2.mean_3d[:2])
                
                if dist_xy < self.params['occlusion_xy_dist_thresh']:
                    is_overlapping = False
                    for cam_id in self.camera_configs.keys():
                        if cam_id in t1.current_bbox and cam_id in t2.current_bbox:
                            b1, b2 = t1.current_bbox[cam_id], t2.current_bbox[cam_id]
                            inter_x1, inter_y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                            inter_x2, inter_y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                            if max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1) > 0:
                                is_overlapping = True
                                break
                    
                    if is_overlapping:
                        occluder, occluded = (t1, t2) if t1.mean_3d[2] < t2.mean_3d[2] else (t2, t1)
                        occluded.is_occluded_by = occluder.global_id
                        occluded.occlusion_timer = self.params['occlusion_patience']

    def _compute_appearance_cost(self, tracks: List[GlobalTrack], groups: List[List[Dict]]) -> np.ndarray:
        num_tracks, num_groups = len(tracks), len(groups)
        if num_tracks == 0 or num_groups == 0: return np.empty((num_tracks, num_groups))
        cost_matrix = np.full((num_tracks, num_groups), 1.0)
        
        group_features_list, valid_group_indices = [], []
        for i, grp in enumerate(groups):
            features = [d['features'] for d in grp if 'features' in d and d['features'] is not None]
            if features:
                group_features_list.append(np.mean(features, axis=0))
                valid_group_indices.append(i)
    
        if not group_features_list:
            return cost_matrix
    
        group_features = np.stack(group_features_list)
    
        for i, track in enumerate(tracks):
            if track.feature_gallery:
                gallery = np.array(list(track.feature_gallery))
                group_costs = cdist(group_features, gallery, metric='cosine').min(axis=1)
                cost_matrix[i, valid_group_indices] = group_costs
                
        return cost_matrix
        
    def _compute_cost_matrix(self, tracks: List[GlobalTrack], groups: List[List[Dict]]) -> np.ndarray:
        num_tracks, num_groups = len(tracks), len(groups)
        if num_tracks == 0 or num_groups == 0:
            return np.empty((0, 0))
        
        cost_matrix = np.full((num_tracks, num_groups), INFTY_COST)
        app_cost = self._compute_appearance_cost(tracks, groups)
        group_3d_infos = [self._triangulate_group(grp) for grp in groups]
        group_visibilities = [self._estimate_group_visibility_ratio(grp) for grp in groups]
        
        track_positions_3d = {t.global_id: t.mean_3d[:3] for t in self.global_tracks.values() if t.has_3d}
        
        for i, track in enumerate(tracks):
            for j, group in enumerate(groups):
                group_3d_info = group_3d_infos[j]
                group_visibility = group_visibilities[j]
                num_views_group = len(group)
                
                # Analyser l'historique de vues de la track
                recent_view_pattern = list(track.multi_view_history)[-5:] if track.multi_view_history else []
                is_transitioning = len(set(recent_view_pattern)) > 1 if recent_view_pattern else False
                
                # Adapter les poids selon le contexte de transition
                if is_transitioning or track.single_view_streak > 0:
                    # Mode transition : plus tolérant
                    if num_views_group == 1 and track.has_3d:
                        # Vue unique mais track a une position 3D
                        w_app, w_mot, w_spat = 0.6, 0.3, 0.1
                        adaptive_thresh = 0.85
                        gating_multiplier = 3.0
                    elif num_views_group >= 2 and not track.has_3d:
                        # Multi-vues mais track sans 3D (récemment en vue unique)
                        w_app, w_mot, w_spat = 0.5, 0.3, 0.2
                        adaptive_thresh = 0.8
                        gating_multiplier = 2.5
                    else:
                        # Cas standard de transition
                        w_app, w_mot, w_spat = 0.5, 0.25, 0.25
                        adaptive_thresh = 0.75
                        gating_multiplier = 2.0
                elif group_visibility < 0.3:  # Détection très partielle
                    if track.state == TrackState.Occluded:
                        w_app, w_mot, w_spat = 0.2, 0.3, 0.5
                        adaptive_thresh = 0.9
                        gating_multiplier = 6.0
                    else:
                        w_app, w_mot, w_spat = 0.3, 0.2, 0.5
                        adaptive_thresh = 0.85
                        gating_multiplier = 5.0
                elif track.state == TrackState.Occluded:
                    w_app, w_mot, w_spat = 0.1, 0.45, 0.45
                    adaptive_thresh = 0.95
                    gating_multiplier = 10.0
                elif track.is_occluded_by is not None or track.time_since_update > 5:
                    w_app, w_mot, w_spat = 0.7, 0.15, 0.15
                    adaptive_thresh = 0.8
                    gating_multiplier = 4.0
                elif track.has_3d and group_3d_info:
                    w_app, w_mot, w_spat = 0.5, 0.1, 0.4
                    adaptive_thresh = 0.7
                    gating_multiplier = 2.0
                else:
                    w_app, w_mot, w_spat = 0.7, 0.3, 0.0
                    adaptive_thresh = 0.8
                    gating_multiplier = 2.0
    
                # Motion cost (2D)
                motion_cost = INFTY_COST
                avg_bbox = np.mean([d['bbox'] for d in group], axis=0)
                measurement_2d = bbox_to_kalman_format(avg_bbox)
                if track.kf_mean is not None:
                    gating_thresh_motion = chi2inv95[4] * gating_multiplier
                    motion_cost = self.kf.gating_distance(track.kf_mean, track.kf_covariance, np.array([measurement_2d]))[0]
                    if motion_cost > gating_thresh_motion:
                        continue
                    motion_cost /= gating_thresh_motion
    
                # Spatial cost (3D)
                spat_cost = INFTY_COST
                if track.has_3d and group_3d_info:
                    pos_3d = group_3d_info[0]
                    gating_thresh_spat = chi2inv95[3] * gating_multiplier
                    spat_cost = track.kf_3d.gating_distance(track.mean_3d, track.covariance_3d, np.array([pos_3d]))[0]
                    if spat_cost > gating_thresh_spat:
                        continue
                    spat_cost /= gating_thresh_spat
                elif track.has_3d and num_views_group == 1:
                    # Track a 3D mais groupe en vue unique : utiliser la position prédite
                    spat_cost = 0.5  # Coût modéré fixe
                
                total_cost = (w_app * app_cost[i, j] + w_mot * motion_cost + w_spat * spat_cost)
                
                # Pénalité de proximité avec d'autres tracks (seulement si 3D disponible)
                if track.has_3d and group_3d_info:
                    proximity_penalty = 0.0
                    current_group_pos = group_3d_info[0]
                    for other_track_id, other_track_pos in track_positions_3d.items():
                        if track.global_id == other_track_id: continue
                        if np.linalg.norm(track.mean_3d[:3] - other_track_pos) < 1200.0:
                            if np.linalg.norm(current_group_pos - other_track_pos) < np.linalg.norm(current_group_pos - track.mean_3d[:3]):
                                proximity_penalty = max(proximity_penalty, 0.2)
                    total_cost += proximity_penalty


                # --- Anti-switch penalty (identity inertia) ---
                switch_penalty = 0.0
                
                if group_3d_info and track.has_3d:
                    current_group_pos = group_3d_info[0]
                    reproj_conf = float(group_3d_info[1])  # 0..1
                
                    # Appliquer seulement si la triangulation est raisonnablement fiable
                    if reproj_conf >= 0.45:
                        d_self = np.linalg.norm(current_group_pos - track.mean_3d[:3])
                
                        best_other_margin = 0.0
                        for other in tracks:
                            if other.global_id == track.global_id or not other.has_3d:
                                continue
                            d_other = np.linalg.norm(current_group_pos - other.mean_3d[:3])
                
                            # Danger de switch si le groupe est plus proche d'un autre track
                            if d_other + 1e-6 < d_self:
                                # marge normalisée (0..1)
                                margin = min(1.0, max(0.0, (d_self - d_other) / max(1.0, d_self)))
                                best_other_margin = max(best_other_margin, margin)
                
                        # Poids dépendant de l'ambiguïté apparence :
                        # app_cost bas = apparence forte → pénaliser moins
                        # app_cost haut = apparence faible → pénaliser plus
                        ambig = float(app_cost[i, j])          # 0..1 (cosine cost)
                        w = 0.15 + 0.20 * min(1.0, ambig)      # ~0.15..0.35
                        switch_penalty = w * best_other_margin
                
                total_cost += switch_penalty

                if total_cost < adaptive_thresh:
                    cost_matrix[i, j] = total_cost
                    
        return cost_matrix

    def _estimate_group_visibility_ratio(self, group: List[Dict]) -> float:
        """Calcule le ratio moyen de joints visibles pour un groupe de détections."""
        ratios = []
        for d in group:
            if 'pose_2d_joint_confidences' in d:
                confs = d['pose_2d_joint_confidences']
                if confs is not None and len(confs) > 0:
                    ratio = np.mean(confs > self.params['min_joint_confidence_for_triangulation'])
                    ratios.append(ratio)
        return float(np.mean(ratios)) if ratios else 0.0

    def _update_matched_track(self, track: GlobalTrack, group: List[Dict], frame_idx: int):
        """Met à jour une track matchée avec validation de cohérence temporelle"""
        
        # Gérer les transitions entre vue unique et vues multiples
        self._handle_view_transitions(track, group, frame_idx)
        
        # Marquer la track comme vue
        track.mark_hit(frame_idx)
        
        # Mettre à jour les données de pose pour chaque caméra
        for d in group:
            self._update_pose_data_in_track(track, d)
        
        # Mise à jour du filtre de Kalman 2D
        avg_bbox = np.mean([d['bbox'] for d in group], axis=0)
        measurement_2d = bbox_to_kalman_format(avg_bbox)
        track.kf_mean, track.kf_covariance = self.kf.update(
            track.kf_mean, track.kf_covariance, measurement_2d
        )
        
        # Calculer le ratio de visibilité
        track.last_visible_ratio = self._estimate_group_visibility_ratio(group)
        
        # Mettre à jour les features d'apparence
        group_features_list = [d["features"] for d in group if 'features' in d and d['features'] is not None]
        if group_features_list:
            group_features = np.mean(group_features_list, axis=0)
            track.update_features(group_features)
        
        # Tentative de triangulation 3D
        group_3d_info = self._triangulate_group(group)
        
        if group_3d_info:
            pos_3d, reproj_conf, pose_conf = group_3d_info
            
            # VALIDATION DE COHÉRENCE TEMPORELLE
            should_update_3d = True
            
            if track.has_3d and len(track.position_3d_history) > 0:
                # Calculer la vitesse maximale plausible
                time_delta = 1  # Assuming frame-to-frame
                max_velocity_per_frame = 500.0  # 500mm/frame à 30fps = ~15m/s max
                
                # Si on a manqué des frames, ajuster la tolérance
                if track.time_since_update > 0:
                    max_velocity_per_frame *= (track.time_since_update + 1)
                
                # Vérifier avec la dernière position connue
                last_pos = track.position_3d_history[-1]
                distance = np.linalg.norm(pos_3d - last_pos)
                
                if distance > max_velocity_per_frame:
                    # Vérifier si c'est cohérent avec la prédiction Kalman
                    if track.mean_3d is not None:
                        predicted_pos = track.mean_3d[:3]
                        dist_to_prediction = np.linalg.norm(pos_3d - predicted_pos)
                        
                        if dist_to_prediction > max_velocity_per_frame * 2:
                            print(f" Track {track.global_id}: Position 3D aberrante détectée")
                            print(f"   Distance depuis dernière: {distance:.0f}mm")
                            print(f"   Distance depuis prédiction: {dist_to_prediction:.0f}mm")
                            should_update_3d = False
                        else:
                            # Position plausible par rapport à la prédiction
                            print(f" Track {track.global_id}: Correction avec prédiction Kalman")
                            # Pondérer entre la mesure et la prédiction
                            alpha = min(0.7, reproj_conf)  # Plus on fait confiance à la mesure, plus alpha est grand
                            pos_3d = alpha * pos_3d + (1 - alpha) * predicted_pos
                
                # Vérifier la cohérence avec l'historique récent
                if should_update_3d and len(track.position_3d_history) >= 3:
                    recent_positions = np.array(list(track.position_3d_history)[-3:])
                    
                    # Calculer la vitesse moyenne récente
                    recent_velocities = np.diff(recent_positions, axis=0)
                    avg_velocity = np.mean(recent_velocities, axis=0)
                    
                    # Prédire la prochaine position basée sur la vitesse
                    expected_pos = last_pos + avg_velocity
                    deviation = np.linalg.norm(pos_3d - expected_pos)
                    
                    # Si la déviation est trop grande, utiliser une position lissée
                    if deviation > 300.0:  # 300mm de déviation max
                        print(f"Track {track.global_id}: Lissage de position (dév={deviation:.0f}mm)")
                        # Moyenne pondérée entre la mesure et l'attendu
                        weight = max(0.3, 1.0 - deviation / 1000.0)
                        pos_3d = weight * pos_3d + (1 - weight) * expected_pos
            
            # Mise à jour 3D si validation OK
            if should_update_3d:
                if track.has_3d:
                    # Vérifier le drift du filtre de Kalman avant mise à jour
                    if track.mean_3d is not None:
                        prediction_error = np.linalg.norm(pos_3d - track.mean_3d[:3])
                        
                        # Si erreur trop importante et track ancienne, réinitialiser
                        if prediction_error > 1500.0 and track.age > 100:
                            print(f" Track {track.global_id}: Reset Kalman 3D (err={prediction_error:.0f}mm)")
                            track.mean_3d, track.covariance_3d = track.kf_3d.initiate(pos_3d)
                        else:
                            # Mise à jour normale
                            track.mean_3d, track.covariance_3d = track.kf_3d.update(
                                track.mean_3d, track.covariance_3d, pos_3d
                            )
                    else:
                        # Première initialisation 3D
                        track.mean_3d, track.covariance_3d = track.kf_3d.initiate(pos_3d)
                else:
                    # Première fois qu'on a une position 3D
                    track.mean_3d, track.covariance_3d = track.kf_3d.initiate(pos_3d)
                    print(f"Track {track.global_id}: Première triangulation 3D")
                
                # Mettre à jour l'état et l'historique
                track.state = TrackState.Confirmed3D
                track.last_known_3d_pos = pos_3d.copy()
                track.position_3d_history.append(pos_3d.copy())
                track.last_triangulated_frame = frame_idx
                
                # Calculer et stocker la variance de position
                if len(track.position_3d_history) >= 3:
                    recent_pos = np.array(list(track.position_3d_history)[-5:])
                    position_variance = np.var(recent_pos, axis=0)
                    track.position_variance_history.append(np.mean(position_variance))
            else:
                # Si on ne peut pas mettre à jour la 3D, au moins prédire
                if track.has_3d and track.mean_3d is not None:
                    # Faire seulement la prédiction sans mise à jour
                    track.mean_3d, track.covariance_3d = track.kf_3d.predict(
                        track.mean_3d, track.covariance_3d
                    )
                    # Augmenter l'incertitude
                    track.covariance_3d *= 1.2
                    print(f" Track {track.global_id}: Prédiction 3D seulement (pas de mise à jour)")
        
        else:
            # Pas de triangulation possible
            if track.has_3d and track.mean_3d is not None:
                # Maintenir la prédiction 3D même sans triangulation
                print(f" Track {track.global_id}: Maintien prédiction 3D (pas de triangulation)")
                # L'incertitude augmentera naturellement avec les prédictions successives
            
            # Si on est en vue unique depuis trop longtemps, dégrader l'état
            if track.single_view_streak > 15:
                if track.state == TrackState.Confirmed3D:
                    track.state = TrackState.Confirmed2D
                    print(f" Track {track.global_id}: Dégradation 3D → 2D (vue unique prolongée)")
        
        # Mise à jour de l'aspect ratio pour détecter les changements de forme
        current_bbox = avg_bbox
        aspect_ratio = (current_bbox[2] - current_bbox[0]) / max(1, current_bbox[3] - current_bbox[1])
        track.aspect_ratio_history.append(aspect_ratio)
        
        # Détection de mouvement rapide
        if track.last_known_3d_pos is not None and group_3d_info:
            movement = np.linalg.norm(group_3d_info[0] - track.last_known_3d_pos)
            if movement > track.HIGH_MOTION_THRESHOLD_MM:
                track.high_motion_timer = track.HIGH_MOTION_DURATION
            elif track.high_motion_timer > 0:
                track.high_motion_timer -= 1
        
        # Détection de proximité avec le bord de l'image
        for d in group:
            bbox = d['bbox']
            cam_id = d['camera_id']
            # Assumons une résolution standard, à adapter selon votre config
            img_width, img_height = 1920, 1080  
            
            edge_margin = self.params['edge_margin_pixels']
            near_edge = (
                bbox[0] < edge_margin or 
                bbox[1] < edge_margin or 
                bbox[2] > img_width - edge_margin or 
                bbox[3] > img_height - edge_margin
            )
            
            if near_edge:
                track.last_near_edge = True
                break
        else:
            track.last_near_edge = False
        
        # Log de debug périodique
        if frame_idx % 30 == 0 and track.has_3d:
            print(f" Track {track.global_id}: pos=({track.mean_3d[0]:.0f}, {track.mean_3d[1]:.0f}, {track.mean_3d[2]:.0f}), "
                  f"vues={len(group)}, vis={track.last_visible_ratio:.2f}")


    def _check_and_reset_kalman(self, track: GlobalTrack):
        """Vérifie et réinitialise le Kalman si nécessaire"""
        if track.covariance_3d is not None:
            # Vérifier la divergence
            trace = np.trace(track.covariance_3d[:3, :3])
            if trace > 1e7:  # Divergence détectée
                if track.last_known_3d_pos is not None:
                    print(f" Track {track.global_id}: Reset Kalman (trace={trace:.0e})")
                    # Réinitialiser avec la dernière position connue
                    track.mean_3d, track.covariance_3d = track.kf_3d.initiate(track.last_known_3d_pos)
                else:
                    # Marquer comme perdue
                    track.state = TrackState.Lost

    def _handle_view_transitions(self, track: GlobalTrack, group: List[Dict], frame_idx: int):
        """Gère les transitions entre vue unique et vues multiples"""
        
        num_views = len(group)
        track.multi_view_history.append(num_views)
        
        # Cas 1 : Passage de multi-vues à vue unique
        if track.has_3d and num_views == 1:
            track.single_view_streak += 1
            
            # Maintenir la position 3D prédite même avec une seule vue
            if track.mean_3d is not None:
                # Garder la prédiction du Kalman 3D active
                track.mean_3d, track.covariance_3d = track.kf_3d.predict(
                    track.mean_3d, track.covariance_3d
                )
                # Augmenter légèrement l'incertitude
                track.covariance_3d *= 1.1
                
            # Ne pas dégrader l'état immédiatement
            if track.single_view_streak < 10:  # Tolérer 10 frames en vue unique
                track.state = TrackState.Confirmed2D
            
            print(f" Track {track.global_id}: Vue unique (streak: {track.single_view_streak})")
            
        # Cas 2 : Passage de vue unique à multi-vues
        elif not track.has_3d and num_views >= 2:
            track.single_view_streak = 0
            triangulation_result = self._triangulate_group(group)
            
            if triangulation_result:
                pos_3d, reproj_conf, pose_conf = triangulation_result
                
                # Initialiser ou réactiver la position 3D
                if track.last_triangulated_frame < 0:
                    # Première triangulation
                    track.mean_3d, track.covariance_3d = track.kf_3d.initiate(pos_3d)
                else:
                    # Réactivation après période en vue unique
                    frames_since_3d = frame_idx - track.last_triangulated_frame
                    if frames_since_3d > 30:
                        # Réinitialiser si trop longtemps
                        track.mean_3d, track.covariance_3d = track.kf_3d.initiate(pos_3d)
                    else:
                        # Mise à jour normale
                        track.mean_3d, track.covariance_3d = track.kf_3d.update(
                            track.mean_3d, track.covariance_3d, pos_3d
                        )
                
                track.last_triangulated_frame = frame_idx
                track.state = TrackState.Confirmed3D
                track.last_known_3d_pos = pos_3d
                
                print(f" Track {track.global_id}: Triangulation réactivée")
                
        # Cas 3 : Maintien en multi-vues
        elif num_views >= 2:
            track.single_view_streak = 0
            track.last_triangulated_frame = frame_idx
            

    
    def _resolve_association_conflicts(self, matches: List[Tuple[int, int]], groups: List[List[Dict]], cost_matrix: np.ndarray, active_tracks: List[GlobalTrack]) -> List[Tuple[int, int]]:
        """
         Vérifie et résout les incohérences après l'assignation.
        1.  Détecte les "Split IDs" : un même ID assigné à plusieurs groupes trop éloignés.
            -> Invalide le match le moins confiant (coût le plus élevé).
        2.  Détecte les "Conflits de Proximité" : deux IDs différents assignés à des groupes très proches.
            -> Invalide le match le moins confiant des deux.
        """
        if not matches:
            return []
    
        # Dictionnaire: track_id -> liste de (group_idx, confidence_score)
        assignments_by_track = defaultdict(list)
        track_map = {t.global_id: i for i, t in enumerate(active_tracks)}
    
        for track_id, group_idx in matches:
            track_idx = track_map.get(track_id)
            if track_idx is None: continue
            
            # Trouver l'index du groupe dans la cost_matrix originale
            # C'est une simplification, il faudrait passer la liste des indices de groupes
            # Pour cet exemple, on suppose que group_idx est l'index dans la matrice de coût
            cost = cost_matrix[track_idx, group_idx]
            assignments_by_track[track_id].append({'group_idx': group_idx, 'cost': cost})
    
        group_positions = {i: self._triangulate_group(g) for i, g in enumerate(groups)}
        
        invalid_matches = set()
    
        # --- 1. Résolution des Split IDs (1 ID, N lieux) ---
        for track_id, assignments in assignments_by_track.items():
            if len(assignments) > 1:
                positions = []
                for assign in assignments:
                    pos_info = group_positions.get(assign['group_idx'])
                    if pos_info:
                        positions.append({'pos': pos_info[0], 'assign': assign})
                
                if len(positions) > 1:
                    # Si les positions sont trop éloignées, c'est un split
                    for i in range(len(positions)):
                        for j in range(i + 1, len(positions)):
                            dist = np.linalg.norm(positions[i]['pos'] - positions[j]['pos'])
                            if dist > self.params['SPATIAL_CONSISTENCY_THRESH_MM']:
                                print(f" Split ID détecté pour Track {track_id} (dist={dist:.0f}mm). Résolution...")
                                # Garder le match avec le coût le plus faible (meilleure confiance)
                                if positions[i]['assign']['cost'] > positions[j]['assign']['cost']:
                                    invalid_matches.add(tuple(sorted((track_id, positions[i]['assign']['group_idx']))))
                                else:
                                    invalid_matches.add(tuple(sorted((track_id, positions[j]['assign']['group_idx']))))
    
        # --- 2. Résolution des Conflits de Proximité (N IDs, 1 lieu) ---
        match_list = list(matches)
        for i in range(len(match_list)):
            for j in range(i + 1, len(match_list)):
                id1, g_idx1 = match_list[i]
                id2, g_idx2 = match_list[j]
    
                pos_info1 = group_positions.get(g_idx1)
                pos_info2 = group_positions.get(g_idx2)
    
                if pos_info1 and pos_info2:
                    dist = np.linalg.norm(pos_info1[0] - pos_info2[0])
                    # Si deux tracks différentes sont assignées à des positions trop proches
                    if dist < 500.0: # Seuil de conflit de proximité (50cm)
                        print(f" Conflit de proximité détecté: ID {id1} vs ID {id2} (dist={dist:.0f}mm).")
                        cost1 = assignments_by_track[id1][0]['cost']
                        cost2 = assignments_by_track[id2][0]['cost']
                        
                        # Annuler le match avec le coût le plus élevé
                        if cost1 > cost2:
                            invalid_matches.add(tuple(sorted((id1, g_idx1))))
                        else:
                            invalid_matches.add(tuple(sorted((id2, g_idx2))))
    
        # Filtrer les matches valides
        resolved_matches = [m for m in matches if tuple(sorted(m)) not in invalid_matches]
        
        if len(invalid_matches) > 0:
            print(f" Résolution des conflits: {len(matches) - len(resolved_matches)} match(es) annulé(s).")
            
        return resolved_matches

    def _update_tracks(self, frame_idx: int, dets_per_cam: Dict[int, List[Dict]], frames: List[np.ndarray]):
        if self.debug:
            if frame_idx % 10 == 0: self.debugger.save_feature_galleries(self.global_tracks, frame_idx)
            self.debugger.save_track_trajectories(self.global_tracks, frame_idx)

        for track in self.global_tracks.values():
            if track.state != TrackState.Lost: track.age += 1
            if track.kf_mean is not None:
                track.kf_mean, track.kf_covariance = self.kf.predict(track.kf_mean, track.kf_covariance)
            if track.mean_3d is not None:
                track.mean_3d, track.covariance_3d = track.kf_3d.predict(track.mean_3d, track.covariance_3d)
            track.current_bbox.clear(); track.current_pose_2d.clear(); track.current_mask.clear()
            if track.occlusion_timer > 0: track.occlusion_timer -= 1
        
        active_tracks = [t for t in self.global_tracks.values() if t.state != TrackState.Lost]
        
        self._update_occlusion_state(active_tracks)

        all_dets = [det for dets in dets_per_cam.values() for det in dets]
        groups = self._group_detections(all_dets)
        
        matches, unmatched_track_ids, unmatched_groups_indices = self._matching_cascade(
            active_tracks, groups)
        cost_matrix = self._compute_cost_matrix(active_tracks, groups)

        matches = self._resolve_association_conflicts(matches, groups, cost_matrix, active_tracks)

        # Recalculer les listes après la validation
        matched_track_ids_set = {m[0] for m in matches}
        matched_group_indices_set = {m[1] for m in matches}
        unmatched_track_ids = [t.global_id for t in active_tracks if t.global_id not in matched_track_ids_set]
        unmatched_groups_indices = [i for i in range(len(groups)) if i not in matched_group_indices_set]


        for track_id, group_idx in matches:
            if track_id in self.global_tracks:
                self._update_matched_track(self.global_tracks[track_id], groups[group_idx], frame_idx)
        
        matched_track_positions = [self.global_tracks[tid].mean_3d[:3] for tid, _ in matches if self.global_tracks.get(tid) and self.global_tracks[tid].has_3d]
        for track_id in unmatched_track_ids:
            track = self.global_tracks.get(track_id)
            if not track: continue
            
            is_potentially_occluded = False
            if track.has_3d and matched_track_positions:
                distances = cdist(np.array([track.mean_3d[:3]]), np.array(matched_track_positions))
                if distances.size > 0 and np.min(distances) < self.params['occlusion_xy_dist_thresh']:
                        is_potentially_occluded = True

            is_near_other_track = is_potentially_occluded
            is_recent_disappearance = track.time_since_update <= 3
            
            if (is_near_other_track or is_recent_disappearance) and track.time_since_update < self.params['max_occlusion_frames']:
                track.state = TrackState.Occluded
                track.time_since_update += 1 
                if track.covariance_3d is not None: 
                    track.covariance_3d *= 1.2
                if track.kf_covariance is not None:
                    track.kf_covariance *= 1.2
            else:
                track.mark_missed(self.params['track_patience'])
        
        for group_idx in unmatched_groups_indices:
            self._create_track(frame_idx, groups[group_idx])
        if self.frame_count % 30 == 0:  # Toutes les secondes
            self._merge_duplicate_tracks()

        to_delete = [tid for tid, t in self.global_tracks.items() if t.state == TrackState.Lost and t.time_since_update > self.params['track_patience']]
        for tid in to_delete:
            if tid in self.global_tracks:
                self.global_tracks.pop(tid)
                print(f" Track {tid} SUPPRIMÉ")
        
        for track_id, track in list(self.global_tracks.items()):
            if track.covariance_3d is not None:
                max_uncertainty = np.max(np.diag(track.covariance_3d[:3, :3]))
                if max_uncertainty > 1e6:  # Covariance a explosé
                    print(f" Track {track_id}: Kalman divergé, suppression")
                    del self.global_tracks[track_id]
                    
                    # Supprimer les tracks fantômes (pas vues depuis longtemps mais pas marquées Lost)
                if track.time_since_update > 50 and track.state != TrackState.Lost:
                    track.state = TrackState.Lost

    def _matching_cascade(self, tracks, groups):
        if not tracks or not groups:
            return [], [t.global_id for t in tracks], list(range(len(groups)))
        
        unmatched_groups_indices = list(range(len(groups)))
        unmatched_tracks = tracks[:]
        matches = []
    
        # Cascade 0: Tracks en transition (priorité maximale)
        transitioning_tracks = [
            t for t in tracks 
            if (t.single_view_streak > 0 and t.single_view_streak < 15) or
               (len(set(list(t.multi_view_history)[-5:])) > 1 if t.multi_view_history else False)
        ]
        
        if transitioning_tracks and unmatched_groups_indices:
            cost_matrix = self._compute_cost_matrix(
                transitioning_tracks, 
                [groups[i] for i in unmatched_groups_indices]
            )
            
            if cost_matrix.size > 0:
                # Réduire les coûts pour favoriser le matching
                cost_matrix[cost_matrix < INFTY_COST] *= 0.9
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                newly_matched_track_ids, newly_matched_group_global_indices = set(), set()
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < INFTY_COST:
                        track = transitioning_tracks[r]
                        group_global_idx = unmatched_groups_indices[c]
                        matches.append((track.global_id, group_global_idx))
                        newly_matched_track_ids.add(track.global_id)
                        newly_matched_group_global_indices.add(group_global_idx)
                
                unmatched_tracks = [t for t in unmatched_tracks if t.global_id not in newly_matched_track_ids]
                unmatched_groups_indices = [i for i in unmatched_groups_indices if i not in newly_matched_group_global_indices]
    
        # Cascade 1: Tracks confirmés et stables
        confirmed_tracks = [
            t for t in unmatched_tracks 
            if t.state in [TrackState.Confirmed3D, TrackState.Confirmed2D] and 
            t.time_since_update < 5 and not t.is_occluded_by
        ]
        
        if confirmed_tracks and unmatched_groups_indices:
            cost_matrix = self._compute_cost_matrix(confirmed_tracks, [groups[i] for i in unmatched_groups_indices])
            
            if self.debug and cost_matrix.size > 0:
                self.debugger.save_cost_matrix(cost_matrix, [t.global_id for t in confirmed_tracks], 
                                              unmatched_groups_indices, self.frame_count, "confirmed")
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            newly_matched_track_ids, newly_matched_group_global_indices = set(), set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < INFTY_COST:
                    track = confirmed_tracks[r]
                    group_global_idx = unmatched_groups_indices[c]
                    matches.append((track.global_id, group_global_idx))
                    newly_matched_track_ids.add(track.global_id)
                    newly_matched_group_global_indices.add(group_global_idx)
    
            unmatched_tracks = [t for t in unmatched_tracks if t.global_id not in newly_matched_track_ids]
            unmatched_groups_indices = [i for i in unmatched_groups_indices if i not in newly_matched_group_global_indices]
    
        # Cascade 2: Tracks occlus et moins certains
        if unmatched_tracks and unmatched_groups_indices:
            cost_matrix = self._compute_cost_matrix(unmatched_tracks, [groups[i] for i in unmatched_groups_indices])
            
            if self.debug and cost_matrix.size > 0:
                self.debugger.save_cost_matrix(cost_matrix, [t.global_id for t in unmatched_tracks], 
                                              unmatched_groups_indices, self.frame_count, "occluded_tentative")
    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            newly_matched_track_ids, newly_matched_group_global_indices = set(), set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < INFTY_COST:
                    track = unmatched_tracks[r]
                    group_global_idx = unmatched_groups_indices[c]
                    matches.append((track.global_id, group_global_idx))
                    newly_matched_track_ids.add(track.global_id)
                    newly_matched_group_global_indices.add(group_global_idx)
            
            unmatched_tracks = [t for t in unmatched_tracks if t.global_id not in newly_matched_track_ids]
            unmatched_groups_indices = [i for i in unmatched_groups_indices if i not in newly_matched_group_global_indices]
    
        # Cascade 3: Détections partielles
        if unmatched_groups_indices:
            partial_groups_indices = []
            for idx in unmatched_groups_indices:
                visibility = self._estimate_group_visibility_ratio(groups[idx])
                if visibility < 0.4:
                    partial_groups_indices.append(idx)
            
            if partial_groups_indices and unmatched_tracks:
                partial_cost_matrix = self._compute_cost_matrix(
                    unmatched_tracks, 
                    [groups[i] for i in partial_groups_indices]
                )
                
                partial_cost_matrix[partial_cost_matrix < INFTY_COST] *= 0.8
                
                row_ind, col_ind = linear_sum_assignment(partial_cost_matrix)
                
                newly_matched_track_ids, newly_matched_group_global_indices = set(), set()
                for r, c in zip(row_ind, col_ind):
                    if partial_cost_matrix[r, c] < INFTY_COST:
                        track = unmatched_tracks[r]
                        group_global_idx = partial_groups_indices[c]
                        matches.append((track.global_id, group_global_idx))
                        newly_matched_track_ids.add(track.global_id)
                        newly_matched_group_global_indices.add(group_global_idx)
                
                unmatched_tracks = [t for t in unmatched_tracks if t.global_id not in newly_matched_track_ids]
                unmatched_groups_indices = [i for i in unmatched_groups_indices if i not in newly_matched_group_global_indices]
    
        final_unmatched_track_ids = [t.global_id for t in unmatched_tracks]
        return matches, final_unmatched_track_ids, unmatched_groups_indices
    
    def _merge_duplicate_tracks(self):
        """Fusionne les pistes qui représentent manifestement la même personne"""
        tracks_to_merge = []
        confirmed_tracks = [t for t in self.global_tracks.values() 
                           if t.state in [TrackState.Confirmed3D] and t.mean_3d is not None]
        
        for i in range(len(confirmed_tracks)):
            for j in range(i+1, len(confirmed_tracks)):
                t1, t2 = confirmed_tracks[i], confirmed_tracks[j]
                
                # Vérifier s'ils n'ont jamais de détections simultanées dans la même caméra
                cameras_overlap = False
                for cam_id in range(len(self.camera_configs)):
                    if cam_id in t1.current_bbox and cam_id in t2.current_bbox:
                        cameras_overlap = True
                        break
                
                if not cameras_overlap:
                    # Vérifier la proximité 3D
                    dist_3d = np.linalg.norm(t1.mean_3d[:3] - t2.mean_3d[:3])
                    if dist_3d < 600.0:
                        # Vérifier la similarité d'apparence
                        if t1.feature_gallery and t2.feature_gallery:
                            feat1 = np.mean(list(t1.feature_gallery), axis=0)
                            feat2 = np.mean(list(t2.feature_gallery), axis=0)
                            app_dist = cdist([feat1], [feat2], metric='cosine')[0, 0]
                            
                            if app_dist < 0.3:
                                tracks_to_merge.append((t1.global_id, t2.global_id, dist_3d, app_dist))
        
        # Fusionner les pistes
        for tid1, tid2, dist, app in tracks_to_merge:
            if tid1 in self.global_tracks and tid2 in self.global_tracks:
                t1, t2 = self.global_tracks[tid1], self.global_tracks[tid2]
                # Garder la piste la plus ancienne
                if t1.first_seen_frame <= t2.first_seen_frame:
                    keep, remove = t1, t2
                    keep_id, remove_id = tid1, tid2
                else:
                    keep, remove = t2, t1
                    keep_id, remove_id = tid2, tid1
                
                # Transférer les informations
                keep.current_bbox.update(remove.current_bbox)
                keep.current_pose_2d.update(remove.current_pose_2d)
                keep.feature_gallery.extend(remove.feature_gallery)
                
                print(f" Fusion: Track {remove_id} → Track {keep_id} (dist:{dist:.0f}mm, app:{app:.2f})")
                del self.global_tracks[remove_id]

    
    def _create_track(self, frame_idx: int, group: List[Dict]):
        group_features_list = [d["features"] for d in group if 'features' in d and d['features'] is not None]
        if not group_features_list:
            return None
        group_features = np.mean(group_features_list, axis=0)
        avg_visibility = self._estimate_group_visibility_ratio(group)
        
        group_3d_info_new = self._triangulate_group(group)

        if group_3d_info_new:
            pos_3d_new, reproj_conf, _ = group_3d_info_new
            
            # Parcourir les pistes déjà mises à jour dans cette frame
            already_updated_tracks = [t for t in self.global_tracks.values() if t.time_since_update == 0 and t.has_3d]
            
            for track in already_updated_tracks:
                # Si une piste déjà active et mise à jour est très proche de ce nouveau groupe...
                dist_3d = np.linalg.norm(pos_3d_new - track.mean_3d[:3])
                
                # Seuil de proximité très strict, car la piste est active
                if dist_3d < 600.0: # 60cm
                    
                    # ...et que l'apparence n'est pas totalement différente...
                    group_features = np.mean([d["features"] for d in group if 'features' in d], axis=0)
                    gallery = np.array(list(track.feature_gallery))
                    app_dist = float(np.min(cdist([group_features], gallery, metric='cosine')))
    
                    if app_dist < 0.5: # Seuil d'apparence assez lâche
                        print(f" Prévention de doublon: Le nouveau groupe est trop proche (dist={dist_3d:.0f}mm, app={app_dist:.2f}) de la piste active ID {track.global_id}. Annulation de la création.")
                        
                        # On pourrait fusionner les informations du groupe dans la piste existante
                        # Par exemple, ajouter les détections 2D du groupe à la piste pour cette frame.
                        # for d in group:
                        #     if d['camera_id'] not in track.current_bbox:
                        #         self._update_pose_data_in_track(track, d)
    
                        return None # Empêche la création de la nouvelle piste
            

        # NOUVEAU : Vérification stricte pour les premières frames
        if self.frame_count < 10:  # Premières 10 frames
            # Exiger une visibilité minimale plus élevée
            if avg_visibility < 0.6:  # 60% des joints visibles minimum
                #print(f" Frame {frame_idx}: Détection trop partielle ignorée (vis={avg_visibility:.2f})")
                return None
            
            # Pour une seule vue, exiger une détection complète
            if len(group) == 1:
                for d in group:
                    bbox = d['bbox']
                    bbox_height = bbox[3] - bbox[1]
                    bbox_width = bbox[2] - bbox[0]
                    
                    # Vérifier que la bbox est suffisamment grande (pas juste un pied/tête)
                    if bbox_height < 100 or bbox_width < 40:
                        print(f" Frame {frame_idx}: BBox trop petite ignorée ({bbox_width}x{bbox_height})")
                        return None
                    
                    # Vérifier la confiance de détection
                    if d.get('confidence', 0) < 0.7:  # Seuil plus strict au début
                        print(f"Frame {frame_idx}: Confiance trop faible ({d.get('confidence', 0):.2f})")
                        return None        
        
        # Vérifier les tracks récemment passées en vue unique
        single_view_tracks = [
            t for t in self.global_tracks.values()
            if t.single_view_streak > 0 and t.state != TrackState.Lost
        ]
        
        for track in single_view_tracks:
            if len(group) == 1:  # Nouvelle détection aussi en vue unique
                if track.feature_gallery and group_features_list:
                    gallery = np.array(list(track.feature_gallery))
                    app_dist = float(np.min(cdist([group_features], gallery, metric='cosine')))
                    
                    if app_dist < 0.35:
                        # Vérifier la cohérence spatiale 2D dans la même caméra
                        for d in group:
                            cam_id = d['camera_id']
                            # Chercher la dernière position connue dans cette caméra
                            if cam_id in track.current_bbox or (track.kf_mean is not None):
                                if cam_id in track.current_bbox:
                                    prev_bbox = track.current_bbox[cam_id]
                                else:
                                    # Utiliser la prédiction Kalman 2D
                                    pred_bbox = kalman_to_bbox_format(track.kf_mean)
                                    prev_bbox = pred_bbox
                                
                                curr_bbox = d['bbox']
                                center_prev = [(prev_bbox[0]+prev_bbox[2])/2, (prev_bbox[1]+prev_bbox[3])/2]
                                center_curr = [(curr_bbox[0]+curr_bbox[2])/2, (curr_bbox[1]+curr_bbox[3])/2]
                                dist_2d = np.linalg.norm(np.array(center_curr) - np.array(center_prev))
                                
                                # Tolérance basée sur le temps depuis dernière vue
                                max_dist = 200 + track.time_since_update * 30
                                if dist_2d < max_dist:
                                    print(f" Réassociation Track {track.global_id} en vue unique (app:{app_dist:.2f}, dist2D:{dist_2d:.0f})")
                                    self._update_matched_track(track, group, frame_idx)
                                    return track
        
        # Vérifier les tracks récemment perdues ou occludées
        recent_lost_tracks = [
            t for t in self.global_tracks.values() 
            if (t.state == TrackState.Occluded or 
                (t.state == TrackState.Lost and t.time_since_update < self.params['max_occlusion_frames']))
        ]
    
        if recent_lost_tracks and group_3d_info_new:
            pos_3d_new = group_3d_info_new[0]
            
            candidates = []
            for track in recent_lost_tracks:
                if track.mean_3d is not None:
                    predicted_pos_3d = track.mean_3d[:3]
                    dist_3d = np.linalg.norm(pos_3d_new - predicted_pos_3d)
    
                    # Seuil adaptatif selon le temps depuis dernière vue
                    thresh = self.params['occlusion_reid_threshold_mm'] + track.time_since_update * 50
                    
                    if dist_3d < thresh:
                        if track.feature_gallery:
                            gallery = np.array(list(track.feature_gallery))
                            app_dist = float(np.min(cdist([group_features], gallery, metric='cosine')))
                            if app_dist < self.params['reactivation_appearance_threshold']:
                                score = app_dist + (dist_3d / thresh)
                                candidates.append((score, track))
            
            if candidates:
                best_score, best_match_track = min(candidates, key=lambda x: x[0])
                if best_score < 0.8:
                    print(f" RE-IDENTIFIED Track {best_match_track.global_id} (score: {best_score:.2f})")
                    self._update_matched_track(best_match_track, group, frame_idx)
                    return best_match_track
        
        # Vérifier les tracks actives pour éviter les doublons
        if group_3d_info_new:
            pos_3d, _, _ = group_3d_info_new
            group_cameras = {d['camera_id'] for d in group}
            
            for track in self.global_tracks.values():
                if track.state in [TrackState.Confirmed3D, TrackState.Confirmed2D, TrackState.Occluded]:
                    if track.mean_3d is not None:
                        predicted_pos = track.mean_3d[:3] if track.state != TrackState.Occluded else track.last_known_3d_pos
                        if predicted_pos is not None:
                            dist_3d = np.linalg.norm(pos_3d - predicted_pos)
                            
                            # Seuil adaptatif
                            threshold = 1200.0 if track.state == TrackState.Occluded else 800.0
                            threshold += track.time_since_update * 50
                            
                            if dist_3d < threshold:
                                if track.feature_gallery and group_features_list:
                                    gallery = np.array(list(track.feature_gallery))
                                    app_dist = float(np.min(cdist([group_features], gallery, metric='cosine')))
                                    
                                    app_threshold = 0.5 if avg_visibility < 0.3 else 0.35
                                    
                                    if app_dist < app_threshold:
                                        print(f" Associé groupe à Track {track.global_id} (dist3D={dist_3d:.0f}mm, app={app_dist:.2f})")
                                        self._update_matched_track(track, group, frame_idx)
                                        return track
        
        # Vérifications de qualité avant création
        if avg_visibility < self.params['min_visibility_for_new_track']:
            return None
        
        if np.mean([d.get('confidence', 0.0) for d in group]) < self.params['min_confidence_for_new_track']:
            return None
        
        # Créer nouvelle track
        gid = self.next_global_id
        self.next_global_id += 1
        track = GlobalTrack(global_id=gid, first_seen_frame=frame_idx, last_seen_frame=frame_idx)
        
        avg_bbox = np.mean([d['bbox'] for d in group], axis=0)
        track.kf_mean, track.kf_covariance = self.kf.initiate(bbox_to_kalman_format(avg_bbox))
        track.update_features(group_features)
        
        if group_3d_info_new:
            pos_3d, _, _ = group_3d_info_new
            track.mean_3d, track.covariance_3d = track.kf_3d.initiate(pos_3d)
            track.last_known_3d_pos = pos_3d
            track.last_triangulated_frame = frame_idx
        
        # Initialiser l'historique de vues
        track.multi_view_history.append(len(group))
        track.single_view_streak = 0 if len(group) > 1 else 1
        
        self.global_tracks[gid] = track
        self._update_matched_track(track, group, frame_idx)
        track.state = TrackState.Tentative
        
        print(f" CREATED Track {gid} (vues: {len(group)})")
        return track

    def project_3d_to_2d(self, pts3d: np.ndarray, P: np.ndarray) -> np.ndarray:
        pts_h = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
        pts2d_h = (P @ pts_h.T).T
        return pts2d_h[:, :2] / (pts2d_h[:, 2:3] + 1e-9)

    def _triangulate_and_draw(self, frames: List[np.ndarray], writers: List[cv2.VideoWriter]):
        for track in self.global_tracks.values():
            track.current_pose_3d, track.pose_3d_confidence = None, 0.0
            if track.state not in [TrackState.Confirmed3D, TrackState.Confirmed2D] or len(track.current_pose_2d) < self.params['min_views_for_3d']: continue
            
            cam_ids, poses2d, joint_confs, projs = [], [], [], []
            for cid in sorted(track.current_pose_2d.keys()):
                #cam_ids.append(cid); poses2d.append(track.current_pose_2d[cid])
                #joint_confs.append(track.current_pose_2d_joint_confidences[cid]); projs.append(self.projection_matrices[cid])
                cam_ids.append(cid); poses2d.append(track.current_pose_2d[cid])
                joint_confs.append(track.current_pose_2d_joint_confidences[cid]); projs.append(self.P_by_camid[cid])


            num_joints = poses2d[0].shape[0]
            pose3d, confs3d = np.full((num_joints, 3), np.nan), np.zeros(num_joints)
            for j in range(num_joints):
                pts, prjs, wgts, cids_joint = [], [], [], []
                for i, cid in enumerate(cam_ids):
                    if j < len(joint_confs[i]) and joint_confs[i][j] >= self.params['min_joint_confidence_for_triangulation']:
                        pts.append(poses2d[i][j]); prjs.append(projs[i]); wgts.append(joint_confs[i][j]); cids_joint.append(cid)
                if len(pts) >= self.params['min_views_for_3d']:
                    try:
                        p3d = weighted_triangulate_points(prjs, pts, wgts)
                        if sum(check_cheirality(p3d, self.cam_params_by_camid[cid]["R"], self.cam_params_by_camid[cid]["t"]) for cid in cids_joint) >= self.params['min_views_for_3d']:

                            errors = [np.linalg.norm(self.project_3d_to_2d(np.array([p3d]), p)[0] - pt) for p, pt in zip(prjs, pts)]
                            if (mean_err := np.mean(errors)) < self.params['reprojection_error_thresh']:
                                pose3d[j], confs3d[j] = p3d, 1.0 - mean_err / self.params['reprojection_error_thresh']
                    except (np.linalg.LinAlgError, ValueError): pass
            valid_mask = ~np.isnan(pose3d[:, 0])
            if np.sum(valid_mask) >= self.params['min_valid_joints']:
                mean_conf = np.mean(confs3d[valid_mask])
                if mean_conf > self.params['pose_confidence_thresh']:
                    track.current_pose_3d, track.pose_3d_confidence = pose3d, float(mean_conf)
        
        self._write_csv_logs()
        self._write_training_data_log()
        num_confirmed = sum(1 for t in self.global_tracks.values() if t.state in [TrackState.Confirmed2D, TrackState.Confirmed3D])
        
        for cam_id, (frame, writer) in enumerate(zip(frames, writers)):
            vis = frame.copy()
            cv2.putText(vis, f"Frame: {self.frame_count} | Tracks: {num_confirmed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for track in self.global_tracks.values():
                if track.state == TrackState.Lost: continue

                if track.state == TrackState.Occluded and track.mean_3d is not None:
                    predicted_pos_3d = track.mean_3d[:3]
                    #proj_2d = self.project_3d_to_2d(np.array([predicted_pos_3d]), self.projection_matrices[cam_id])[0]
                    proj_2d = self.project_3d_to_2d(np.array([predicted_pos_3d]), self.P_by_camid[cam_id])[0]

                    #  Vérifier que la prédiction est plausible
                    if 0 < proj_2d[0] < vis.shape[1] and 0 < proj_2d[1] < vis.shape[0]:
                        # Vérifier la cohérence temporelle de la prédiction
                        if track.time_since_update < 30:  # Seulement si récemment vu
                            # Vérifier que la covariance n'a pas trop divergé
                            max_uncertainty = np.max(np.diag(track.covariance_3d[:3, :3]))
                            if max_uncertainty < 100000:  # Seuil d'incertitude max
                                cv2.circle(vis, tuple(proj_2d.astype(int)), 15, (255, 100, 100), -1) 
                                cv2.putText(vis, f"ID:{track.global_id}?", (int(proj_2d[0])+20, int(proj_2d[1])+5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)

                if not track.current_bbox: continue
                
                color = tuple(int(c) for c in track.color)
                label = f"ID:{track.global_id}"

                if track.state == TrackState.Occluded:
                    color = (128, 128, 128)
                    label += f" [OCC]"
                elif track.is_occluded_by is not None:
                    color = (0, 0, 255)
                    label += f" [OCC by {track.is_occluded_by}]"
                elif track.last_visible_ratio < self.params['low_visibility_threshold']:
                    color = (0, 165, 255)
                    label += f" [VIS:{track.last_visible_ratio:.2f}]"
                elif track.state == TrackState.Tentative:
                    color = (255, 255, 0)
                
                if cam_id in track.current_bbox:
                    x1, y1, x2, y2 = map(int, track.current_bbox[cam_id])
                    
                    if cam_id in track.current_mask:
                        mask = track.current_mask[cam_id]
                        if mask.shape != vis.shape[:2]:
                           mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        colored_mask = np.zeros_like(vis)
                        colored_mask[mask > 0.5] = color
                        vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.4, 0)

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    
                    cv2.putText(vis, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if cam_id in track.current_pose_2d:
                        kpts = track.current_pose_2d[cam_id].copy()
                        kpts[track.current_pose_2d_joint_confidences[cam_id] < 0.3] = np.nan
                        self._draw_skeleton_custom(vis, kpts, self.joint_edges, (0, 255, 0), 2)
                
                if track.current_pose_3d is not None and np.any(~np.isnan(track.current_pose_3d[:, 0])):
                    valid_mask = ~np.isnan(track.current_pose_3d[:, 0])
                    #proj2d = self.project_3d_to_2d(track.current_pose_3d[valid_mask], self.projection_matrices[cam_id])
                    proj2d = self.project_3d_to_2d(track.current_pose_3d[valid_mask], self.P_by_camid[cam_id])

                    full_proj_pts = np.full_like(track.current_pose_3d[:, :2], np.nan)
                    full_proj_pts[valid_mask] = proj2d
                    self._draw_skeleton_custom(vis, full_proj_pts, self.joint_edges, (255, 0, 0), 3)
            writer.write(cv2.resize(vis, self.save_resolution))


    def _make_writer(self,path, fps, size):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not w.isOpened():
            # Fallback 1: H.264 (si FFmpeg + libx264)
            alt_path = os.path.splitext(path)[0] + "_avc1.mp4"
            w = cv2.VideoWriter(alt_path, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        if not w.isOpened():
            # Fallback 2: MJPG/AVI (quasi universel)
            alt_path = os.path.splitext(path)[0] + "_mjpg.avi"
            w = cv2.VideoWriter(alt_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        return w


    def init_csv(self, filename: str = None):
        filename = filename or f"poses_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if self.joint_names:
            self.csv_writer.writerow(['frame', 'person_id'] + [f'{jn}_{ax}' for jn in self.joint_names for ax in 'xyz'])
            print(f" CSV créé: {filename}")


# Fonction pour initialiser le fichier CSV d'entraînement
    def init_training_csv(self, filename: str):
        """Initialise le fichier CSV pour les données d'entraînement."""
        self.csv_file_train = open(filename, 'w', newline='')
        self.csv_writer_train = csv.writer(self.csv_file_train)
        header = [
            'frame_id', 'person_id', 'camera_id', 
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'nose_x', 'nose_y', 'nose_z'
        ]
        self.csv_writer_train.writerow(header)
        print(f" Fichier CSV d'entraînement créé : {filename}")

    def _write_csv_logs(self):
        if self.csv_writer:
            for track in self.global_tracks.values():
                if track.state == TrackState.Confirmed3D and track.current_pose_3d is not None:
                    row = [self.frame_count, track.global_id] + track.current_pose_3d.flatten().tolist()
                    self.csv_writer.writerow(row)
            if self.csv_file: self.csv_file.flush()

#  Fonction pour écrire les logs dans le fichier d'entraînement
    def _write_training_data_log(self):
        """Écrit les données pour l'entraînement dans le second fichier CSV."""
        if not self.csv_writer_train:
            return

        for track in self.global_tracks.values():
            # On ne sauvegarde que les tracks confirmées avec une pose 3D valide
            if track.state == TrackState.Confirmed3D and track.current_pose_3d is not None:
                # Le point de référence 3D est le nez (index 0)
                nose_3d = track.current_pose_3d[0]
                
                # Vérifier si la position du nez est valide (non NaN)
                if not np.any(np.isnan(nose_3d)):
                    # Pour chaque caméra où cette personne a été vue cette frame...
                    for cam_id, bbox in track.current_bbox.items():
                        x1, y1, x2, y2 = bbox
                        row = [
                            self.frame_count,
                            track.global_id,
                            cam_id,
                            int(x1), int(y1), int(x2), int(y2),
                            *nose_3d.tolist() # Dépaqueter les coordonnées x, y, z
                        ]
                        self.csv_writer_train.writerow(row)
        
        if self.csv_file_train:
            self.csv_file_train.flush()
    
    def run(self, save_lab_csv=True, lab_csv_filename=None, output_dir=None):
        if not self.P_by_camid:
            print("Calibration manquante.")
            return
    
        # Ordre stable des caméras
        cam_ids = sorted(self.camera_configs.keys())
    
        # Ouvrir les vidéos
        caps = [cv2.VideoCapture(self.camera_configs[cid]["source"]) for cid in cam_ids]
        if not all(c.isOpened() for c in caps):
            print("Impossible d'ouvrir les vidéos.")
            for c in caps: c.release()
            return
    
        # FPS robustes
        fps0 = caps[0].get(cv2.CAP_PROP_FPS)
        if not fps0 or fps0 != fps0:  # NaN
            fps0 = 30
        self.video_fps = int(round(fps0))
    
        # Nombre total de frames (si 0, on sort proprement)
        frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        total_frames = min(frame_counts) if all(fc > 0 for fc in frame_counts) else 0
        if total_frames <= 0:
            print("Impossible de lire le nombre total de frames.")
            for c in caps: c.release()
            return
    
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
        # activity_name pour logs
        first_src = self.camera_configs[cam_ids[0]]["source"]
        base_name_first = os.path.splitext(os.path.basename(first_src))[0]
        # ex: "activity.CAM03" -> "activity"
        activity_name = "_".join(base_name_first.split(".")[:-2]) or base_name_first
    
        # Writers (un par caméra) avec fallback de codec
        writers = []
        for cid in cam_ids:
            src = self.camera_configs[cid]["source"]
            base_name = os.path.splitext(os.path.basename(src))[0]
            output_base = f"{base_name}_tracked_v68"
            output_path = os.path.join(output_dir or "", output_base + ".mp4")
            w = self._make_writer(output_path, self.video_fps, self.save_resolution)
            if not w.isOpened():
                print(f"Impossible d'ouvrir un VideoWriter pour {output_path}. Vérifie ta stack FFmpeg/codecs.")
                for c in caps: c.release()
                return
            print(f" Écriture vers: {output_path}")
            writers.append(w)
    
        if save_lab_csv:
            self.init_csv(lab_csv_filename)
            # NOUVEAU : Initialiser le second CSV pour l'entraînement
            first_src = self.camera_configs[sorted(self.camera_configs.keys())[0]]["source"]
            activity_name = "_".join(os.path.basename(first_src).split(".")[:-2])
            training_csv_filename = os.path.join(output_dir or "", f"{activity_name}_training_data.csv")
            self.init_training_csv(training_csv_filename)
    
        print(f"Traitement de {total_frames} frames avec le tracker v68.0...")
        try:
            for self.frame_count in tqdm(range(total_frames), desc=f"Processing {activity_name}"):

                # --- ÉTAPE 4: Lire les images dans ce même ordre ---
                frames = []
                for i, cap in enumerate(caps):
                    ret, f = cap.read()
                    if not ret or f is None:
                        print(f"\n Frame manquante de la caméra {cam_ids[i]} à l'index {self.frame_count}. Arrêt.")
                        raise StopIteration
                    # UNDISTORT: Redresser la vidéo (identique à project_markers_c3d.py)
                    if cam_ids[i] in self.undistort_maps:
                        map_x, map_y = self.undistort_maps[cam_ids[i]]
                        f = cv2.remap(f, map_x, map_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    frames.append(f)
    
                # Détection + MAJ tracks
                dets_per_cam = {i: self._process_single_view(frames[i], cam_ids[i]) for i in range(len(cam_ids))}
                self._update_tracks(self.frame_count, dets_per_cam, frames)
    
                # Dessin + écriture (garanti d’écrire même en cas d’erreur de dessin)
                try:
                    self._triangulate_and_draw(frames, writers)
                except Exception as e:
                    if self.debug:
                        print(f" _triangulate_and_draw a échoué: {e}")
                    # Écrit quand même les frames brutes pour ne pas vider la vidéo
                    for i, w in enumerate(writers):
                        vis = cv2.resize(frames[i], self.save_resolution)
                        w.write(vis)
    
        except KeyboardInterrupt:
            print("\n Arrêt demandé")
        finally:
            for c in caps: c.release()
            for w in writers: w.release()
            if self.csv_file:
                self.csv_file.close()
            if self.csv_file_train:
                self.csv_file_train.close()
            print(f" Traitement de l'activité {activity_name} terminé!")


def main():
    BASE_VIDEOS_PATH = r"\\intram\Paris\Labo\IBHGC\PROJETS\metasport\DATA\METASPORT11\Session1\Files\fichiers_video\Bruts" 
    BASE_OUTPUT_PATH = "./results/videos_V2_detect/ResultsMETASPORT11Rtmlib_V3" 
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    print(f"Dossier de sortie principal: {BASE_OUTPUT_PATH}")
    
    tracker = RobustMultiViewTracker(
        model_path="yolo26x.pt",
        feature_model_path="./pretrained_models/osnet_ain_x1_0_M.onnx",
        rtmlib_device='cuda',
        rtmlib_backend='onnxruntime',
        save_res=(1280, 720),
        debug=False,
        confidence_thresh=0.35,
        association_thresh=0.7,
        min_confidence_for_new_track=0.6,
        reactivation_appearance_threshold=0.30,
        min_views_for_3d=2,
        reprojection_error_thresh=120,
        min_valid_joints=4,

        occlusion_patience=50,
        occlusion_xy_dist_thresh=750.0,

        SPATIAL_CONSISTENCY_THRESH_MM=1500.0,  # Augmenter pour mouvements rapides
        min_joint_confidence_for_triangulation=0.25,  # Réduire pour plus de flexibilité
        track_patience=150,  # Réduire pour nettoyer plus vite
    )

    xcp_files = glob.glob(os.path.join(BASE_VIDEOS_PATH, "*.xcp"))
    activity_to_xcp = {os.path.basename(f).replace('.xcp', ''): f for f in xcp_files}
    for activity_name, xcp_file_path in activity_to_xcp.items():
        # Vérifier si l'activité est déjà traitée
        expected_csv = os.path.join(BASE_OUTPUT_PATH, f"{activity_name}_poses_3d_rtmlib_v68_0.csv")
        if os.path.exists(expected_csv):
            print(f"\n {activity_name} déjà traité, passage à la suivante.")
            continue
        print("\n" + "=" * 80)
        print(f" Démarrage du traitement pour l'activité : {activity_name}")
        print(f"  Fichier de calibration utilisé : {xcp_file_path}")
        tracker.reset_for_new_activity()
        video_files = sorted(glob.glob(os.path.join(BASE_VIDEOS_PATH, f"{activity_name}.*.avi")))
        if len(video_files) < 2:
            print(f"Pas assez de fichiers vidéo. Passage à la suivante.")
            continue
        for i, fpath in enumerate(video_files):
            tracker.add_camera(i, fpath, f"Cam_{os.path.basename(fpath).split('.')[1]}")
        activity_output_dir = os.path.join(BASE_OUTPUT_PATH)
        os.makedirs(activity_output_dir, exist_ok=True)
        if tracker.initialize(xcp_file=xcp_file_path, force_variant="2B"):
            lab_csv_filename = os.path.join(activity_output_dir, f"{activity_name}_poses_3d_rtmlib_v68_0.csv")
            tracker.run(
                save_lab_csv=True, lab_csv_filename=lab_csv_filename,
                output_dir=activity_output_dir
            )
        else:
            print(f"Échec de l'initialisation pour {activity_name}. Passage à la suivante.")
    print("\n" + "=" * 80)
    print("Toutes les activités ont été traitées.")


if __name__ == "__main__":
    main()

