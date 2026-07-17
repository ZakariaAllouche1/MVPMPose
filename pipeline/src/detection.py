from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

import albumentations as A
from albumentations.pytorch import ToTensorV2
from rtmlib import BodyWithFeet


BoxXYXY = Tuple[int, int, int, int]


@dataclass
class ViewDetection:
    camera_name: str
    bbox_xyxy: BoxXYXY
    confidence: float
    tracklet_id: Optional[int]
    feature: Optional[np.ndarray]
    pose_2d: Optional[np.ndarray]
    pose_confidence: float
    joint_confidences: Optional[np.ndarray]

    # Qualité géométrique de la bbox dans l'image
    image_width: int = 0
    image_height: int = 0
    bbox_width_ratio: float = 0.0
    bbox_height_ratio: float = 0.0
    bbox_area_ratio: float = 0.0
    touches_border: bool = False
    edge_truncated: bool = False
    valid_joint_count: int = 0
    valid_joint_ratio: float = 0.0
    reliable_for_3d: bool = False
    unreliable_reason: str = ""

    body_completeness_score: float = 0.0
    partial_body: bool = False
    upper_body_visible: bool = False
    lower_body_visible: bool = False
    torso_visible: bool = False

    @property
    def bbox_tlbr(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return y1, x1, y2, x2


class ReIdFeatureExtractor:
    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Modèle ReID introuvable: {model_path}")

        self.session = ort.InferenceSession(str(model_path))
        input_shape = self.session.get_inputs()[0].shape

        if len(input_shape) != 4:
            raise ValueError(f"Shape ONNX inattendue pour ReID: {input_shape}")

        height, width = input_shape[2], input_shape[3]

        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                f"Le modèle ReID doit avoir une taille fixe, reçu: {input_shape}"
            )

        self.input_name = self.session.get_inputs()[0].name
        self.transform = A.Compose([
            A.Resize(height, width),
            A.Normalize(),
            ToTensorV2(),
        ])

    def predict(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        if image_bgr is None or image_bgr.size == 0:
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=image_rgb)["image"]
        tensor = np.expand_dims(tensor, axis=0)

        feature = self.session.run(None, {self.input_name: tensor})[0]

        if feature.ndim != 2 or feature.shape[0] != 1:
            return None

        feature = feature[0].astype(np.float32)
        norm = np.linalg.norm(feature)

        if norm <= 1e-12:
            return None

        return feature / norm


def clip_bbox_xyxy(
    bbox: BoxXYXY,
    image_shape: Tuple[int, int, int],
) -> Optional[BoxXYXY]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def bbox_center(bbox: BoxXYXY) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array(
        [
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
        ],
        dtype=np.float32,
    )


def valid_pose_center(
    pose_2d: np.ndarray,
    joint_confidences: np.ndarray,
    min_joint_confidence: float,
    min_valid_joints: int,
) -> Optional[np.ndarray]:
    if pose_2d is None or joint_confidences is None:
        return None

    if len(pose_2d) == 0 or len(joint_confidences) == 0:
        return None

    valid = (
        np.isfinite(pose_2d[:, 0])
        & np.isfinite(pose_2d[:, 1])
        & (joint_confidences > min_joint_confidence)
    )

    if int(np.sum(valid)) < min_valid_joints:
        return None

    return np.mean(pose_2d[valid], axis=0).astype(np.float32)


def point_inside_bbox(point: np.ndarray, bbox: BoxXYXY) -> bool:
    x1, y1, x2, y2 = bbox
    return bool(x1 <= point[0] <= x2 and y1 <= point[1] <= y2)


def get_torso_bbox(
    pose_2d: np.ndarray,
    joint_confidences: np.ndarray,
    image_shape: Tuple[int, int, int],
    min_confidence: float = 0.4,
) -> Optional[BoxXYXY]:
    if pose_2d is None or joint_confidences is None:
        return None

    torso_indices = [5, 6, 11, 12]
    valid_points = []

    for index in torso_indices:
        if index >= len(pose_2d) or index >= len(joint_confidences):
            continue

        if joint_confidences[index] <= min_confidence:
            continue

        point = pose_2d[index]

        if not np.isfinite(point).all():
            continue

        valid_points.append(point)

    if len(valid_points) < 3:
        return None

    points = np.asarray(valid_points, dtype=np.float32)

    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)

    margin_x = 0.15 * max(1.0, x2 - x1)
    margin_y = 0.15 * max(1.0, y2 - y1)

    bbox = (
        int(round(x1 - margin_x)),
        int(round(y1 - margin_y)),
        int(round(x2 + margin_x)),
        int(round(y2 + margin_y)),
    )

    return clip_bbox_xyxy(bbox, image_shape)


def shrink_bbox(
    bbox: BoxXYXY,
    image_shape: Tuple[int, int, int],
    margin_ratio: float,
) -> Optional[BoxXYXY]:
    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    margin_x = int(round(width * margin_ratio))
    margin_y = int(round(height * margin_ratio))

    shrunk = (
        x1 + margin_x,
        y1 + margin_y,
        x2 - margin_x,
        y2 - margin_y,
    )

    clipped = clip_bbox_xyxy(shrunk, image_shape)

    if clipped is not None:
        return clipped

    return clip_bbox_xyxy(bbox, image_shape)


def crop_image(image_bgr: np.ndarray, bbox: BoxXYXY) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image_bgr[y1:y2, x1:x2].copy()


def compute_bbox_image_quality(
    bbox: BoxXYXY,
    image_shape: Tuple[int, int, int],
    border_margin_px: int = 8,
) -> Dict[str, object]:
    """Calcule des indicateurs simples de tronquage/échelle bbox.
    """
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox

    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))

    touches_left = x1 <= border_margin_px
    touches_top = y1 <= border_margin_px
    touches_right = x2 >= width - border_margin_px
    touches_bottom = y2 >= height - border_margin_px

    touches_border = bool(touches_left or touches_top or touches_right or touches_bottom)

    return {
        "image_width": int(width),
        "image_height": int(height),
        "bbox_width_ratio": float(box_w / max(1.0, float(width))),
        "bbox_height_ratio": float(box_h / max(1.0, float(height))),
        "bbox_area_ratio": float((box_w * box_h) / max(1.0, float(width * height))),
        "touches_left": bool(touches_left),
        "touches_top": bool(touches_top),
        "touches_right": bool(touches_right),
        "touches_bottom": bool(touches_bottom),
        "touches_border": touches_border,
    }


def compute_joint_visibility(
    joint_confidences: Optional[np.ndarray],
    min_joint_confidence: float = 0.25,
) -> tuple[int, float]:
    if joint_confidences is None or len(joint_confidences) == 0:
        return 0, 0.0

    scores = np.asarray(joint_confidences, dtype=np.float32)
    finite = np.isfinite(scores)

    if not np.any(finite):
        return 0, 0.0

    valid = finite & (scores > float(min_joint_confidence))
    return int(np.sum(valid)), float(np.sum(valid) / max(1, len(scores)))




def compute_body_completeness(
    joint_confidences: Optional[np.ndarray],
    min_joint_confidence: float = 0.25,
) -> Dict[str, object]:
    """Estime si le corps est suffisamment complet pour une reconstruction 3D.
    """
    if joint_confidences is None or len(joint_confidences) == 0:
        return {
            "body_completeness_score": 0.0,
            "partial_body": True,
            "upper_body_visible": False,
            "lower_body_visible": False,
            "torso_visible": False,
        }

    scores = np.asarray(joint_confidences, dtype=np.float32)

    def count_valid(indices: List[int]) -> int:
        count = 0
        for index in indices:
            if index >= len(scores):
                continue
            value = float(scores[index])
            if np.isfinite(value) and value > float(min_joint_confidence):
                count += 1
        return count

    head = count_valid([0, 1, 2, 3, 4])
    shoulders = count_valid([5, 6])
    arms = count_valid([7, 8, 9, 10])
    hips = count_valid([11, 12])
    legs = count_valid([13, 14, 15, 16])

    torso_visible = shoulders >= 1 and hips >= 1
    upper_body_visible = shoulders >= 1 and (head >= 1 or arms >= 1)
    lower_body_visible = hips >= 1 and legs >= 2

    score_parts = [
        1.0 if torso_visible else 0.0,
        min(1.0, head / 2.0),
        min(1.0, shoulders / 2.0),
        min(1.0, arms / 2.0),
        min(1.0, hips / 2.0),
        min(1.0, legs / 2.0),
    ]
    completeness = float(np.mean(score_parts))

    # Une vue est considérée partielle si le torse manque, si le haut/bas du
    # corps est absent, ou si la complétude globale est trop faible
    partial_body = bool(
        not torso_visible
        or not upper_body_visible
        or not lower_body_visible
        or completeness < 0.52
    )

    return {
        "body_completeness_score": completeness,
        "partial_body": partial_body,
        "upper_body_visible": bool(upper_body_visible),
        "lower_body_visible": bool(lower_body_visible),
        "torso_visible": bool(torso_visible),
    }


def estimate_view_reliability_for_3d(
    bbox: BoxXYXY,
    image_shape: Tuple[int, int, int],
    pose_2d: Optional[np.ndarray],
    joint_confidences: Optional[np.ndarray],
    pose_confidence: float,
    min_joint_confidence: float = 0.25,
    min_valid_joints: int = 8,
    min_valid_joint_ratio: float = 0.38,
    min_pose_confidence: float = 0.20,
) -> Dict[str, object]:
    """Estime si une vue peut être utilisée pour la pose 3D learned.
    """
    quality = compute_bbox_image_quality(bbox, image_shape)
    valid_joint_count, valid_joint_ratio = compute_joint_visibility(
        joint_confidences,
        min_joint_confidence=min_joint_confidence,
    )
    body_quality = compute_body_completeness(
        joint_confidences,
        min_joint_confidence=min_joint_confidence,
    )

    has_pose = pose_2d is not None and joint_confidences is not None

    touches_left = bool(quality["touches_left"])
    touches_right = bool(quality["touches_right"])
    touches_top = bool(quality["touches_top"])
    touches_bottom = bool(quality["touches_bottom"])

    edge_truncated = bool(
        touches_left
        or touches_right
        or touches_top
        or (touches_bottom and valid_joint_ratio < 0.65)
    )

    reasons: List[str] = []

    if not has_pose:
        reasons.append("no_pose")

    if valid_joint_count < int(min_valid_joints):
        reasons.append("few_valid_joints")

    if valid_joint_ratio < float(min_valid_joint_ratio):
        reasons.append("low_joint_visibility")

    if float(pose_confidence) < float(min_pose_confidence):
        reasons.append("low_pose_confidence")

    if edge_truncated:
        reasons.append("edge_truncated")

    if bool(body_quality["partial_body"]):
        reasons.append("partial_body")

    if float(quality["bbox_height_ratio"]) < 0.08:
        reasons.append("bbox_too_short")

    if float(quality["bbox_width_ratio"]) < 0.025:
        reasons.append("bbox_too_narrow")

    reliable = len(reasons) == 0

    quality.update({
        "edge_truncated": bool(edge_truncated),
        "valid_joint_count": int(valid_joint_count),
        "valid_joint_ratio": float(valid_joint_ratio),
        "body_completeness_score": float(body_quality["body_completeness_score"]),
        "partial_body": bool(body_quality["partial_body"]),
        "upper_body_visible": bool(body_quality["upper_body_visible"]),
        "lower_body_visible": bool(body_quality["lower_body_visible"]),
        "torso_visible": bool(body_quality["torso_visible"]),
        "reliable_for_3d": bool(reliable),
        "unreliable_reason": "|".join(reasons),
    })

    return quality


def detection_bbox_area(detection: ViewDetection) -> float:
    x1, y1, x2, y2 = detection.bbox_xyxy
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def detection_intersection_area(a: ViewDetection, b: ViewDetection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    return float(max(0, ix2 - ix1) * max(0, iy2 - iy1))


def detection_iou(a: ViewDetection, b: ViewDetection) -> float:
    inter = detection_intersection_area(a, b)

    if inter <= 0:
        return 0.0

    area_a = detection_bbox_area(a)
    area_b = detection_bbox_area(b)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return float(inter / union)


def detection_containment(a: ViewDetection, b: ViewDetection) -> float:
    """pour détecter une bbox partielle incluse dans une bbox complète.
    """
    inter = detection_intersection_area(a, b)

    if inter <= 0:
        return 0.0

    area_a = detection_bbox_area(a)
    area_b = detection_bbox_area(b)
    small_area = max(1.0, min(area_a, area_b))

    return float(inter / small_area)


def detection_center_distance_ratio(a: ViewDetection, b: ViewDetection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy

    center_a = np.array(
        [0.5 * (ax1 + ax2), 0.5 * (ay1 + ay2)],
        dtype=np.float32,
    )
    center_b = np.array(
        [0.5 * (bx1 + bx2), 0.5 * (by1 + by2)],
        dtype=np.float32,
    )

    scale = np.sqrt(max(detection_bbox_area(a), detection_bbox_area(b), 1.0))
    return float(np.linalg.norm(center_a - center_b) / scale)


def cosine_distance_optional(
    feature_a: Optional[np.ndarray],
    feature_b: Optional[np.ndarray],
) -> Optional[float]:
    if feature_a is None or feature_b is None:
        return None

    a = np.asarray(feature_a, dtype=np.float32)
    b = np.asarray(feature_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return None

    a = a / norm_a
    b = b / norm_b

    return float(1.0 - np.dot(a, b))


def detection_valid_pose_center(detection: ViewDetection) -> Optional[np.ndarray]:
    if detection.pose_2d is None or detection.joint_confidences is None:
        return None

    return valid_pose_center(
        pose_2d=detection.pose_2d,
        joint_confidences=detection.joint_confidences,
        min_joint_confidence=0.25,
        min_valid_joints=4,
    )


def pose_centers_are_clearly_different(
    a: ViewDetection,
    b: ViewDetection,
    threshold_ratio: float = 0.35,
) -> bool:
    """
    True si les poses suggèrent clairement deux personnes différentes
    Dans ce cas, même avec beaucoup d'overlap, on garde les deux bbox
    """
    center_a = detection_valid_pose_center(a)
    center_b = detection_valid_pose_center(b)

    if center_a is None or center_b is None:
        return False

    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy

    scale = max(
        float(ax2 - ax1),
        float(ay2 - ay1),
        float(bx2 - bx1),
        float(by2 - by1),
        1.0,
    )

    distance = float(np.linalg.norm(center_a - center_b))
    return distance / scale > threshold_ratio


def detection_quality_score(detection: ViewDetection) -> float:
    """
    Score simple pour choisir quelle bbox garder quand deux bbox sont
    très probablement le même humain.
    """
    area = detection_bbox_area(detection)

    pose_bonus = 0.0
    if detection.pose_2d is not None:
        pose_bonus += 0.20

    visibility_bonus = 0.0
    if detection.joint_confidences is not None and len(detection.joint_confidences) > 0:
        visibility_bonus = 0.15 * float(np.mean(detection.joint_confidences > 0.25))

    area_bonus = 0.03 * float(np.log1p(area / 10000.0))

    return float(detection.confidence + pose_bonus + visibility_bonus + area_bonus)


def detection_size_similarity(a: ViewDetection, b: ViewDetection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox_xyxy
    bx1, by1, bx2, by2 = b.bbox_xyxy

    aw = max(1.0, float(ax2 - ax1))
    ah = max(1.0, float(ay2 - ay1))
    bw = max(1.0, float(bx2 - bx1))
    bh = max(1.0, float(by2 - by1))

    width_similarity = min(aw, bw) / max(aw, bw)
    height_similarity = min(ah, bh) / max(ah, bh)
    return float(min(width_similarity, height_similarity))


def should_suppress_as_duplicate(
    a: ViewDetection,
    b: ViewDetection,
) -> bool:
    """
    Retourne True seulement si a et b sont très probablement deux bbox
    de la même personne dans la même caméra.
    """
    iou = detection_iou(a, b)
    containment = detection_containment(a, b)
    center_ratio = detection_center_distance_ratio(a, b)
    size_similarity = detection_size_similarity(a, b)

    if iou >= 0.92 and center_ratio <= 0.08 and size_similarity >= 0.88:
        return True

    if iou >= 0.84 and center_ratio <= 0.05 and size_similarity >= 0.82:
        return True

    appearance_distance = cosine_distance_optional(a.feature, b.feature)

    if pose_centers_are_clearly_different(a, b):
        return False

    if appearance_distance is not None and appearance_distance > 0.30:
        return False

    if iou >= 0.85 and center_ratio <= 0.15 and size_similarity >= 0.75:
        return True

    if containment >= 0.88 and center_ratio <= 0.25:
        if appearance_distance is not None and appearance_distance < 0.18:
            return True

        if appearance_distance is None and containment >= 0.96 and center_ratio <= 0.15:
            return True

    return False


def suppress_duplicate_detections_same_camera(
    detections: List[ViewDetection],
) -> List[ViewDetection]:
    """
    Supprime les doublons intra-caméra du type :
      grande bbox corps complet
      petite bbox partielle sur la même personne

    """
    if len(detections) <= 1:
        return detections

    suppressed: set[int] = set()

    for i in range(len(detections)):
        if i in suppressed:
            continue

        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue

            det_i = detections[i]
            det_j = detections[j]

            if det_i.camera_name != det_j.camera_name:
                continue

            if not should_suppress_as_duplicate(det_i, det_j):
                continue

            score_i = detection_quality_score(det_i)
            score_j = detection_quality_score(det_j)

            area_i = detection_bbox_area(det_i)
            area_j = detection_bbox_area(det_j)

            if abs(score_i - score_j) < 0.20:
                keep_i = area_i >= area_j
            else:
                keep_i = score_i >= score_j

            if keep_i:
                suppressed.add(j)
            else:
                suppressed.add(i)
                break

    return [
        detection
        for index, detection in enumerate(detections)
        if index not in suppressed
    ]


class SingleCameraDetector:
    def __init__(
        self,
        camera_name: str,
        yolo_model_path: str | Path,
        reid_model_path: str | Path,
        pose_estimator: BodyWithFeet,
        confidence_threshold: float,
        person_class_id: int,
        yolo_tracker_config: str,
        pose_min_joint_confidence: float,
        pose_min_valid_joints: int,
        bbox_pose_association_max_distance: float,
        use_torso_crop: bool,
        bbox_margin_ratio: float,
    ) -> None:
        self.camera_name = camera_name
        self.yolo = YOLO(str(yolo_model_path))
        self.reid = ReIdFeatureExtractor(reid_model_path)
        self.pose_estimator = pose_estimator

        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.yolo_tracker_config = yolo_tracker_config

        self.pose_min_joint_confidence = pose_min_joint_confidence
        self.pose_min_valid_joints = pose_min_valid_joints
        self.bbox_pose_association_max_distance = bbox_pose_association_max_distance

        self.use_torso_crop = use_torso_crop
        self.bbox_margin_ratio = bbox_margin_ratio

        self.last_raw_yolo_debug: List[Dict[str, object]] = []
        self.last_detection_counts: Dict[str, int] = {
            "raw_yolo": 0,
            "valid_after_clip": 0,
            "before_suppression": 0,
            "after_suppression": 0,
        }

    def detect_pose(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            keypoints, scores = self.pose_estimator(frame_bgr)
        except Exception as exc:
            print(f"[detection] RTMLib erreur {self.camera_name}: {exc}")
            return (
                np.empty((0, 0, 2), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
            )

        if keypoints is None or len(keypoints) == 0:
            return (
                np.empty((0, 0, 2), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
            )

        return (
            np.asarray(keypoints, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
        )

    def detect_yolo(self, frame_bgr: np.ndarray):
        return self.yolo.track(
            frame_bgr,
            persist=True,
            conf=self.confidence_threshold,
            verbose=False,
            classes=[self.person_class_id],
            tracker=self.yolo_tracker_config,
        )

    def associate_boxes_and_poses(
        self,
        boxes_xyxy: List[BoxXYXY],
        poses_2d: np.ndarray,
        pose_scores: np.ndarray,
    ) -> Dict[int, int]:
        if len(boxes_xyxy) == 0 or len(poses_2d) == 0:
            return {}

        cost = np.full(
            (len(boxes_xyxy), len(poses_2d)),
            fill_value=1e6,
            dtype=np.float32,
        )

        for box_index, bbox in enumerate(boxes_xyxy):
            box_ctr = bbox_center(bbox)

            for pose_index, pose in enumerate(poses_2d):
                center = valid_pose_center(
                    pose_2d=pose,
                    joint_confidences=pose_scores[pose_index],
                    min_joint_confidence=self.pose_min_joint_confidence,
                    min_valid_joints=self.pose_min_valid_joints,
                )

                if center is None:
                    continue

                if not point_inside_bbox(center, bbox):
                    continue

                cost[box_index, pose_index] = float(np.linalg.norm(box_ctr - center))

        box_indices, pose_indices = linear_sum_assignment(cost)

        associations = {}

        for box_index, pose_index in zip(box_indices, pose_indices):
            if cost[box_index, pose_index] <= self.bbox_pose_association_max_distance:
                associations[int(box_index)] = int(pose_index)

        return associations

    def select_reid_crop(
        self,
        frame_bgr: np.ndarray,
        bbox: BoxXYXY,
        pose_2d: Optional[np.ndarray],
        joint_confidences: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        crop_box = None

        if self.use_torso_crop and pose_2d is not None and joint_confidences is not None:
            crop_box = get_torso_bbox(
                pose_2d=pose_2d,
                joint_confidences=joint_confidences,
                image_shape=frame_bgr.shape,
            )

        if crop_box is None:
            crop_box = shrink_bbox(
                bbox=bbox,
                image_shape=frame_bgr.shape,
                margin_ratio=self.bbox_margin_ratio,
            )

        if crop_box is None:
            return None

        crop = crop_image(frame_bgr, crop_box)

        if crop.size == 0:
            return None

        return crop

    def process(self, frame_bgr: np.ndarray) -> List[ViewDetection]:
        self.last_raw_yolo_debug = []
        self.last_detection_counts = {
            "raw_yolo": 0,
            "valid_after_clip": 0,
            "before_suppression": 0,
            "after_suppression": 0,
        }

        poses_2d, pose_scores = self.detect_pose(frame_bgr)
        yolo_results = self.detect_yolo(frame_bgr)

        if yolo_results is None or len(yolo_results) == 0:
            return []

        result = yolo_results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        raw_boxes = result.boxes
        yolo_ids = raw_boxes.id
        self.last_detection_counts["raw_yolo"] = int(len(raw_boxes))

        boxes_xyxy: List[BoxXYXY] = []
        confidences: List[float] = []
        tracklet_ids: List[Optional[int]] = []

        for index, box in enumerate(raw_boxes):
            xyxy = box.xyxy[0].detach().cpu().numpy()
            raw_bbox = tuple(int(round(value)) for value in xyxy)
            bbox = clip_bbox_xyxy(raw_bbox, frame_bgr.shape)

            confidence = float(box.conf[0].detach().cpu().item())

            tracklet_id = None
            if yolo_ids is not None:
                try:
                    tracklet_id = int(yolo_ids[index].detach().cpu().item())
                except Exception:
                    tracklet_id = None

            class_id = self.person_class_id
            try:
                if hasattr(box, "cls") and box.cls is not None:
                    class_id = int(box.cls[0].detach().cpu().item())
            except Exception:
                class_id = self.person_class_id

            self.last_raw_yolo_debug.append({
                "raw_index": int(index),
                "tracklet_id": tracklet_id,
                "confidence": float(confidence),
                "class_id": int(class_id),
                "valid_after_clip": bbox is not None,
                "bbox_xyxy": bbox,
                "raw_bbox_xyxy": raw_bbox,
            })

            if bbox is None:
                continue

            boxes_xyxy.append(bbox)
            confidences.append(confidence)
            tracklet_ids.append(tracklet_id)

        self.last_detection_counts["valid_after_clip"] = int(len(boxes_xyxy))

        associations = self.associate_boxes_and_poses(
            boxes_xyxy=boxes_xyxy,
            poses_2d=poses_2d,
            pose_scores=pose_scores,
        )

        detections: List[ViewDetection] = []

        for box_index, bbox in enumerate(boxes_xyxy):
            pose_2d = None
            joint_confidences = None
            pose_confidence = 0.0

            if box_index in associations:
                pose_index = associations[box_index]
                pose_2d = poses_2d[pose_index]
                joint_confidences = pose_scores[pose_index]
                pose_confidence = float(np.mean(joint_confidences))

            reid_crop = self.select_reid_crop(
                frame_bgr=frame_bgr,
                bbox=bbox,
                pose_2d=pose_2d,
                joint_confidences=joint_confidences,
            )

            feature = None
            if reid_crop is not None:
                feature = self.reid.predict(reid_crop)

            view_quality = estimate_view_reliability_for_3d(
                bbox=bbox,
                image_shape=frame_bgr.shape,
                pose_2d=pose_2d,
                joint_confidences=joint_confidences,
                pose_confidence=pose_confidence,
                min_joint_confidence=self.pose_min_joint_confidence,
            )

            detections.append(
                ViewDetection(
                    camera_name=self.camera_name,
                    bbox_xyxy=bbox,
                    confidence=confidences[box_index],
                    tracklet_id=tracklet_ids[box_index],
                    feature=feature,
                    pose_2d=pose_2d,
                    pose_confidence=pose_confidence,
                    joint_confidences=joint_confidences,
                    image_width=int(view_quality["image_width"]),
                    image_height=int(view_quality["image_height"]),
                    bbox_width_ratio=float(view_quality["bbox_width_ratio"]),
                    bbox_height_ratio=float(view_quality["bbox_height_ratio"]),
                    bbox_area_ratio=float(view_quality["bbox_area_ratio"]),
                    touches_border=bool(view_quality["touches_border"]),
                    edge_truncated=bool(view_quality["edge_truncated"]),
                    valid_joint_count=int(view_quality["valid_joint_count"]),
                    valid_joint_ratio=float(view_quality["valid_joint_ratio"]),
                    reliable_for_3d=bool(view_quality["reliable_for_3d"]),
                    unreliable_reason=str(view_quality["unreliable_reason"]),
                    body_completeness_score=float(view_quality["body_completeness_score"]),
                    partial_body=bool(view_quality["partial_body"]),
                    upper_body_visible=bool(view_quality["upper_body_visible"]),
                    lower_body_visible=bool(view_quality["lower_body_visible"]),
                    torso_visible=bool(view_quality["torso_visible"]),
                )
            )

        self.last_detection_counts["before_suppression"] = int(len(detections))
        final_detections = suppress_duplicate_detections_same_camera(detections)
        self.last_detection_counts["after_suppression"] = int(len(final_detections))

        return final_detections


class MultiCameraDetector:
    def __init__(
        self,
        camera_names: List[str],
        yolo_model_path: str | Path,
        reid_model_path: str | Path,
        rtmlib_device: str,
        rtmlib_backend: str,
        yolo_confidence: float,
        yolo_person_class: int,
        yolo_tracker_config: str,
        pose_min_joint_confidence: float,
        pose_min_valid_joints: int,
        bbox_pose_association_max_distance: float,
        reid_use_torso_crop: bool,
        reid_bbox_margin_ratio: float,
    ) -> None:
        self.camera_names = list(camera_names)

        if rtmlib_device == "cuda" and "CUDAExecutionProvider" not in ort.get_available_providers():
            print("[detection] CUDA indisponible pour onnxruntime, RTMLib passe en CPU.")
            rtmlib_device = "cpu"

        self.pose_estimator = BodyWithFeet(
            det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
            det_input_size=(640, 640),
            pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-384x288-734182ce_20230605.zip",
            pose_input_size=(288, 384),
            to_openpose=False,
            backend=rtmlib_backend,
            device=rtmlib_device,
        )

        self.detectors: Dict[str, SingleCameraDetector] = {}

        for camera_name in self.camera_names:
            self.detectors[camera_name] = SingleCameraDetector(
                camera_name=camera_name,
                yolo_model_path=yolo_model_path,
                reid_model_path=reid_model_path,
                pose_estimator=self.pose_estimator,
                confidence_threshold=yolo_confidence,
                person_class_id=yolo_person_class,
                yolo_tracker_config=yolo_tracker_config,
                pose_min_joint_confidence=pose_min_joint_confidence,
                pose_min_valid_joints=pose_min_valid_joints,
                bbox_pose_association_max_distance=bbox_pose_association_max_distance,
                use_torso_crop=reid_use_torso_crop,
                bbox_margin_ratio=reid_bbox_margin_ratio,
            )

    def process(self, frames_by_camera: Dict[str, np.ndarray]) -> Dict[str, List[ViewDetection]]:
        detections_by_camera: Dict[str, List[ViewDetection]] = {}

        for camera_name in self.camera_names:
            if camera_name not in frames_by_camera:
                raise KeyError(f"Frame manquante pour {camera_name}")

            detections_by_camera[camera_name] = self.detectors[camera_name].process(
                frames_by_camera[camera_name]
            )

        return detections_by_camera

    def raw_yolo_debug_by_camera(self) -> Dict[str, List[Dict[str, object]]]:
        return {
            camera_name: list(self.detectors[camera_name].last_raw_yolo_debug)
            for camera_name in self.camera_names
        }

    def detection_counts_by_camera(self) -> Dict[str, Dict[str, int]]:
        return {
            camera_name: dict(self.detectors[camera_name].last_detection_counts)
            for camera_name in self.camera_names
        }


def print_detection_summary(
    frame_id: int,
    detections_by_camera: Dict[str, List[ViewDetection]],
) -> None:
    print(f"Détections frame {frame_id}")

    for camera_name, detections in detections_by_camera.items():
        with_pose = sum(1 for detection in detections if detection.pose_2d is not None)
        with_feature = sum(1 for detection in detections if detection.feature is not None)

        print(
            f"  {camera_name}: "
            f"{len(detections)} bbox | "
            f"{with_pose} poses | "
            f"{with_feature} features"
        )