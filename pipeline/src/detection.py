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
        poses_2d, pose_scores = self.detect_pose(frame_bgr)
        yolo_results = self.detect_yolo(frame_bgr)

        if yolo_results is None or len(yolo_results) == 0:
            return []

        result = yolo_results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        raw_boxes = result.boxes
        yolo_ids = raw_boxes.id

        boxes_xyxy: List[BoxXYXY] = []
        confidences: List[float] = []
        tracklet_ids: List[Optional[int]] = []

        for index, box in enumerate(raw_boxes):
            xyxy = box.xyxy[0].detach().cpu().numpy()
            bbox = tuple(int(round(value)) for value in xyxy)
            bbox = clip_bbox_xyxy(bbox, frame_bgr.shape)

            if bbox is None:
                continue

            confidence = float(box.conf[0].detach().cpu().item())

            tracklet_id = None
            if yolo_ids is not None:
                try:
                    tracklet_id = int(yolo_ids[index].detach().cpu().item())
                except Exception:
                    tracklet_id = None

            boxes_xyxy.append(bbox)
            confidences.append(confidence)
            tracklet_ids.append(tracklet_id)

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
                )
            )

        return detections


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