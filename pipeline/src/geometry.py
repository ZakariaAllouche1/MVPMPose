from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from mvn.utils.multiview import Camera

from src.cameras import CameraData


BoxXYXY = Tuple[int, int, int, int]
BoxTLBR = Tuple[int, int, int, int]


@dataclass
class PreparedView:
    camera_name: str
    image_bgr: np.ndarray
    crop_bgr: np.ndarray
    box_xyxy: BoxXYXY
    box_tlbr: BoxTLBR
    camera: Camera
    projection: np.ndarray


def xyxy_to_tlbr(box: BoxXYXY) -> BoxTLBR:
    x1, y1, x2, y2 = box
    return y1, x1, y2, x2


def tlbr_to_xyxy(box: BoxTLBR) -> BoxXYXY:
    top, left, bottom, right = box
    return left, top, right, bottom


def clip_box_xyxy(
    box: BoxXYXY,
    image_shape: Tuple[int, int, int],
) -> Optional[BoxXYXY]:
    height, width = image_shape[:2]

    x1, y1, x2, y2 = box

    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(x1 + 1, min(int(round(x2)), width))
    y2 = max(y1 + 1, min(int(round(y2)), height))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def clip_box_tlbr(
    box: BoxTLBR,
    image_shape: Tuple[int, int, int],
) -> Optional[BoxTLBR]:
    return None if (clipped := clip_box_xyxy(tlbr_to_xyxy(box), image_shape)) is None else xyxy_to_tlbr(clipped)


def expand_box_xyxy(
    box: BoxXYXY,
    image_shape: Tuple[int, int, int],
    scale: float,
) -> Optional[BoxXYXY]:
    if scale <= 0:
        raise ValueError(f"scale doit être > 0, reçu: {scale}")

    clipped = clip_box_xyxy(box, image_shape)

    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped

    width = x2 - x1
    height = y2 - y1

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    new_width = width * scale
    new_height = height * scale

    expanded = (
        int(round(cx - new_width / 2.0)),
        int(round(cy - new_height / 2.0)),
        int(round(cx + new_width / 2.0)),
        int(round(cy + new_height / 2.0)),
    )

    return clip_box_xyxy(expanded, image_shape)


def make_square_box_xyxy(
    box: BoxXYXY,
    image_shape: Tuple[int, int, int],
    scale: float = 1.0,
) -> Optional[BoxXYXY]:
    clipped = clip_box_xyxy(box, image_shape)

    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped

    width = x2 - x1
    height = y2 - y1

    side = max(width, height) * scale

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    square = (
        int(round(cx - side / 2.0)),
        int(round(cy - side / 2.0)),
        int(round(cx + side / 2.0)),
        int(round(cy + side / 2.0)),
    )

    return clip_box_xyxy(square, image_shape)


def crop_and_resize(
    image_bgr: np.ndarray,
    box_xyxy: BoxXYXY,
    output_size: Tuple[int, int],
) -> np.ndarray:
    clipped = clip_box_xyxy(box_xyxy, image_bgr.shape)

    if clipped is None:
        raise ValueError(f"BBox invalide pour crop: {box_xyxy}")

    x1, y1, x2, y2 = clipped
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(f"Crop vide pour bbox: {box_xyxy}")

    output_width, output_height = output_size

    return cv2.resize(
        crop,
        (output_width, output_height),
        interpolation=cv2.INTER_LINEAR,
    )


def update_intrinsics_after_crop(
    K: np.ndarray,
    box_tlbr: BoxTLBR,
) -> np.ndarray:
    top, left, _, _ = box_tlbr

    K_new = np.asarray(K, dtype=np.float64).copy()
    K_new[0, 2] -= float(left)
    K_new[1, 2] -= float(top)

    return K_new


def update_intrinsics_after_resize(
    K: np.ndarray,
    crop_size: Tuple[int, int],
    output_size: Tuple[int, int],
) -> np.ndarray:
    crop_width, crop_height = crop_size
    output_width, output_height = output_size

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"Taille de crop invalide: {crop_size}")

    K_new = np.asarray(K, dtype=np.float64).copy()

    scale_x = output_width / float(crop_width)
    scale_y = output_height / float(crop_height)

    K_new[0, 0] *= scale_x
    K_new[0, 2] *= scale_x

    K_new[1, 1] *= scale_y
    K_new[1, 2] *= scale_y

    return K_new


def intrinsics_after_crop_resize(
    K: np.ndarray,
    box_tlbr: BoxTLBR,
    output_size: Tuple[int, int],
) -> np.ndarray:
    top, left, bottom, right = box_tlbr

    crop_width = right - left
    crop_height = bottom - top

    K_crop = update_intrinsics_after_crop(K, box_tlbr)

    return update_intrinsics_after_resize(
        K_crop,
        crop_size=(crop_width, crop_height),
        output_size=output_size,
    )


def projection_from_camera(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)

    if t.ndim == 1:
        t = t.reshape(3, 1)

    if t.shape == (1, 3):
        t = t.reshape(3, 1)

    return np.asarray(K, dtype=np.float64) @ np.hstack([R, t])


def projection_after_crop_resize(
    camera_data: CameraData,
    box_tlbr: BoxTLBR,
    output_size: Tuple[int, int],
) -> np.ndarray:
    K_new = intrinsics_after_crop_resize(
        K=camera_data.K,
        box_tlbr=box_tlbr,
        output_size=output_size,
    )

    P = projection_from_camera(
        K=K_new,
        R=camera_data.R,
        t=camera_data.t,
    )

    return P.astype(np.float32)


def camera_after_crop_resize(
    camera_data: CameraData,
    box_tlbr: BoxTLBR,
    output_size: Tuple[int, int],
) -> Camera:
    top, left, bottom, right = box_tlbr

    crop_width = right - left
    crop_height = bottom - top

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"BBox invalide: {box_tlbr}")

    dist = None

    if camera_data.dist is not None:
        dist = np.asarray(camera_data.dist, dtype=np.float64).copy()

    cam = Camera(
        camera_data.R.copy(),
        camera_data.t.copy(),
        camera_data.K.copy(),
        dist,
        camera_data.name,
    )

    cam.update_after_crop([left, top, right, bottom])

    output_width, output_height = output_size
    cam.update_after_resize(
        (crop_height, crop_width),
        (output_height, output_width),
    )

    return cam


def prepare_view_for_person(
    camera_name: str,
    image_bgr: np.ndarray,
    box_xyxy: BoxXYXY,
    camera_data: CameraData,
    output_size: Tuple[int, int],
    bbox_scale: float = 1.0,
    square_crop: bool = True,
) -> Optional[PreparedView]:
    if square_crop:
        prepared_box = make_square_box_xyxy(
            box=box_xyxy,
            image_shape=image_bgr.shape,
            scale=bbox_scale,
        )
    else:
        prepared_box = expand_box_xyxy(
            box=box_xyxy,
            image_shape=image_bgr.shape,
            scale=bbox_scale,
        )

    if prepared_box is None:
        return None

    box_tlbr = xyxy_to_tlbr(prepared_box)

    crop_bgr = crop_and_resize(
        image_bgr=image_bgr,
        box_xyxy=prepared_box,
        output_size=output_size,
    )

    projection = projection_after_crop_resize(
        camera_data=camera_data,
        box_tlbr=box_tlbr,
        output_size=output_size,
    )

    camera = camera_after_crop_resize(
        camera_data=camera_data,
        box_tlbr=box_tlbr,
        output_size=output_size,
    )

    return PreparedView(
        camera_name=camera_name,
        image_bgr=image_bgr,
        crop_bgr=crop_bgr,
        box_xyxy=prepared_box,
        box_tlbr=box_tlbr,
        camera=camera,
        projection=projection,
    )


def prepare_views_for_track(
    frames_by_camera: Dict[str, np.ndarray],
    boxes_by_camera: Dict[str, BoxXYXY],
    cameras_by_camera: Dict[str, CameraData],
    output_size: Tuple[int, int],
    min_views: int,
    bbox_scale: float = 1.0,
    square_crop: bool = True,
) -> List[PreparedView]:
    prepared_views: List[PreparedView] = []

    for camera_name, box_xyxy in boxes_by_camera.items():
        if camera_name not in frames_by_camera:
            continue

        if camera_name not in cameras_by_camera:
            continue

        prepared = prepare_view_for_person(
            camera_name=camera_name,
            image_bgr=frames_by_camera[camera_name],
            box_xyxy=box_xyxy,
            camera_data=cameras_by_camera[camera_name],
            output_size=output_size,
            bbox_scale=bbox_scale,
            square_crop=square_crop,
        )

        if prepared is not None:
            prepared_views.append(prepared)

    prepared_views = sorted(prepared_views, key=lambda view: view.camera_name)

    if len(prepared_views) < min_views:
        return []

    return prepared_views


def project_points(
    points_3d: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    points_3d = np.asarray(points_3d, dtype=np.float64)

    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"points_3d doit avoir shape [N, 3], reçu: {points_3d.shape}")

    points_h = np.hstack(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float64)]
    )

    projected_h = (projection @ points_h.T).T

    z = projected_h[:, 2:3]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)

    return projected_h[:, :2] / z


def reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    projection: np.ndarray,
) -> float:
    projected = project_points(points_3d, projection)
    points_2d = np.asarray(points_2d, dtype=np.float64)

    if projected.shape != points_2d.shape:
        raise ValueError(
            f"Shapes incompatibles: projected={projected.shape}, points_2d={points_2d.shape}"
        )

    valid = (
        np.isfinite(projected[:, 0])
        & np.isfinite(projected[:, 1])
        & np.isfinite(points_2d[:, 0])
        & np.isfinite(points_2d[:, 1])
    )

    if not np.any(valid):
        return float("inf")

    return float(np.mean(np.linalg.norm(projected[valid] - points_2d[valid], axis=1)))


def weighted_triangulate_point(
    projections: List[np.ndarray],
    points_2d: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    if len(projections) != len(points_2d):
        raise ValueError("projections et points_2d doivent avoir la même longueur")

    if len(projections) < 2:
        raise ValueError("Il faut au moins 2 vues pour trianguler")

    if weights is None:
        weights = [1.0] * len(projections)

    if len(weights) != len(projections):
        raise ValueError("weights doit avoir la même longueur que projections")

    A = np.zeros((2 * len(projections), 4), dtype=np.float64)

    for i, (P, point, weight) in enumerate(zip(projections, points_2d, weights)):
        x, y = float(point[0]), float(point[1])
        w = float(weight)

        A[2 * i] = w * (x * P[2] - P[0])
        A[2 * i + 1] = w * (y * P[2] - P[1])

    _, _, vh = np.linalg.svd(A)
    X = vh[-1]

    if abs(X[3]) < 1e-12:
        raise ValueError("Triangulation instable: coordonnée homogène trop proche de zéro")

    return (X[:3] / X[3]).astype(np.float32)


def is_in_front_of_camera(
    point_3d: np.ndarray,
    camera_data: CameraData,
) -> bool:
    point = np.asarray(point_3d, dtype=np.float64).reshape(3, 1)
    cam_point = camera_data.R @ point + camera_data.t

    return bool(cam_point[2, 0] > 0)