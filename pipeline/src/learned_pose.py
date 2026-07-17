from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

from mvn.models.triangulation import (
    AlgebraicTriangulationNet,
    VolumetricTriangulationNet,
)

try:
    from src.vol_multihead_multiV2V_inference import (
        VolumetricTriangulationNetMultiV2V,
    )
except ImportError:
    VolumetricTriangulationNetMultiV2V = None

from src.cameras import CameraData
from src.geometry import BoxXYXY, PreparedView, prepare_views_for_track
from src.joints import (
    N_MODEL83,
    get_root_position_18,
    model19_to_18_numpy,
    model83_to_18_numpy,
    model83_to_custom19_numpy,
    model83_to_ecomocap32_numpy,
    model83_to_metasport32_numpy,
)


@dataclass
class LearnedPoseResult:
    person_id: int
    views_used: List[str]

    keypoints18: np.ndarray
    root_position: np.ndarray
    score: float

    keypoints19: np.ndarray

    keypoints83: Optional[np.ndarray] = None
    keypoints_custom19: Optional[np.ndarray] = None
    keypoints_metasport32: Optional[np.ndarray] = None
    keypoints_ecomocap32: Optional[np.ndarray] = None


def safe_torch_load(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value

    return cleaned


def extract_pred_3d(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, (tuple, list)):
        return model_output[0]

    return model_output


def extract_base_point(model_output: Any) -> Optional[torch.Tensor]:
    if not isinstance(model_output, (tuple, list)):
        return None

    if len(model_output) <= 6:
        return None

    item = model_output[6]

    if torch.is_tensor(item):
        return item

    return None


HEAD_ORDER = ["custom", "ecomocap", "metasport"]
HEAD_NUM_JOINTS = {"custom": 19, "ecomocap": 32, "metasport": 32}


class MultiHeadFinalLayer(nn.Module):

    def __init__(self, template_conv: nn.Conv2d) -> None:
        super().__init__()

        if not isinstance(template_conv, nn.Conv2d):
            raise TypeError(
                f"template_conv doit être nn.Conv2d, reçu: {type(template_conv)}"
            )

        self.heads = nn.ModuleDict()
        for head_name in HEAD_ORDER:
            self.heads[head_name] = nn.Conv2d(
                in_channels=template_conv.in_channels,
                out_channels=HEAD_NUM_JOINTS[head_name],
                kernel_size=template_conv.kernel_size,
                stride=template_conv.stride,
                padding=template_conv.padding,
                dilation=template_conv.dilation,
                groups=template_conv.groups,
                bias=(template_conv.bias is not None),
                padding_mode=template_conv.padding_mode,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.heads[name](x) for name in HEAD_ORDER], dim=1)


def patch_pose_resnet_final_layer_to_multihead(model: nn.Module) -> None:
    """Patch Algebraic/Volumetric PoseResNet to match multi-head checkpoints."""
    if not hasattr(model, "backbone"):
        raise AttributeError("model.backbone introuvable")

    backbone = model.backbone

    if not hasattr(backbone, "final_layer"):
        raise AttributeError("backbone.final_layer introuvable")

    if isinstance(backbone.final_layer, MultiHeadFinalLayer):
        return

    if not isinstance(backbone.final_layer, nn.Conv2d):
        raise TypeError(
            f"backbone.final_layer doit être nn.Conv2d avant patch, reçu: {type(backbone.final_layer)}"
        )

    backbone.final_layer = MultiHeadFinalLayer(backbone.final_layer)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32,
        device=images.device,
    ).view(1, 1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
        device=images.device,
    ).view(1, 1, 3, 1, 1)

    return (images - mean) / std


def image_bgr_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr est vide")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = image_rgb.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    return torch.from_numpy(image)


def try_extract_confidences(model_output: Any, pred3d: torch.Tensor) -> torch.Tensor:
    batch_size, n_joints, _ = pred3d.shape

    if isinstance(model_output, (tuple, list)):
        for item in model_output[1:]:
            if not torch.is_tensor(item):
                continue

            tensor = item.detach()

            if tensor.ndim == 2 and tensor.shape == (batch_size, n_joints):
                return torch.nan_to_num(
                    tensor.float(),
                    nan=1.0,
                    posinf=1.0,
                    neginf=0.0,
                )

            if (
                tensor.ndim == 3
                and tensor.shape[0] == batch_size
                and tensor.shape[1] == n_joints
                and tensor.shape[2] == 1
            ):
                return torch.nan_to_num(
                    tensor[:, :, 0].float(),
                    nan=1.0,
                    posinf=1.0,
                    neginf=0.0,
                )

    return torch.ones(
        (batch_size, n_joints),
        dtype=torch.float32,
        device=pred3d.device,
    )


def make_pred_keypoints_3d_list(
    pred3d: torch.Tensor,
    alg_output: Any,
) -> List[np.ndarray]:
    confidences = try_extract_confidences(alg_output, pred3d)

    pred_np = pred3d.detach().cpu().numpy().astype(np.float32)
    conf_np = confidences.detach().cpu().numpy().astype(np.float32)

    output: List[np.ndarray] = []

    for sample_index in range(pred_np.shape[0]):
        arr = np.zeros((pred_np.shape[1], 4), dtype=np.float32)
        arr[:, :3] = pred_np[sample_index]
        arr[:, 3] = conf_np[sample_index]
        output.append(arr)

    return output


def build_alg_model_config(
    n_joints: int,
    image_size: tuple[int, int],
    heatmap_size: tuple[int, int],
) -> EasyDict:
    width, height = image_size
    heatmap_width, heatmap_height = heatmap_size

    return EasyDict({
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
                "num_joints": n_joints,
                "image_size": [width, height],
                "heatmap_size": [heatmap_width, heatmap_height],
                "init_weights": False,
                "alg_confidences": True,
                "vol_confidences": False,
            },
        }
    })


def build_vol_model_config(
    n_joints: int,
    image_size: tuple[int, int],
    heatmap_size: tuple[int, int],
    cuboid_side: float,
    volume_size: int,
) -> EasyDict:
    width, height = image_size
    heatmap_width, heatmap_height = heatmap_size

    return EasyDict({
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
                "num_joints": n_joints,
                "image_size": [width, height],
                "heatmap_size": [heatmap_width, heatmap_height],
                "init_weights": False,
            },
            "num_joints": n_joints,
            "volume_size": int(volume_size),
            "volume_multiplier": 1.0,
            "volume_softmax": True,
            "volume_aggregation_method": "softmax",
            "use_gt_pelvis": False,
            "cuboid_side": float(cuboid_side),
            "heatmap_softmax": True,
            "heatmap_multiplier": 100.0,
        }
    })


def extract_checkpoint_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint invalide: state_dict doit être un dict, reçu {type(state_dict)}"
        )

    return strip_module_prefix(state_dict)


def checkpoint_is_multiV2V(state_dict: Dict[str, Any]) -> bool:
    return any(str(key).startswith("volume_net.nets.") for key in state_dict.keys())


class LearnedPoseEstimator:
    def __init__(
        self,
        alg_weights_path: str | Path,
        vol_weights_path: str | Path,
        device: str,
        image_size: tuple[int, int],
        n_model_joints: int,
        cuboid_side: float,
        heatmap_size: tuple[int, int] = (96, 96),
        volume_size: int = 96,
    ) -> None:
        self.alg_weights_path = Path(alg_weights_path)
        self.vol_weights_path = Path(vol_weights_path)
        self.device = device
        self.image_size = image_size
        self.n_model_joints = int(n_model_joints)
        self.cuboid_side = float(cuboid_side)
        self.heatmap_size = heatmap_size
        self.volume_size = int(volume_size)
        self.is_multihead = self.n_model_joints == N_MODEL83

        if self.n_model_joints not in {19, N_MODEL83}:
            raise ValueError(
                "n_model_joints doit être 19 pour l'ancien modèle ou 83 pour le modèle multi-head, "
                f"reçu: {self.n_model_joints}"
            )

        self.alg_model = self.load_alg_model()
        self.vol_model = self.load_vol_model()

    def load_alg_model(self) -> torch.nn.Module:
        if not self.alg_weights_path.exists():
            raise FileNotFoundError(f"Poids ALG introuvables: {self.alg_weights_path}")

        config = build_alg_model_config(
            n_joints=self.n_model_joints,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
        )

        model = AlgebraicTriangulationNet(config, device=self.device)

        if self.is_multihead:
            patch_pose_resnet_final_layer_to_multihead(model)

        checkpoint = safe_torch_load(self.alg_weights_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = strip_module_prefix(state_dict)

        result = model.load_state_dict(state_dict, strict=bool(self.is_multihead))

        if result.missing_keys:
            print(f"[learned pose] ALG missing keys: {result.missing_keys}")

        if result.unexpected_keys:
            print(f"[learned pose] ALG unexpected keys: {result.unexpected_keys}")

        model.to(self.device)
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        return model

    def load_vol_model(self) -> torch.nn.Module:
        if not self.vol_weights_path.exists():
            raise FileNotFoundError(f"Poids VOL introuvables: {self.vol_weights_path}")

        config = build_vol_model_config(
            n_joints=self.n_model_joints,
            image_size=self.image_size,
            heatmap_size=self.heatmap_size,
            cuboid_side=self.cuboid_side,
            volume_size=self.volume_size,
        )

        checkpoint = safe_torch_load(self.vol_weights_path, map_location=self.device)
        state_dict = extract_checkpoint_state_dict(checkpoint)
        use_multi_v2v = bool(self.is_multihead and checkpoint_is_multiV2V(state_dict))

        if use_multi_v2v:
            if VolumetricTriangulationNetMultiV2V is None:
                raise ImportError(
                    "Le checkpoint VOL est un multi-V2V, mais "
                    "src.vol_multihead_multiV2V_inference est introuvable. "
                    "Ajoute ce fichier dans src/."
                )
            model = VolumetricTriangulationNetMultiV2V(config, device=self.device)
            print("[learned pose] VOL architecture détectée: multi-V2V")
        else:
            model = VolumetricTriangulationNet(config, device=self.device)
            if self.is_multihead:
                patch_pose_resnet_final_layer_to_multihead(model)
            print("[learned pose] VOL architecture détectée: standard")

        result = model.load_state_dict(state_dict, strict=bool(self.is_multihead))

        if result.missing_keys:
            print(f"[learned pose] VOL missing keys: {result.missing_keys}")

        if result.unexpected_keys:
            print(f"[learned pose] VOL unexpected keys: {result.unexpected_keys}")

        model.to(self.device)
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        return model

    def prepared_views_to_model_inputs(
        self,
        prepared_views: List[PreparedView],
    ) -> tuple[torch.Tensor, torch.Tensor, List[List[Any]], List[str]]:
        if len(prepared_views) < 1:
            raise ValueError("prepared_views est vide")

        image_tensors = []
        projection_tensors = []
        cameras_batch: List[List[Any]] = []
        views_used = []

        for view in prepared_views:
            image_tensors.append(image_bgr_to_tensor(view.crop_bgr))
            projection_tensors.append(torch.from_numpy(view.projection).float())
            cameras_batch.append([view.camera])
            views_used.append(view.camera_name)


        images = torch.stack(image_tensors, dim=0).unsqueeze(0)
        projections = torch.stack(projection_tensors, dim=0).unsqueeze(0)

        images = images.to(self.device).float()
        projections = projections.to(self.device).float()

        images = normalize_images(images)

        return images, projections, cameras_batch, views_used

    def split_prediction_numpy(
        self,
        pred_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Retourne custom19, custom18, keypoints83, metasport32, ecomocap32.

        En mode multi-head, pred_np est [83, 3].
        En mode ancien modèle, pred_np est [19, 3].
        """
        pred_np = np.asarray(pred_np, dtype=np.float32)

        if self.is_multihead:
            keypoints83 = pred_np
            custom19 = model83_to_custom19_numpy(keypoints83)
            keypoints18 = model83_to_18_numpy(keypoints83)
            metasport32 = model83_to_metasport32_numpy(keypoints83)
            ecomocap32 = model83_to_ecomocap32_numpy(keypoints83)

            return custom19, keypoints18, keypoints83, metasport32, ecomocap32

        custom19 = pred_np
        keypoints18 = model19_to_18_numpy(custom19)

        return custom19, keypoints18, None, None, None

    @torch.no_grad()
    def infer_prepared_views(
        self,
        person_id: int,
        prepared_views: List[PreparedView],
    ) -> Optional[LearnedPoseResult]:
        if not prepared_views:
            return None

        images, projections, cameras_batch, views_used = self.prepared_views_to_model_inputs(
            prepared_views
        )

        alg_output = self.alg_model(images, projections, None)
        alg_pred = extract_pred_3d(alg_output)

        if alg_pred.shape[-2:] != (self.n_model_joints, 3):
            raise ValueError(
                f"Sortie ALG inattendue: {tuple(alg_pred.shape)} ; attendu [..., {self.n_model_joints}, 3]"
            )

        pred_keypoints_3d = make_pred_keypoints_3d_list(alg_pred, alg_output)

        vol_batch = {
            "cameras": cameras_batch,
            "pred_keypoints_3d": pred_keypoints_3d,
        }

        vol_output = self.vol_model(images, projections, vol_batch)
        vol_pred = extract_pred_3d(vol_output)

        if vol_pred.shape[-2:] != (self.n_model_joints, 3):
            raise ValueError(
                f"Sortie VOL inattendue: {tuple(vol_pred.shape)} ; attendu [..., {self.n_model_joints}, 3]"
            )

        if not torch.isfinite(vol_pred).all():
            return None

        pred_np = vol_pred[0].detach().cpu().numpy().astype(np.float32)
        keypoints19, keypoints18, keypoints83, metasport32, ecomocap32 = self.split_prediction_numpy(
            pred_np
        )
        root_position = get_root_position_18(keypoints18)

        score = self.estimate_score(
            keypoints19=keypoints19,
            keypoints18=keypoints18,
            keypoints83=keypoints83,
            views_used=views_used,
        )

        return LearnedPoseResult(
            person_id=person_id,
            views_used=views_used,
            keypoints18=keypoints18,
            root_position=root_position,
            score=score,
            keypoints19=keypoints19,
            keypoints83=keypoints83,
            keypoints_custom19=keypoints19,
            keypoints_metasport32=metasport32,
            keypoints_ecomocap32=ecomocap32,
        )

    def estimate_score(
        self,
        keypoints19: np.ndarray,
        keypoints18: np.ndarray,
        views_used: List[str],
        keypoints83: Optional[np.ndarray] = None,
    ) -> float:
        if not np.isfinite(keypoints19).all():
            return 0.0

        if not np.isfinite(keypoints18).all():
            return 0.0

        if keypoints83 is not None and not np.isfinite(keypoints83).all():
            return 0.0

        view_score = min(1.0, len(views_used) / 4.0)

        spread = np.linalg.norm(
            keypoints18 - keypoints18[2:3],
            axis=1,
        )

        if not np.isfinite(spread).all():
            return 0.0

        if float(np.max(spread)) < 100.0:
            shape_score = 0.3
        else:
            shape_score = 1.0

        return float(view_score * shape_score)

    def infer_track(
        self,
        person_id: int,
        frames_by_camera: Dict[str, np.ndarray],
        boxes_by_camera: Dict[str, BoxXYXY],
        cameras_by_camera: Dict[str, CameraData],
        min_views: int,
        bbox_scale: float,
        square_crop: bool,
    ) -> Optional[LearnedPoseResult]:
        prepared_views = prepare_views_for_track(
            frames_by_camera=frames_by_camera,
            boxes_by_camera=boxes_by_camera,
            cameras_by_camera=cameras_by_camera,
            output_size=self.image_size,
            min_views=min_views,
            bbox_scale=bbox_scale,
            square_crop=square_crop,
        )

        if len(prepared_views) < min_views:
            return None

        return self.infer_prepared_views(
            person_id=person_id,
            prepared_views=prepared_views,
        )
