from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from mvn.models.triangulation import VolumetricTriangulationNet
from mvn.models.v2v import V2VModel


HEAD_ORDER: List[str] = ["custom", "ecomocap", "metasport"]
HEAD_NUM_JOINTS: Dict[str, int] = {"custom": 19, "ecomocap": 32, "metasport": 32}
N_MODEL83 = sum(HEAD_NUM_JOINTS.values())
V2V_INPUT_CHANNELS = 32


class MultiV2V(nn.Module):

    def __init__(
        self,
        in_channels: int = V2V_INPUT_CHANNELS,
        head_num_joints: Dict[str, int] | None = None,
        head_order: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.head_order = list(head_order or HEAD_ORDER)
        self.head_num_joints = dict(head_num_joints or HEAD_NUM_JOINTS)

        self.nets = nn.ModuleDict({
            name: V2VModel(in_channels, int(self.head_num_joints[name]))
            for name in self.head_order
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.nets[name](x) for name in self.head_order], dim=1)
        if out.shape[1] != N_MODEL83:
            raise RuntimeError(
                f"MultiV2V a produit {out.shape[1]} canaux, attendu {N_MODEL83}."
            )
        return out


class VolumetricTriangulationNetMultiV2V(VolumetricTriangulationNet):

    def __init__(self, config: Any, device: str = "cuda:0") -> None:
        super().__init__(config, device=device)

        if int(self.num_joints) != N_MODEL83:
            raise ValueError(
                f"config num_joints={self.num_joints}, mais le multi-V2V attend {N_MODEL83}."
            )

        self.volume_net = MultiV2V(
            in_channels=V2V_INPUT_CHANNELS,
            head_num_joints=HEAD_NUM_JOINTS,
            head_order=HEAD_ORDER,
        )
