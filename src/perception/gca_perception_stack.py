from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.perception.cross_modal_fusion import CrossModalFusion
from src.perception.depth_geometry_encoder import DepthGeometryEncoder, DepthGeometryEncoding
from src.perception.rgb_semantic_encoder import RGBSemanticEncoder


@dataclass
class PerceptionFeatureBundle:
    rgb_feature_map: torch.Tensor
    depth_feature_map: torch.Tensor
    fused_feature_map: torch.Tensor
    normal_map: torch.Tensor
    depth_variance_map: torch.Tensor
    rgb_global_feature: torch.Tensor
    depth_global_feature: torch.Tensor
    fused_global_feature: torch.Tensor


class GCAPerceptionStack(nn.Module):
    """RGB-depth dual-branch perception stack aligned with the planned architecture."""

    def __init__(self, feature_channels: int = 128, input_size: tuple[int, int] = (240, 320)) -> None:
        super().__init__()
        self.input_size = input_size
        self.rgb_encoder = RGBSemanticEncoder(output_channels=feature_channels)
        self.depth_encoder = DepthGeometryEncoder(output_channels=feature_channels)
        self.fusion = CrossModalFusion(channels=feature_channels, num_heads=4)

    def forward(self, rgb_tensor: torch.Tensor, depth_tensor: torch.Tensor) -> PerceptionFeatureBundle:
        rgb_tensor = F.interpolate(rgb_tensor, size=self.input_size, mode="bilinear", align_corners=False)
        depth_tensor = F.interpolate(depth_tensor, size=self.input_size, mode="bilinear", align_corners=False)
        depth_encoding: DepthGeometryEncoding = self.depth_encoder(depth_tensor)
        target_hw = depth_encoding.feature_map.shape[-2:]
        rgb_feature_map = self.rgb_encoder.forward_resized(rgb_tensor, target_hw=target_hw)
        fused_feature_map = self.fusion(rgb_feature_map, depth_encoding.feature_map)

        rgb_global = F.adaptive_avg_pool2d(rgb_feature_map, output_size=1).flatten(1)
        depth_global = F.adaptive_avg_pool2d(depth_encoding.feature_map, output_size=1).flatten(1)
        fused_global = F.adaptive_avg_pool2d(fused_feature_map, output_size=1).flatten(1)

        return PerceptionFeatureBundle(
            rgb_feature_map=rgb_feature_map,
            depth_feature_map=depth_encoding.feature_map,
            fused_feature_map=fused_feature_map,
            normal_map=depth_encoding.normal_map,
            depth_variance_map=depth_encoding.depth_variance_map,
            rgb_global_feature=rgb_global,
            depth_global_feature=depth_global,
            fused_global_feature=fused_global,
        )
