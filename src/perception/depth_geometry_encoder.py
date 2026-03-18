from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DepthGeometryEncoding:
    feature_map: torch.Tensor
    normal_map: torch.Tensor
    depth_variance_map: torch.Tensor


class DepthGeometryEncoder(nn.Module):
    """Encode depth into lightweight geometry-aware feature maps."""

    def __init__(self, output_channels: int = 128) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth_tensor: torch.Tensor) -> DepthGeometryEncoding:
        if depth_tensor.ndim != 4 or depth_tensor.shape[1] != 1:
            raise ValueError("depth_tensor must have shape [B, 1, H, W]")

        depth_tensor = torch.nan_to_num(depth_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        normal_map = self._estimate_normals(depth_tensor)
        variance_map = self._estimate_local_variance(depth_tensor)
        geometry_input = torch.cat([depth_tensor, normal_map, variance_map], dim=1)
        feature_map = self.encoder(geometry_input)
        return DepthGeometryEncoding(
            feature_map=feature_map,
            normal_map=normal_map,
            depth_variance_map=variance_map,
        )

    def _estimate_normals(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        grad_x = depth_tensor[:, :, :, 1:] - depth_tensor[:, :, :, :-1]
        grad_y = depth_tensor[:, :, 1:, :] - depth_tensor[:, :, :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        normal_x = -grad_x
        normal_y = -grad_y
        normal_z = torch.ones_like(depth_tensor)
        normals = torch.cat([normal_x, normal_y, normal_z], dim=1)
        return F.normalize(normals, p=2, dim=1, eps=1e-6)

    def _estimate_local_variance(self, depth_tensor: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        padding = kernel_size // 2
        mean = F.avg_pool2d(depth_tensor, kernel_size=kernel_size, stride=1, padding=padding)
        mean_sq = F.avg_pool2d(depth_tensor * depth_tensor, kernel_size=kernel_size, stride=1, padding=padding)
        variance = torch.clamp(mean_sq - (mean * mean), min=0.0)
        return variance
