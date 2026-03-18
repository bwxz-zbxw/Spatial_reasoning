from __future__ import annotations

from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small


class RGBSemanticEncoder(nn.Module):
    """Lightweight RGB encoder built on MobileNetV3-Small."""

    def __init__(self, output_channels: int = 128) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.projection = nn.Sequential(
            nn.Conv2d(576, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_tensor):
        if rgb_tensor.ndim != 4 or rgb_tensor.shape[1] != 3:
            raise ValueError("rgb_tensor must have shape [B, 3, H, W]")

        features = self.features(rgb_tensor)
        return self.projection(features)

    def forward_resized(self, rgb_tensor, target_hw: tuple[int, int]):
        feature_map = self.forward(rgb_tensor)
        return F.interpolate(feature_map, size=target_hw, mode="bilinear", align_corners=False)
