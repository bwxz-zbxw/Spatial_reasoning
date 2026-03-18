from __future__ import annotations

from torch import nn


class CrossModalFusion(nn.Module):
    """Cross-attention fusion for RGB and depth geometry features."""

    def __init__(self, channels: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(channels)
        self.depth_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, rgb_feature_map, depth_feature_map):
        if rgb_feature_map.shape != depth_feature_map.shape:
            raise ValueError("RGB and depth feature maps must have the same shape for fusion")

        batch_size, channels, height, width = rgb_feature_map.shape
        rgb_tokens = rgb_feature_map.flatten(2).transpose(1, 2)
        depth_tokens = depth_feature_map.flatten(2).transpose(1, 2)

        query = self.rgb_norm(rgb_tokens)
        key_value = self.depth_norm(depth_tokens)
        attended, _ = self.attention(query=query, key=key_value, value=key_value)
        fused = rgb_tokens + attended
        fused = fused + self.feed_forward(fused)
        return fused.transpose(1, 2).reshape(batch_size, channels, height, width)
