from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.geometry.rgbd_geometry_branch import RGBDGeometryBranch, SceneGeometryState
from src.perception.gca_perception_stack import GCAPerceptionStack, PerceptionFeatureBundle, PerceptionGuidanceMaps
from src.reasoning.gca_constraint_layer import GCAConstraintLayer, GCAConstraintResult


@dataclass
class FusedGeometryResult:
    geometry_state: SceneGeometryState
    gca_result: GCAConstraintResult
    feature_bundle: PerceptionFeatureBundle
    guidance_maps: PerceptionGuidanceMaps


class FusedGeometryPipeline:
    """Connect the perception stack to the geometry branch through guidance maps."""

    def __init__(
        self,
        feature_channels: int = 128,
        input_size: tuple[int, int] = (240, 320),
        robot_width_m: float = 0.55,
        safety_margin_m: float = 0.15,
    ) -> None:
        self.perception_stack = GCAPerceptionStack(
            feature_channels=feature_channels,
            input_size=input_size,
        )
        self.geometry_branch = RGBDGeometryBranch(
            robot_width_m=robot_width_m,
            safety_margin_m=safety_margin_m,
        )
        self.gca_layer = GCAConstraintLayer()
        self.perception_stack.eval()

    def reset_tracks(self) -> None:
        self.geometry_branch.reset_tracks()

    def run(
        self,
        rgb_image: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        frame_index: int | None = None,
    ) -> FusedGeometryResult:
        rgb_tensor = torch.from_numpy(rgb_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        depth_clean = np.nan_to_num(depth_m.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        depth_tensor = torch.from_numpy(depth_clean).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            feature_bundle = self.perception_stack(rgb_tensor=rgb_tensor, depth_tensor=depth_tensor)
            guidance_maps = self.perception_stack.build_guidance_maps(
                feature_bundle,
                output_size=depth_clean.shape,
            )

        geometry_state = self.geometry_branch.estimate(
            depth_m=depth_m,
            intrinsics=intrinsics,
            detected_objects=[],
            frame_index=frame_index,
            guidance_maps={
                "wall_guidance_map": guidance_maps.wall_guidance_map.squeeze(0).squeeze(0).cpu().numpy(),
                "obstacle_guidance_map": guidance_maps.obstacle_guidance_map.squeeze(0).squeeze(0).cpu().numpy(),
            },
        )
        gca_result = self.gca_layer.apply(
            geometry_state=geometry_state,
            normal_map=feature_bundle.normal_map.squeeze(0).cpu().numpy(),
        )
        return FusedGeometryResult(
            geometry_state=geometry_state,
            gca_result=gca_result,
            feature_bundle=feature_bundle,
            guidance_maps=guidance_maps,
        )
