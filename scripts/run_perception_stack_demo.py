import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.gca_perception_stack import GCAPerceptionStack
from src.perception.openloris_loader import OpenLORISLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RGB-depth perception stack on one OpenLORIS frame.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Path to an OpenLORIS packaged sequence, e.g. corridor1-1",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=732,
        help="Frame index to inspect.",
    )
    args = parser.parse_args()

    loader = OpenLORISLoader()
    frames = loader.load_sequence(args.sequence_dir)
    frame = frames[args.frame_index]

    rgb = loader.load_color_rgb(frame).astype(np.float32) / 255.0
    depth = loader.load_aligned_depth_meters(frame).astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)

    stack = GCAPerceptionStack(feature_channels=128)
    stack.eval()
    with torch.inference_mode():
        bundle = stack(rgb_tensor=rgb_tensor, depth_tensor=depth_tensor)

    payload = {
        "frame_index": frame.frame_index,
        "rgb_feature_map_shape": list(bundle.rgb_feature_map.shape),
        "depth_feature_map_shape": list(bundle.depth_feature_map.shape),
        "fused_feature_map_shape": list(bundle.fused_feature_map.shape),
        "normal_map_shape": list(bundle.normal_map.shape),
        "depth_variance_map_shape": list(bundle.depth_variance_map.shape),
        "rgb_global_feature_shape": list(bundle.rgb_global_feature.shape),
        "depth_global_feature_shape": list(bundle.depth_global_feature.shape),
        "fused_global_feature_shape": list(bundle.fused_global_feature.shape),
        "rgb_global_feature_norm": float(torch.linalg.vector_norm(bundle.rgb_global_feature).item()),
        "depth_global_feature_norm": float(torch.linalg.vector_norm(bundle.depth_global_feature).item()),
        "fused_global_feature_norm": float(torch.linalg.vector_norm(bundle.fused_global_feature).item()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
