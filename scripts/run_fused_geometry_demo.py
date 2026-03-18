import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.fused_geometry_pipeline import FusedGeometryPipeline
from src.perception.openloris_loader import OpenLORISLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fused perception-to-geometry pipeline on one OpenLORIS frame.")
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
    rgb = loader.load_color_rgb(frame)
    depth_m = loader.load_aligned_depth_meters(frame)
    intrinsics = loader.load_color_intrinsics(args.sequence_dir)

    pipeline = FusedGeometryPipeline()
    result = pipeline.run(rgb_image=rgb, depth_m=depth_m, intrinsics=intrinsics, frame_index=frame.frame_index)

    payload = asdict(result.geometry_state)
    payload["gca"] = {
        "geometry_valid": result.gca_result.geometry_valid,
        "required_width_m": result.gca_result.required_width_m,
        "facts": result.gca_result.facts,
        "constrained_state": asdict(result.gca_result.constrained_state),
        "evaluations": [asdict(item) for item in result.gca_result.evaluations],
    }
    payload["frame_index"] = frame.frame_index
    payload["rgb_feature_map_shape"] = list(result.feature_bundle.rgb_feature_map.shape)
    payload["depth_feature_map_shape"] = list(result.feature_bundle.depth_feature_map.shape)
    payload["fused_feature_map_shape"] = list(result.feature_bundle.fused_feature_map.shape)
    payload["wall_guidance_shape"] = list(result.guidance_maps.wall_guidance_map.shape)
    payload["obstacle_guidance_shape"] = list(result.guidance_maps.obstacle_guidance_map.shape)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
