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
from src.reasoning.corridor_navigation_reasoner import CorridorNavigationReasoner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corridor navigation reasoning on OpenLORIS frames.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Path to an OpenLORIS packaged sequence, e.g. corridor1-1",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=732,
        help="First frame index to inspect.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to process.",
    )
    parser.add_argument(
        "--robot-width-m",
        type=float,
        default=0.55,
        help="Robot body width in meters.",
    )
    parser.add_argument(
        "--safety-margin-m",
        type=float,
        default=0.15,
        help="Required lateral safety margin in meters.",
    )
    args = parser.parse_args()

    loader = OpenLORISLoader()
    frames = loader.load_sequence(args.sequence_dir)
    if not frames:
        raise ValueError(f"No frames found in {args.sequence_dir}")
    if args.start_frame < 0 or args.start_frame >= len(frames):
        raise IndexError(f"start-frame {args.start_frame} out of range 0..{len(frames) - 1}")

    end_frame = min(len(frames), args.start_frame + args.num_frames)
    intrinsics = loader.load_color_intrinsics(args.sequence_dir)
    pipeline = FusedGeometryPipeline(
        robot_width_m=args.robot_width_m,
        safety_margin_m=args.safety_margin_m,
    )
    reasoner = CorridorNavigationReasoner()
    pipeline.reset_tracks()
    reasoner.reset()

    outputs = []
    for frame in frames[args.start_frame:end_frame]:
        rgb = loader.load_color_rgb(frame)
        depth_m = loader.load_aligned_depth_meters(frame)
        result = pipeline.run(
            rgb_image=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            frame_index=frame.frame_index,
        )
        decision = reasoner.decide(result.geometry_state, result.gca_result)
        outputs.append(
            {
                "frame_index": frame.frame_index,
                "color_timestamp": frame.color_timestamp,
                "decision": asdict(decision),
                "geometry": {
                    "corridor_width_m": result.gca_result.constrained_state.corridor_width_m,
                    "traversable_width_m": result.gca_result.constrained_state.traversable_width_m,
                    "passable": result.gca_result.constrained_state.passable,
                    "nearest_obstacle": (
                        asdict(result.gca_result.constrained_state.nearest_obstacle)
                        if result.gca_result.constrained_state.nearest_obstacle is not None
                        else None
                    ),
                },
                "gca": {
                    "geometry_valid": result.gca_result.geometry_valid,
                    "facts": result.gca_result.facts,
                },
            }
        )

    if len(outputs) == 1:
        print(json.dumps(outputs[0], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
