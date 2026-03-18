import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.rgbd_geometry_branch import RGBDGeometryBranch
from src.perception.openloris_loader import OpenLORISLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RGB-D geometry branch on one OpenLORIS frame.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Path to an OpenLORIS packaged sequence, e.g. corridor1-1",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to inspect.",
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
    if args.frame_index < 0 or args.frame_index >= len(frames):
        raise IndexError(f"frame-index {args.frame_index} out of range 0..{len(frames) - 1}")

    frame = frames[args.frame_index]
    intrinsics = loader.load_color_intrinsics(args.sequence_dir)
    depth_m = loader.load_aligned_depth_meters(frame)

    branch = RGBDGeometryBranch(
        robot_width_m=args.robot_width_m,
        safety_margin_m=args.safety_margin_m,
    )
    state = branch.estimate(depth_m=depth_m, intrinsics=intrinsics, detected_objects=[])
    payload = asdict(state)
    payload["frame_index"] = frame.frame_index
    payload["color_timestamp"] = frame.color_timestamp
    payload["depth_timestamp"] = frame.depth_timestamp
    payload["color_path"] = str(frame.color_path)
    payload["aligned_depth_path"] = str(frame.aligned_depth_path)
    payload["groundtruth"] = frame.groundtruth.tolist() if frame.groundtruth is not None else None

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
