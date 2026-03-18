import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.open3d_geometry import (
    cluster_obstacles,
    depth_to_robot_point_cloud,
    remove_floor_and_far_background,
)
from src.perception.openloris_loader import OpenLORISLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Open3D obstacle clustering on one OpenLORIS frame.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Path to an OpenLORIS packaged sequence, e.g. corridor1-1",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        required=True,
        help="Frame index to inspect.",
    )
    args = parser.parse_args()

    loader = OpenLORISLoader()
    frames = loader.load_sequence(args.sequence_dir)
    frame = frames[args.frame_index]
    intrinsics = loader.load_color_intrinsics(args.sequence_dir)
    color_rgb = loader.load_color_rgb(frame)
    depth_m = loader.load_aligned_depth_meters(frame)

    pcd, points = depth_to_robot_point_cloud(depth_m=depth_m, intrinsics=intrinsics, color_rgb=color_rgb, stride=2)
    filtered = remove_floor_and_far_background(pcd)
    clusters = cluster_obstacles(filtered)

    payload = {
        "frame_index": frame.frame_index,
        "point_count": int(len(points)),
        "filtered_point_count": int(len(filtered.points)),
        "clusters": [asdict(cluster) for cluster in clusters],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
