import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.rgbd_geometry_branch import RGBDGeometryBranch
from src.perception.openloris_loader import OpenLORISLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OpenLORIS geometry outputs over a frame range.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Path to an OpenLORIS packaged sequence, e.g. corridor1-1",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start frame index.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=200,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/geometry_timeseries"),
        help="Directory for CSV and plot outputs.",
    )
    args = parser.parse_args()

    loader = OpenLORISLoader()
    frames = loader.load_sequence(args.sequence_dir)
    if not frames:
        raise ValueError(f"No frames found in {args.sequence_dir}")

    end_frame = min(len(frames), args.start_frame + args.num_frames)
    if args.start_frame < 0 or args.start_frame >= len(frames):
        raise IndexError(f"start-frame {args.start_frame} out of range 0..{len(frames) - 1}")

    selected_frames = frames[args.start_frame:end_frame]
    intrinsics = loader.load_color_intrinsics(args.sequence_dir)
    branch = RGBDGeometryBranch(
        robot_width_m=args.robot_width_m,
        safety_margin_m=args.safety_margin_m,
    )
    branch.reset_tracks()

    rows = []
    for frame in selected_frames:
        depth_m = loader.load_aligned_depth_meters(frame)
        state = branch.estimate(
            depth_m=depth_m,
            intrinsics=intrinsics,
            detected_objects=[],
            frame_index=frame.frame_index,
        )
        rows.append(
            {
                "frame_index": frame.frame_index,
                "color_timestamp": frame.color_timestamp,
                "depth_timestamp": frame.depth_timestamp,
                "left_wall_distance_m": state.left_wall.lateral_distance_m if state.left_wall else None,
                "right_wall_distance_m": state.right_wall.lateral_distance_m if state.right_wall else None,
                "corridor_width_m": state.corridor_width_m,
                "traversable_width_m": state.traversable_width_m,
                "passable": state.passable,
                "nearest_obstacle_distance_m": state.nearest_obstacle.distance_m if state.nearest_obstacle else None,
                "nearest_obstacle_forward_m": state.nearest_obstacle.forward_distance_m if state.nearest_obstacle else None,
                "nearest_obstacle_lateral_m": state.nearest_obstacle.lateral_offset_m if state.nearest_obstacle else None,
                "nearest_obstacle_width_m": state.nearest_obstacle.width_m if state.nearest_obstacle else None,
                "nearest_obstacle_source": state.nearest_obstacle.source if state.nearest_obstacle else None,
                "blocking_obstacle_count": len(state.blocking_obstacle_ids),
                "left_wall_forward_m": state.left_wall.forward_distance_m if state.left_wall else None,
                "right_wall_forward_m": state.right_wall.forward_distance_m if state.right_wall else None,
                "left_wall_visible": branch._wall_tracks.get("left").visible if branch._wall_tracks.get("left") else None,
                "right_wall_visible": branch._wall_tracks.get("right").visible if branch._wall_tracks.get("right") else None,
                "left_wall_confidence": branch._wall_tracks.get("left").confidence if branch._wall_tracks.get("left") else None,
                "right_wall_confidence": branch._wall_tracks.get("right").confidence if branch._wall_tracks.get("right") else None,
            }
        )

    df = pd.DataFrame(rows)
    output_dir = args.output_dir / args.sequence_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"frames_{args.start_frame}_{end_frame - 1}.csv"
    plot_path = output_dir / f"frames_{args.start_frame}_{end_frame - 1}.png"
    stats_path = output_dir / f"frames_{args.start_frame}_{end_frame - 1}_stats.txt"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    _plot_timeseries(df, plot_path, args.sequence_dir.name)
    _write_stats(df, stats_path)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved stats: {stats_path}")
    print(_stats_summary(df))


def _plot_timeseries(df: pd.DataFrame, plot_path: Path, sequence_name: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(df["frame_index"], df["left_wall_distance_m"], label="left wall", linewidth=1.5)
    axes[0].plot(df["frame_index"], df["right_wall_distance_m"], label="right wall", linewidth=1.5)
    axes[0].set_ylabel("Wall Distance (m)")
    axes[0].set_title(f"OpenLORIS Geometry Timeseries: {sequence_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["frame_index"], df["corridor_width_m"], label="corridor width", linewidth=1.5)
    axes[1].plot(df["frame_index"], df["traversable_width_m"], label="traversable width", linewidth=1.5)
    axes[1].set_ylabel("Width (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    passable_numeric = df["passable"].map({True: 1, False: 0})
    axes[2].step(df["frame_index"], passable_numeric, where="mid", label="passable", linewidth=1.5)
    axes[2].set_ylabel("Passable")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["false", "true"])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(df["frame_index"], df["nearest_obstacle_distance_m"], label="nearest obstacle distance", linewidth=1.5)
    axes[3].plot(df["frame_index"], df["nearest_obstacle_lateral_m"], label="nearest obstacle lateral", linewidth=1.5)
    axes[3].set_xlabel("Frame Index")
    axes[3].set_ylabel("Obstacle")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _write_stats(df: pd.DataFrame, stats_path: Path) -> None:
    stats_path.write_text(_stats_summary(df), encoding="utf-8")


def _stats_summary(df: pd.DataFrame) -> str:
    fields = [
        "left_wall_distance_m",
        "right_wall_distance_m",
        "corridor_width_m",
        "traversable_width_m",
        "nearest_obstacle_distance_m",
        "nearest_obstacle_lateral_m",
    ]
    lines = []
    for field in fields:
        series = df[field].dropna()
        if series.empty:
            lines.append(f"{field}: no valid data")
            continue
        lines.append(
            (
                f"{field}: count={len(series)}, mean={series.mean():.3f}, std={series.std(ddof=0):.3f}, "
                f"min={series.min():.3f}, max={series.max():.3f}"
            )
        )

    passable_series = df["passable"].dropna()
    if passable_series.empty:
        lines.append("passable: no valid data")
    else:
        true_ratio = float(passable_series.astype(bool).mean())
        transitions = int(passable_series.astype(str).ne(passable_series.astype(str).shift()).sum() - 1)
        lines.append(
            f"passable: true_ratio={true_ratio:.3f}, transitions={max(0, transitions)}, valid_frames={len(passable_series)}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
