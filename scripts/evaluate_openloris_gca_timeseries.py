import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.fused_geometry_pipeline import FusedGeometryPipeline
from src.perception.openloris_loader import OpenLORISLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fused geometry + GCA constraints over an OpenLORIS frame range.")
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
        default=120,
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
        default=Path("results/gca_timeseries"),
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

    intrinsics = loader.load_color_intrinsics(args.sequence_dir)
    pipeline = FusedGeometryPipeline(
        robot_width_m=args.robot_width_m,
        safety_margin_m=args.safety_margin_m,
    )
    pipeline.reset_tracks()

    rows = []
    for frame in frames[args.start_frame:end_frame]:
        rgb = loader.load_color_rgb(frame)
        depth_m = loader.load_aligned_depth_meters(frame)
        result = pipeline.run(
            rgb_image=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            frame_index=frame.frame_index,
        )
        raw_state = result.geometry_state
        gca_state = result.gca_result.constrained_state
        gca_facts = result.gca_result.facts
        rows.append(
            {
                "frame_index": frame.frame_index,
                "color_timestamp": frame.color_timestamp,
                "raw_left_wall_m": raw_state.left_wall.lateral_distance_m if raw_state.left_wall else None,
                "raw_right_wall_m": raw_state.right_wall.lateral_distance_m if raw_state.right_wall else None,
                "raw_corridor_width_m": raw_state.corridor_width_m,
                "raw_traversable_width_m": raw_state.traversable_width_m,
                "raw_passable": raw_state.passable,
                "gca_left_wall_m": gca_state.left_wall.lateral_distance_m if gca_state.left_wall else None,
                "gca_right_wall_m": gca_state.right_wall.lateral_distance_m if gca_state.right_wall else None,
                "gca_corridor_width_m": gca_state.corridor_width_m,
                "gca_traversable_width_m": gca_state.traversable_width_m,
                "gca_passable": gca_state.passable,
                "gca_geometry_valid": result.gca_result.geometry_valid,
                "ground_normal_z": gca_facts.get("ground_normal_z"),
                "left_wall_normal_z_abs": gca_facts.get("left_wall_normal_z_abs"),
                "right_wall_normal_z_abs": gca_facts.get("right_wall_normal_z_abs"),
                "wall_parallel_abs_cos": gca_facts.get("wall_parallel_abs_cos"),
                "left_wall_confidence": raw_state.left_wall_confidence,
                "right_wall_confidence": raw_state.right_wall_confidence,
                "nearest_obstacle_distance_m": raw_state.nearest_obstacle.distance_m if raw_state.nearest_obstacle else None,
                "nearest_obstacle_lateral_m": raw_state.nearest_obstacle.lateral_offset_m if raw_state.nearest_obstacle else None,
                "required_width_m": result.gca_result.required_width_m,
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

    axes[0].plot(df["frame_index"], df["raw_corridor_width_m"], label="raw corridor width", linewidth=1.4, alpha=0.8)
    axes[0].plot(df["frame_index"], df["gca_corridor_width_m"], label="gca corridor width", linewidth=1.6)
    axes[0].plot(df["frame_index"], df["required_width_m"], label="required width", linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("Width (m)")
    axes[0].set_title(f"OpenLORIS GCA Timeseries: {sequence_name}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["frame_index"], df["raw_traversable_width_m"], label="raw traversable", linewidth=1.4, alpha=0.8)
    axes[1].plot(df["frame_index"], df["gca_traversable_width_m"], label="gca traversable", linewidth=1.6)
    axes[1].set_ylabel("Free Width (m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    validity = df["gca_geometry_valid"].map({True: 1, False: 0})
    passable = df["gca_passable"].map({True: 1, False: 0})
    axes[2].step(df["frame_index"], validity, where="mid", label="geometry valid", linewidth=1.6)
    axes[2].step(df["frame_index"], passable, where="mid", label="gca passable", linewidth=1.4)
    axes[2].set_ylabel("State")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["false", "true"])
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(df["frame_index"], df["ground_normal_z"], label="ground normal z", linewidth=1.5)
    axes[3].plot(df["frame_index"], df["wall_parallel_abs_cos"], label="wall parallel |cos|", linewidth=1.5)
    axes[3].plot(df["frame_index"], df["left_wall_normal_z_abs"], label="left wall |nz|", linewidth=1.2)
    axes[3].plot(df["frame_index"], df["right_wall_normal_z_abs"], label="right wall |nz|", linewidth=1.2)
    axes[3].set_xlabel("Frame Index")
    axes[3].set_ylabel("Constraint Metrics")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _write_stats(df: pd.DataFrame, stats_path: Path) -> None:
    stats_path.write_text(_stats_summary(df), encoding="utf-8")


def _stats_summary(df: pd.DataFrame) -> str:
    fields = [
        "raw_corridor_width_m",
        "gca_corridor_width_m",
        "raw_traversable_width_m",
        "gca_traversable_width_m",
        "ground_normal_z",
        "left_wall_normal_z_abs",
        "right_wall_normal_z_abs",
        "wall_parallel_abs_cos",
        "nearest_obstacle_distance_m",
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

    for field in ("gca_geometry_valid", "gca_passable"):
        series = df[field].dropna()
        if series.empty:
            lines.append(f"{field}: no valid data")
            continue
        true_ratio = float(series.astype(bool).mean())
        transitions = int(series.astype(str).ne(series.astype(str).shift()).sum() - 1)
        lines.append(f"{field}: true_ratio={true_ratio:.3f}, transitions={max(0, transitions)}, valid_frames={len(series)}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
