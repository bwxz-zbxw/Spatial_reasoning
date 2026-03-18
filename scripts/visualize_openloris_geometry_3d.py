import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.rgbd_geometry_branch import RGBDGeometryBranch, ObstacleEstimate, SceneGeometryState
from src.perception.openloris_loader import OpenLORISLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OpenLORIS geometry state in 3D.")
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Path to an OpenLORIS packaged sequence, e.g. corridor1-1",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        required=True,
        help="First frame index to visualize.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to visualize.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride for saved outputs.",
    )
    parser.add_argument(
        "--warmup-start-frame",
        type=int,
        default=0,
        help="Frame index to start track warmup from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/geometry_visualizations_3d"),
        help="Directory for rendered PNG outputs.",
    )
    parser.add_argument(
        "--point-stride",
        type=int,
        default=8,
        help="Subsampling stride for depth points.",
    )
    args = parser.parse_args()

    loader = OpenLORISLoader()
    frames = loader.load_sequence(args.sequence_dir)
    if not frames:
        raise ValueError(f"No frames found in {args.sequence_dir}")
    if args.start_frame < 0 or args.start_frame >= len(frames):
        raise IndexError(f"start-frame {args.start_frame} out of range 0..{len(frames) - 1}")

    last_frame = min(len(frames) - 1, args.start_frame + args.num_frames - 1)
    intrinsics = loader.load_color_intrinsics(args.sequence_dir)
    branch = RGBDGeometryBranch()
    branch.reset_tracks()

    warmup_start = max(0, min(args.warmup_start_frame, args.start_frame))
    output_dir = args.output_dir / args.sequence_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames[warmup_start:args.start_frame]:
        depth_m = loader.load_aligned_depth_meters(frame)
        branch.estimate(depth_m=depth_m, intrinsics=intrinsics, detected_objects=[], frame_index=frame.frame_index)

    for frame in frames[args.start_frame:last_frame + 1]:
        depth_m = loader.load_aligned_depth_meters(frame)
        state = branch.estimate(depth_m=depth_m, intrinsics=intrinsics, detected_objects=[], frame_index=frame.frame_index)
        if (frame.frame_index - args.start_frame) % max(1, args.stride) != 0:
            continue

        points = depth_to_robot_points(depth_m, intrinsics, stride=max(1, args.point_stride))
        output_path = output_dir / f"frame_{frame.frame_index:05d}_3d.png"
        render_3d(points, state, frame.frame_index, output_path)
        print(f"Saved 3D view: {output_path}")


def depth_to_robot_points(depth_m: np.ndarray, intrinsics: np.ndarray, stride: int = 8) -> np.ndarray:
    sampled = depth_m[::stride, ::stride]
    valid = np.isfinite(sampled)
    if int(valid.sum()) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    rows, cols = np.where(valid)
    z = sampled[valid].astype(np.float32)

    u = (cols.astype(np.float32) * stride)
    v = (rows.astype(np.float32) * stride)

    x_camera = ((u - intrinsics[0, 2]) / intrinsics[0, 0]) * z
    y_camera = ((v - intrinsics[1, 2]) / intrinsics[1, 1]) * z

    forward = z
    lateral = -x_camera
    vertical = -y_camera

    mask = (
        (forward > 0.2)
        & (forward < 4.0)
        & (vertical > -0.3)
        & (vertical < 2.0)
        & (np.abs(lateral) < 3.0)
    )
    return np.stack([forward[mask], lateral[mask], vertical[mask]], axis=1)


def render_3d(points: np.ndarray, state: SceneGeometryState, frame_index: int, output_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_top = fig.add_subplot(1, 2, 2)

    if len(points) > 0:
        colors = np.clip(points[:, 0] / 4.0, 0.0, 1.0)
        ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap="viridis", s=2, alpha=0.45)
        ax_top.scatter(points[:, 0], points[:, 1], c=colors, cmap="viridis", s=2, alpha=0.25)

    _draw_walls_3d(ax3d, ax_top, state)
    _draw_obstacles_3d(ax3d, ax_top, state.obstacles)
    _draw_robot(ax3d, ax_top, state.robot_width_m)

    ax3d.set_title(f"3D Geometry State - frame {frame_index}")
    ax3d.set_xlabel("Forward (m)")
    ax3d.set_ylabel("Lateral (m)")
    ax3d.set_zlabel("Vertical (m)")
    ax3d.set_xlim(0.0, 4.0)
    ax3d.set_ylim(-2.5, 2.5)
    ax3d.set_zlim(-0.2, 2.0)
    ax3d.view_init(elev=22, azim=-62)

    ax_top.set_title("Top-down View")
    ax_top.set_xlabel("Forward (m)")
    ax_top.set_ylabel("Lateral (m)")
    ax_top.set_xlim(0.0, 4.0)
    ax_top.set_ylim(-2.5, 2.5)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_aspect("equal", adjustable="box")

    summary = [
        f"corridor_width={_fmt(state.corridor_width_m)} m",
        f"traversable_width={_fmt(state.traversable_width_m)} m",
        f"passable={state.passable}",
        f"left_wall={_fmt(state.left_wall.lateral_distance_m if state.left_wall else None)} m",
        f"right_wall={_fmt(state.right_wall.lateral_distance_m if state.right_wall else None)} m",
        f"obstacles={len(state.obstacles)}",
    ]
    fig.text(
        0.51,
        0.02,
        "\n".join(summary),
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "black", "alpha": 0.75, "edgecolor": "white", "pad": 8},
        color="white",
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _draw_walls_3d(ax3d, ax_top, state: SceneGeometryState) -> None:
    forward = np.linspace(0.0, 4.0, 25)
    vertical = np.linspace(0.0, 1.8, 10)
    f_grid, z_grid = np.meshgrid(forward, vertical)

    if state.left_wall is not None:
        y_left = np.full_like(f_grid, state.left_wall.lateral_distance_m)
        color = "#40a9ff"
        ax3d.plot_surface(f_grid, y_left, z_grid, alpha=0.18, color=color, linewidth=0)
        ax_top.plot(forward, np.full_like(forward, state.left_wall.lateral_distance_m), color=color, linewidth=2)

    if state.right_wall is not None:
        y_right = np.full_like(f_grid, -state.right_wall.lateral_distance_m)
        color = "#ff8c3c"
        ax3d.plot_surface(f_grid, y_right, z_grid, alpha=0.18, color=color, linewidth=0)
        ax_top.plot(forward, np.full_like(forward, -state.right_wall.lateral_distance_m), color=color, linewidth=2)


def _draw_obstacles_3d(ax3d, ax_top, obstacles: list[ObstacleEstimate]) -> None:
    for obstacle in obstacles:
        draw_box(
            ax3d,
            center=(obstacle.forward_distance_m, obstacle.lateral_offset_m, 0.4),
            size=(0.45, max(0.12, obstacle.width_m), 0.8),
            color="#ffd200" if obstacle.source == "depth_geometry" else "#00ffa0",
        )
        x = obstacle.forward_distance_m
        y = obstacle.lateral_offset_m
        half_w = max(0.06, obstacle.width_m / 2.0)
        ax_top.add_patch(plt.Rectangle((x - 0.225, y - half_w), 0.45, 2 * half_w, fill=False, color="#ffd200", linewidth=2))
        ax_top.text(x, y, obstacle.category, color="#ffd200", fontsize=8)


def _draw_robot(ax3d, ax_top, robot_width_m: float) -> None:
    draw_box(ax3d, center=(0.0, 0.0, 0.18), size=(0.45, robot_width_m, 0.36), color="#ff4d4f")
    ax_top.add_patch(plt.Rectangle((-0.225, -robot_width_m / 2.0), 0.45, robot_width_m, fill=False, color="#ff4d4f", linewidth=2))


def draw_box(ax, center: tuple[float, float, float], size: tuple[float, float, float], color: str) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    x = [cx - sx / 2.0, cx + sx / 2.0]
    y = [cy - sy / 2.0, cy + sy / 2.0]
    z = [cz - sz / 2.0, cz + sz / 2.0]

    corners = np.array(
        [
            [x[0], y[0], z[0]],
            [x[1], y[0], z[0]],
            [x[1], y[1], z[0]],
            [x[0], y[1], z[0]],
            [x[0], y[0], z[1]],
            [x[1], y[0], z[1]],
            [x[1], y[1], z[1]],
            [x[0], y[1], z[1]],
        ]
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color,
            linewidth=1.6,
        )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
