import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.rgbd_geometry_branch import ObstacleEstimate, SceneGeometryState
from src.perception.fused_geometry_pipeline import FusedGeometryResult, FusedGeometryPipeline
from src.perception.openloris_loader import OpenLORISLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay fused geometry + GCA results on OpenLORIS RGB frames.")
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
        help="Frame stride for visualization output.",
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
        default=Path("results/geometry_visualizations"),
        help="Directory for annotated PNG outputs.",
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

    last_frame = min(len(frames) - 1, args.start_frame + args.num_frames - 1)
    intrinsics = loader.load_color_intrinsics(args.sequence_dir)
    pipeline = FusedGeometryPipeline(
        robot_width_m=args.robot_width_m,
        safety_margin_m=args.safety_margin_m,
    )
    pipeline.reset_tracks()

    warmup_start = max(0, min(args.warmup_start_frame, args.start_frame))
    output_dir = args.output_dir / args.sequence_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames[warmup_start:args.start_frame]:
        color_rgb = loader.load_color_rgb(frame)
        depth_m = loader.load_aligned_depth_meters(frame)
        pipeline.run(rgb_image=color_rgb, depth_m=depth_m, intrinsics=intrinsics, frame_index=frame.frame_index)

    for frame in frames[args.start_frame:last_frame + 1]:
        color_rgb = loader.load_color_rgb(frame)
        depth_m = loader.load_aligned_depth_meters(frame)
        result = pipeline.run(rgb_image=color_rgb, depth_m=depth_m, intrinsics=intrinsics, frame_index=frame.frame_index)
        if (frame.frame_index - args.start_frame) % max(1, args.stride) != 0:
            continue
        annotated = _draw_overlay(color_rgb, intrinsics, result, frame.frame_index)
        output_path = output_dir / f"frame_{frame.frame_index:05d}.png"
        annotated.save(output_path)
        print(f"Saved overlay: {output_path}")


def _draw_overlay(
    image_rgb: np.ndarray,
    intrinsics: np.ndarray,
    result: FusedGeometryResult,
    frame_index: int,
) -> Image.Image:
    state = result.geometry_state
    constrained_state = result.gca_result.constrained_state

    image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    width, height = image.size
    left_roi = (0, int(height * 0.28), int(width * 0.22), int(height * 0.78))
    right_roi = (int(width * 0.78), int(height * 0.28), width - 1, int(height * 0.78))

    _draw_wall_roi(
        draw,
        left_roi,
        "left",
        state.left_wall,
        state.left_wall_visible,
        state.left_wall_confidence,
        result.gca_result.facts.get("left_wall_normal_z_abs"),
        font,
    )
    _draw_wall_roi(
        draw,
        right_roi,
        "right",
        state.right_wall,
        state.right_wall_visible,
        state.right_wall_confidence,
        result.gca_result.facts.get("right_wall_normal_z_abs"),
        font,
    )

    for obstacle in constrained_state.obstacles:
        _draw_obstacle(draw, intrinsics, obstacle, width, height, font)

    _draw_summary(draw, constrained_state, frame_index, width, font)
    _draw_gca_panel(draw, result, width, height, font)
    return image


def _draw_wall_roi(
    draw: ImageDraw.ImageDraw,
    roi: tuple[int, int, int, int],
    side: str,
    wall,
    visible: bool | None,
    confidence: float | None,
    wall_normal_z_abs: float | None,
    font: ImageFont.ImageFont,
) -> None:
    color = (60, 170, 255) if side == "left" else (255, 140, 60)
    draw.rectangle(roi, outline=color, width=3)

    label_y = max(4, roi[1] - 28)
    status = "visible" if visible else "tracked"
    if wall is None:
        text = f"{side}: none"
    else:
        conf_text = "n/a" if confidence is None else f"{confidence:.2f}"
        nz_text = "n/a" if wall_normal_z_abs is None else f"|nz| {wall_normal_z_abs:.02f}"
        text = f"{side}: {wall.lateral_distance_m:.2f}m | {status} | conf {conf_text} | {nz_text}"
    draw.text((roi[0] + 4, label_y), text, fill=color, font=font)


def _draw_obstacle(
    draw: ImageDraw.ImageDraw,
    intrinsics: np.ndarray,
    obstacle: ObstacleEstimate,
    image_width: int,
    image_height: int,
    font: ImageFont.ImageFont,
) -> None:
    if obstacle.forward_distance_m <= 0.05:
        return

    fx = float(intrinsics[0, 0])
    cx = float(intrinsics[0, 2])
    center_u = cx - (fx * obstacle.lateral_offset_m / obstacle.forward_distance_m)
    box_width_px = max(24.0, (fx * obstacle.width_m / obstacle.forward_distance_m))
    box_height_px = max(48.0, box_width_px * 1.4)

    x1 = int(max(0, center_u - (box_width_px / 2.0)))
    x2 = int(min(image_width - 1, center_u + (box_width_px / 2.0)))
    y2 = int(image_height * 0.80)
    y1 = int(max(0, y2 - box_height_px))

    color = (255, 210, 0) if obstacle.source == "depth_geometry" else (0, 255, 160)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    text = f"{obstacle.category} {obstacle.forward_distance_m:.2f}m {obstacle.lateral_offset_m:+.2f}m"
    draw.text((x1 + 2, max(0, y1 - 14)), text, fill=color, font=font)


def _draw_summary(
    draw: ImageDraw.ImageDraw,
    state: SceneGeometryState,
    frame_index: int,
    image_width: int,
    font: ImageFont.ImageFont,
) -> None:
    lines = [
        f"frame {frame_index}",
        f"corridor_width={_fmt_float(state.corridor_width_m)} m",
        f"traversable_width={_fmt_float(state.traversable_width_m)} m",
        f"passable={state.passable}",
        f"obstacles={len(state.obstacles)}",
    ]
    if state.nearest_obstacle is not None:
        lines.append(
            "nearest="
            f"{state.nearest_obstacle.distance_m:.2f}m "
            f"({state.nearest_obstacle.lateral_offset_m:+.2f}m)"
        )

    box_x1 = max(0, image_width - 300)
    box_y1 = 10
    box_x2 = image_width - 10
    box_y2 = box_y1 + 18 * len(lines) + 10
    draw.rectangle((box_x1, box_y1, box_x2, box_y2), fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    for idx, line in enumerate(lines):
        draw.text((box_x1 + 8, box_y1 + 6 + (18 * idx)), line, fill=(255, 255, 255), font=font)


def _draw_gca_panel(
    draw: ImageDraw.ImageDraw,
    result: FusedGeometryResult,
    image_width: int,
    image_height: int,
    font: ImageFont.ImageFont,
) -> None:
    facts = result.gca_result.facts
    evals = {item.name: item for item in result.gca_result.evaluations}
    panel_lines = [
        ("GCA", result.gca_result.geometry_valid),
        (f"ground_z={_fmt_float(facts.get('ground_normal_z'))}", evals.get("ground_normal_z").passed if evals.get("ground_normal_z") else False),
        (
            f"left_wall_vertical={_fmt_float(facts.get('left_wall_normal_z_abs'))}",
            evals.get("left_wall_normal_z_abs").passed if evals.get("left_wall_normal_z_abs") else False,
        ),
        (
            f"right_wall_vertical={_fmt_float(facts.get('right_wall_normal_z_abs'))}",
            evals.get("right_wall_normal_z_abs").passed if evals.get("right_wall_normal_z_abs") else False,
        ),
        (
            f"wall_parallel={_fmt_float(facts.get('wall_parallel_abs_cos'))}",
            evals.get("wall_parallel_abs_cos").passed if evals.get("wall_parallel_abs_cos") else False,
        ),
        (
            f"required_width={_fmt_float(result.gca_result.required_width_m)}",
            True,
        ),
    ]

    box_x1 = max(0, image_width - 320)
    box_y2 = image_height - 10
    box_y1 = max(0, box_y2 - (20 * len(panel_lines)) - 12)
    draw.rectangle((box_x1, box_y1, image_width - 10, box_y2), fill=(0, 0, 0), outline=(255, 255, 255), width=2)

    for idx, (text, passed) in enumerate(panel_lines):
        y = box_y1 + 6 + (20 * idx)
        bullet_color = (0, 220, 120) if passed else (255, 80, 80)
        draw.ellipse((box_x1 + 8, y + 4, box_x1 + 16, y + 12), fill=bullet_color)
        draw.text((box_x1 + 22, y), text, fill=(255, 255, 255), font=font)


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
