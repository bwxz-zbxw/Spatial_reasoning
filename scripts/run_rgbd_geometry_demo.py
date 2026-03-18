import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.rgbd_geometry_branch import RGBDGeometryBranch
from src.perception.observation_protocol import ObservedObject


def build_synthetic_depth(height: int = 480, width: int = 640) -> tuple[np.ndarray, np.ndarray]:
    fx = 520.0
    fy = 520.0
    cx = width / 2.0
    cy = height / 2.0
    intrinsics = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    depth_m = np.full((height, width), np.nan, dtype=np.float32)
    left_lateral = 0.62
    right_lateral = 0.68

    y_start = int(height * 0.28)
    y_end = int(height * 0.78)

    for u in range(0, int(width * 0.22)):
        x_term = abs((u - cx) / fx)
        if x_term < 1e-6:
            continue
        depth_m[y_start:y_end, u] = left_lateral / x_term

    for u in range(int(width * 0.78), width):
        x_term = abs((u - cx) / fx)
        if x_term < 1e-6:
            continue
        depth_m[y_start:y_end, u] = right_lateral / x_term

    center_depth = 3.5
    depth_m[y_start:y_end, int(width * 0.22):int(width * 0.78)] = center_depth
    return depth_m, intrinsics


def build_demo_objects() -> list[ObservedObject]:
    return [
        ObservedObject(
            object_id="person_1",
            category="human",
            position_robot_frame=(1.8, -0.12, 0.0),
            size=(0.4, 0.48, 1.7),
            bbox_xyxy=(260.0, 120.0, 340.0, 410.0),
            confidence=0.94,
            attributes={"source": "synthetic_demo"},
        ),
        ObservedObject(
            object_id="cart_1",
            category="cart",
            position_robot_frame=(2.4, 0.3, 0.0),
            size=(0.7, 0.55, 1.1),
            bbox_xyxy=(350.0, 160.0, 430.0, 390.0),
            confidence=0.89,
            attributes={"source": "synthetic_demo"},
        ),
    ]


def main() -> None:
    depth_m, intrinsics = build_synthetic_depth()
    objects = build_demo_objects()

    branch = RGBDGeometryBranch(robot_width_m=0.55, safety_margin_m=0.15)
    state = branch.estimate(depth_m=depth_m, intrinsics=intrinsics, detected_objects=objects)

    print(json.dumps(asdict(state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
