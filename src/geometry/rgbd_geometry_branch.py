from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

import numpy as np

from src.geometry.spatial_language import euclidean_distance_3d
from src.perception.observation_protocol import ObservedObject


@dataclass
class WallEstimate:
    side: str
    lateral_distance_m: float
    forward_distance_m: float
    distance_m: float
    valid_point_count: int


@dataclass
class ObstacleEstimate:
    object_id: str
    category: str
    forward_distance_m: float
    lateral_offset_m: float
    distance_m: float
    width_m: float
    source: str


@dataclass
class CorridorGap:
    start_m: float
    end_m: float
    width_m: float


@dataclass
class SceneGeometryState:
    reference_frame: str
    robot_width_m: float
    safety_margin_m: float
    left_wall: WallEstimate | None
    right_wall: WallEstimate | None
    corridor_width_m: float | None
    traversable_width_m: float | None
    passable: bool | None
    nearest_obstacle_distance_m: float | None
    nearest_obstacle: ObstacleEstimate | None
    gaps: List[CorridorGap] = field(default_factory=list)
    blocking_obstacle_ids: List[str] = field(default_factory=list)


class RGBDGeometryBranch:
    """Compute robot-centric corridor geometry from depth and detected objects."""

    def __init__(
        self,
        robot_width_m: float = 0.55,
        safety_margin_m: float = 0.15,
        max_obstacle_forward_m: float = 3.0,
    ) -> None:
        self.robot_width_m = robot_width_m
        self.safety_margin_m = safety_margin_m
        self.max_obstacle_forward_m = max_obstacle_forward_m

    def estimate(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        detected_objects: Iterable[ObservedObject] | None = None,
    ) -> SceneGeometryState:
        depth_clean = self._sanitize_depth(depth_m)
        walls = self._extract_corridor_walls(depth_clean, intrinsics)
        left_wall = next((wall for wall in walls if wall.side == "left"), None)
        right_wall = next((wall for wall in walls if wall.side == "right"), None)

        corridor_width = None
        if left_wall is not None and right_wall is not None:
            corridor_width = round(left_wall.lateral_distance_m + right_wall.lateral_distance_m, 3)

        obstacles = list(detected_objects or [])
        nearest_obstacle = self._nearest_obstacle(obstacles)
        gaps, blocking_ids, traversable_width = self._estimate_free_space(left_wall, right_wall, obstacles)

        passable = None
        if traversable_width is not None:
            passable = traversable_width >= (self.robot_width_m + self.safety_margin_m)
        elif corridor_width is not None:
            passable = corridor_width >= (self.robot_width_m + self.safety_margin_m)

        return SceneGeometryState(
            reference_frame="robot_base",
            robot_width_m=self.robot_width_m,
            safety_margin_m=self.safety_margin_m,
            left_wall=left_wall,
            right_wall=right_wall,
            corridor_width_m=corridor_width,
            traversable_width_m=traversable_width,
            passable=passable,
            nearest_obstacle_distance_m=nearest_obstacle.distance_m if nearest_obstacle else None,
            nearest_obstacle=nearest_obstacle,
            gaps=gaps,
            blocking_obstacle_ids=blocking_ids,
        )

    def _sanitize_depth(self, depth_m: np.ndarray) -> np.ndarray:
        depth_clean = np.asarray(depth_m, dtype=np.float32).copy()
        invalid = ~np.isfinite(depth_clean) | (depth_clean <= 0.05) | (depth_clean >= 10.0)
        depth_clean[invalid] = np.nan
        return depth_clean

    def _extract_corridor_walls(self, depth_m: np.ndarray, intrinsics: np.ndarray) -> List[WallEstimate]:
        estimates: List[WallEstimate] = []
        for side in ("left", "right"):
            estimate = self._estimate_side_wall(depth_m, intrinsics, side)
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    def _estimate_side_wall(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        side: str,
    ) -> WallEstimate | None:
        height, width = depth_m.shape
        y_start = int(height * 0.28)
        y_end = int(height * 0.78)
        x_start, x_end = (0, int(width * 0.22)) if side == "left" else (int(width * 0.78), width)

        roi = depth_m[y_start:y_end, x_start:x_end]
        valid = np.isfinite(roi)
        if int(valid.sum()) < 40:
            return None

        _, u_coords = np.where(valid)
        z = roi[valid]
        u = u_coords.astype(np.float32) + x_start
        x_camera = ((u - intrinsics[0, 2]) / intrinsics[0, 0]) * z

        side_mask = x_camera < 0 if side == "left" else x_camera > 0
        if int(side_mask.sum()) < 25:
            return None

        lateral = float(np.nanmedian(np.abs(x_camera[side_mask])))
        forward = float(np.nanmedian(z[side_mask]))
        return WallEstimate(
            side=side,
            lateral_distance_m=round(lateral, 3),
            forward_distance_m=round(forward, 3),
            distance_m=round(float(np.sqrt(lateral**2 + forward**2)), 3),
            valid_point_count=int(side_mask.sum()),
        )

    def _nearest_obstacle(self, objects: List[ObservedObject]) -> ObstacleEstimate | None:
        if not objects:
            return None

        nearest = min(objects, key=lambda item: euclidean_distance_3d(item.position_robot_frame))
        forward, lateral, _ = nearest.position_robot_frame
        return ObstacleEstimate(
            object_id=nearest.object_id,
            category=nearest.category,
            forward_distance_m=round(float(forward), 3),
            lateral_offset_m=round(float(lateral), 3),
            distance_m=round(euclidean_distance_3d(nearest.position_robot_frame), 3),
            width_m=round(float(nearest.size[1]), 3),
            source=nearest.attributes.get("source", "detector"),
        )

    def _estimate_free_space(
        self,
        left_wall: WallEstimate | None,
        right_wall: WallEstimate | None,
        objects: List[ObservedObject],
    ) -> tuple[List[CorridorGap], List[str], float | None]:
        if left_wall is None or right_wall is None:
            return [], [], None

        left_boundary = float(left_wall.lateral_distance_m)
        right_boundary = -float(right_wall.lateral_distance_m)

        occupied: List[tuple[float, float, str]] = []
        blocking_ids: List[str] = []
        for obj in objects:
            forward, lateral, _ = obj.position_robot_frame
            if forward <= 0.0 or forward > self.max_obstacle_forward_m:
                continue

            half_width = max(0.05, float(obj.size[1]) / 2.0)
            start = max(right_boundary, float(lateral) - half_width)
            end = min(left_boundary, float(lateral) + half_width)
            if end <= start:
                continue

            occupied.append((start, end, obj.object_id))
            blocking_ids.append(obj.object_id)

        merged = self._merge_intervals(occupied)
        gaps = self._compute_gaps(right_boundary, left_boundary, merged)
        if not gaps:
            return [], blocking_ids, 0.0

        traversable_width = round(max(gap.width_m for gap in gaps), 3)
        return gaps, blocking_ids, traversable_width

    def _merge_intervals(self, intervals: List[tuple[float, float, str]]) -> List[tuple[float, float]]:
        if not intervals:
            return []

        sorted_intervals = sorted((start, end) for start, end, _ in intervals)
        merged: List[tuple[float, float]] = [sorted_intervals[0]]

        for start, end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def _compute_gaps(
        self,
        right_boundary: float,
        left_boundary: float,
        occupied_intervals: List[tuple[float, float]],
    ) -> List[CorridorGap]:
        gaps: List[CorridorGap] = []
        cursor = right_boundary

        for start, end in occupied_intervals:
            if start > cursor:
                gaps.append(
                    CorridorGap(
                        start_m=round(cursor, 3),
                        end_m=round(start, 3),
                        width_m=round(start - cursor, 3),
                    )
                )
            cursor = max(cursor, end)

        if cursor < left_boundary:
            gaps.append(
                CorridorGap(
                    start_m=round(cursor, 3),
                    end_m=round(left_boundary, 3),
                    width_m=round(left_boundary - cursor, 3),
                )
            )
        return gaps
