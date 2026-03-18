from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List

import numpy as np

from src.geometry.rgbd_geometry_branch import SceneGeometryState, WallEstimate
from src.reasoning.protocol import ConstraintEvaluation


@dataclass
class GCAConstraintConfig:
    min_ground_normal_z: float = 0.90
    min_wall_confidence: float = 0.35
    min_wall_points: int = 25
    tracked_wall_keep_confidence: float = 0.45
    max_corridor_width_m: float = 3.50
    max_wall_normal_z_abs: float = 0.25
    min_wall_parallel_abs_cos: float = 0.75


@dataclass
class GCAConstraintResult:
    constrained_state: SceneGeometryState
    evaluations: List[ConstraintEvaluation]
    geometry_valid: bool
    required_width_m: float
    facts: Dict[str, Any]


class GCAConstraintLayer:
    """Apply explicit geometry rules to raw scene geometry estimates."""

    def __init__(self, config: GCAConstraintConfig | None = None) -> None:
        self.config = config or GCAConstraintConfig()

    def apply(
        self,
        geometry_state: SceneGeometryState,
        normal_map: np.ndarray | None = None,
    ) -> GCAConstraintResult:
        ground_normal_z = self._estimate_ground_normal_z(normal_map)
        left_wall = self._filter_wall(
            wall=geometry_state.left_wall,
            visible=geometry_state.left_wall_visible,
            confidence=geometry_state.left_wall_confidence,
        )
        right_wall = self._filter_wall(
            wall=geometry_state.right_wall,
            visible=geometry_state.right_wall_visible,
            confidence=geometry_state.right_wall_confidence,
        )

        corridor_width_m = None
        if left_wall is not None and right_wall is not None:
            corridor_width_m = round(left_wall.lateral_distance_m + right_wall.lateral_distance_m, 3)

        left_wall_normal_z_abs = self._wall_normal_z_abs(left_wall)
        right_wall_normal_z_abs = self._wall_normal_z_abs(right_wall)
        wall_parallel_abs_cos = self._wall_parallel_abs_cos(left_wall, right_wall)

        traversable_width_m = geometry_state.traversable_width_m
        if corridor_width_m is not None:
            if traversable_width_m is None:
                traversable_width_m = corridor_width_m
            else:
                traversable_width_m = round(
                    min(max(float(traversable_width_m), 0.0), corridor_width_m),
                    3,
                )

        required_width_m = round(float(geometry_state.robot_width_m + geometry_state.safety_margin_m), 3)
        usable_width_m = traversable_width_m if traversable_width_m is not None else corridor_width_m
        passable = None if usable_width_m is None else usable_width_m >= required_width_m

        constrained_state = replace(
            geometry_state,
            left_wall=left_wall,
            right_wall=right_wall,
            corridor_width_m=corridor_width_m,
            traversable_width_m=traversable_width_m,
            passable=passable,
        )

        facts = {
            "ground_normal_z": self._round_or_none(ground_normal_z),
            "left_wall_confidence": self._round_or_none(geometry_state.left_wall_confidence),
            "right_wall_confidence": self._round_or_none(geometry_state.right_wall_confidence),
            "left_wall_points": left_wall.valid_point_count if left_wall else None,
            "right_wall_points": right_wall.valid_point_count if right_wall else None,
            "left_wall_normal_z_abs": self._round_or_none(left_wall_normal_z_abs),
            "right_wall_normal_z_abs": self._round_or_none(right_wall_normal_z_abs),
            "wall_parallel_abs_cos": self._round_or_none(wall_parallel_abs_cos),
            "left_wall_plane_inliers": left_wall.plane_inlier_count if left_wall else None,
            "right_wall_plane_inliers": right_wall.plane_inlier_count if right_wall else None,
            "left_wall_plane_error_m": self._round_or_none(left_wall.plane_fit_error_m if left_wall else None),
            "right_wall_plane_error_m": self._round_or_none(right_wall.plane_fit_error_m if right_wall else None),
            "corridor_width_m": corridor_width_m,
            "traversable_width_m": traversable_width_m,
            "required_width_m": required_width_m,
            "passable": passable,
        }

        evaluations = [
            self._build_eval(
                name="ground_normal_z",
                actual_value=facts["ground_normal_z"],
                operator=">=",
                expected_value=self.config.min_ground_normal_z,
                source="gca_ground_constraint",
            ),
            self._build_eval(
                name="left_wall_confidence",
                actual_value=facts["left_wall_confidence"],
                operator=">=",
                expected_value=self.config.min_wall_confidence,
                source="gca_wall_tracking_constraint",
            ),
            self._build_eval(
                name="right_wall_confidence",
                actual_value=facts["right_wall_confidence"],
                operator=">=",
                expected_value=self.config.min_wall_confidence,
                source="gca_wall_tracking_constraint",
            ),
            self._build_eval(
                name="left_wall_points",
                actual_value=facts["left_wall_points"],
                operator=">=",
                expected_value=self.config.min_wall_points,
                source="gca_wall_support_constraint",
            ),
            self._build_eval(
                name="right_wall_points",
                actual_value=facts["right_wall_points"],
                operator=">=",
                expected_value=self.config.min_wall_points,
                source="gca_wall_support_constraint",
            ),
            self._build_eval(
                name="left_wall_normal_z_abs",
                actual_value=facts["left_wall_normal_z_abs"],
                operator="<=",
                expected_value=self.config.max_wall_normal_z_abs,
                source="gca_wall_vertical_constraint",
            ),
            self._build_eval(
                name="right_wall_normal_z_abs",
                actual_value=facts["right_wall_normal_z_abs"],
                operator="<=",
                expected_value=self.config.max_wall_normal_z_abs,
                source="gca_wall_vertical_constraint",
            ),
            self._build_eval(
                name="wall_parallel_abs_cos",
                actual_value=facts["wall_parallel_abs_cos"],
                operator=">=",
                expected_value=self.config.min_wall_parallel_abs_cos,
                source="gca_wall_parallel_constraint",
            ),
            self._build_eval(
                name="corridor_width_m",
                actual_value=facts["corridor_width_m"],
                operator="<=",
                expected_value=self.config.max_corridor_width_m,
                source="gca_corridor_sanity_constraint",
            ),
            self._build_eval(
                name="corridor_width_m",
                actual_value=facts["corridor_width_m"],
                operator=">=",
                expected_value=required_width_m,
                source="gca_passability_constraint",
            ),
            self._build_eval(
                name="traversable_width_m",
                actual_value=facts["traversable_width_m"],
                operator=">=",
                expected_value=required_width_m,
                source="gca_passability_constraint",
            ),
        ]

        geometry_valid = all(
            item.passed
            for item in evaluations
            if item.name
            in {
                "ground_normal_z",
                "left_wall_confidence",
                "right_wall_confidence",
                "left_wall_points",
                "right_wall_points",
                "left_wall_normal_z_abs",
                "right_wall_normal_z_abs",
                "wall_parallel_abs_cos",
            }
        )
        return GCAConstraintResult(
            constrained_state=constrained_state,
            evaluations=evaluations,
            geometry_valid=geometry_valid,
            required_width_m=required_width_m,
            facts=facts,
        )

    def _filter_wall(
        self,
        wall: WallEstimate | None,
        visible: bool | None,
        confidence: float | None,
    ) -> WallEstimate | None:
        if wall is None or confidence is None:
            return None

        if visible is False and confidence < self.config.tracked_wall_keep_confidence:
            return None
        if confidence < self.config.min_wall_confidence:
            return None
        if wall.valid_point_count < self.config.min_wall_points:
            return None
        return wall

    def _wall_normal_z_abs(self, wall: WallEstimate | None) -> float | None:
        if wall is None or wall.normal_robot_frame is None:
            return None
        return abs(float(wall.normal_robot_frame[2]))

    def _wall_parallel_abs_cos(
        self,
        left_wall: WallEstimate | None,
        right_wall: WallEstimate | None,
    ) -> float | None:
        if left_wall is None or right_wall is None:
            return None
        if left_wall.normal_robot_frame is None or right_wall.normal_robot_frame is None:
            return None

        left_normal = np.asarray(left_wall.normal_robot_frame, dtype=np.float32)
        right_normal = np.asarray(right_wall.normal_robot_frame, dtype=np.float32)
        left_norm = float(np.linalg.norm(left_normal))
        right_norm = float(np.linalg.norm(right_normal))
        if left_norm < 1e-6 or right_norm < 1e-6:
            return None
        left_normal = left_normal / left_norm
        right_normal = right_normal / right_norm
        return abs(float(np.dot(left_normal, right_normal)))

    def _estimate_ground_normal_z(self, normal_map: np.ndarray | None) -> float | None:
        if normal_map is None:
            return None

        normals = np.asarray(normal_map, dtype=np.float32)
        if normals.ndim == 4:
            normals = normals[0]

        if normals.ndim != 3:
            return None

        if normals.shape[0] == 3:
            normal_z = normals[2]
        elif normals.shape[-1] == 3:
            normal_z = normals[..., 2]
        else:
            return None

        height, width = normal_z.shape
        y_start = int(height * 0.70)
        x_start = int(width * 0.15)
        x_end = int(width * 0.85)
        roi = normal_z[y_start:, x_start:x_end]
        roi = roi[np.isfinite(roi)]
        if roi.size < 32:
            return None
        return float(np.nanmean(roi))

    def _build_eval(
        self,
        name: str,
        actual_value: Any,
        operator: str,
        expected_value: Any,
        source: str,
    ) -> ConstraintEvaluation:
        return ConstraintEvaluation(
            name=name,
            operator=operator,
            expected_value=expected_value,
            actual_value=actual_value,
            passed=self._compare(actual_value, operator, expected_value),
            source=source,
        )

    def _compare(self, actual: Any, operator: str, expected: Any) -> bool:
        if actual is None:
            return False
        if operator == ">=":
            return actual >= expected
        if operator == "<=":
            return actual <= expected
        if operator == "==":
            return actual == expected
        raise ValueError(f"Unsupported operator: {operator}")

    def _round_or_none(self, value: float | None) -> float | None:
        if value is None:
            return None
        return round(float(value), 3)
