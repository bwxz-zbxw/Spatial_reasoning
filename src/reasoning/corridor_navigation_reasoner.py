from __future__ import annotations

from dataclasses import dataclass

from src.geometry.rgbd_geometry_branch import CorridorGap, SceneGeometryState
from src.reasoning.gca_constraint_layer import GCAConstraintResult
from src.reasoning.protocol import ReasoningDecision


@dataclass
class CorridorReasonerConfig:
    obstacle_center_band_m: float = 0.18
    gap_center_band_m: float = 0.12
    stop_distance_m: float = 0.75
    slow_distance_m: float = 1.25
    action_confirm_frames: int = 3


class CorridorNavigationReasoner:
    """Convert constrained corridor geometry into robot navigation actions."""

    def __init__(self, config: CorridorReasonerConfig | None = None) -> None:
        self.config = config or CorridorReasonerConfig()
        self._stable_action = "go_straight"
        self._candidate_action = "go_straight"
        self._candidate_count = 0

    def reset(self) -> None:
        self._stable_action = "go_straight"
        self._candidate_action = "go_straight"
        self._candidate_count = 0

    def decide(
        self,
        geometry_state: SceneGeometryState,
        gca_result: GCAConstraintResult,
    ) -> ReasoningDecision:
        state = gca_result.constrained_state
        obstacle_side = self._infer_obstacle_side(state)
        left_free_width_m, right_free_width_m = self._estimate_side_free_widths(state)
        free_side = self._infer_free_side(
            state=state,
            obstacle_side=obstacle_side,
            left_free_width_m=left_free_width_m,
            right_free_width_m=right_free_width_m,
            required_width_m=gca_result.required_width_m,
        )
        nearest_distance = state.nearest_obstacle_distance_m

        raw_action, confidence, rationale = self._resolve_action(
            state=state,
            gca_result=gca_result,
            obstacle_side=obstacle_side,
            free_side=free_side,
            nearest_distance=nearest_distance,
        )
        action = self._stabilize_action(raw_action)

        facts = {
            "geometry_valid": gca_result.geometry_valid,
            "passable": state.passable,
            "required_width_m": gca_result.required_width_m,
            "corridor_width_m": state.corridor_width_m,
            "traversable_width_m": state.traversable_width_m,
            "nearest_obstacle_distance_m": nearest_distance,
            "nearest_obstacle_lateral_m": (
                state.nearest_obstacle.lateral_offset_m if state.nearest_obstacle is not None else None
            ),
            "nearest_obstacle_width_m": state.nearest_obstacle.width_m if state.nearest_obstacle is not None else None,
            "left_free_width_m": left_free_width_m,
            "right_free_width_m": right_free_width_m,
            "obstacle_side": obstacle_side,
            "free_side": free_side,
            "raw_action": raw_action,
            "stable_action": action,
            "gca_ground_normal_z": gca_result.facts.get("ground_normal_z"),
            "gca_wall_parallel_abs_cos": gca_result.facts.get("wall_parallel_abs_cos"),
        }
        return ReasoningDecision(
            action=action,
            confidence=confidence,
            rationale=rationale,
            facts=facts,
        )

    def _resolve_action(
        self,
        state: SceneGeometryState,
        gca_result: GCAConstraintResult,
        obstacle_side: str,
        free_side: str,
        nearest_distance: float | None,
    ) -> tuple[str, float, list[str]]:
        if not gca_result.geometry_valid:
            return "slow_down", 0.72, ["gca_geometry_invalid"]

        if state.passable is False:
            return "stop", 0.94, ["passability_constraint_failed"]

        if state.nearest_obstacle is None:
            return "go_straight", 0.93, ["no_obstacle_detected"]

        if nearest_distance is not None and nearest_distance <= self.config.stop_distance_m and free_side == "none":
            return "stop", 0.96, ["obstacle_close_and_no_free_side"]

        if obstacle_side == "left" and free_side == "right":
            action = "avoid_right"
            confidence = 0.9
            rationale = ["obstacle_on_left", "right_gap_available"]
        elif obstacle_side == "right" and free_side == "left":
            action = "avoid_left"
            confidence = 0.9
            rationale = ["obstacle_on_right", "left_gap_available"]
        elif obstacle_side == "center" and free_side in {"left", "right"}:
            action = f"avoid_{free_side}"
            confidence = 0.86
            rationale = ["obstacle_near_centerline", f"{free_side}_gap_is_widest"]
        elif free_side == "center":
            action = "go_straight"
            confidence = 0.8
            rationale = ["center_gap_available"]
        elif free_side in {"left", "right"}:
            action = f"avoid_{free_side}"
            confidence = 0.78
            rationale = ["free_side_available_but_obstacle_side_ambiguous"]
        else:
            action = "slow_down"
            confidence = 0.74
            rationale = ["free_side_undetermined"]

        if nearest_distance is not None and nearest_distance <= self.config.slow_distance_m and action != "stop":
            return "slow_down", min(0.92, confidence + 0.04), rationale + ["obstacle_within_slow_distance"]
        return action, confidence, rationale

    def _infer_obstacle_side(self, state: SceneGeometryState) -> str:
        obstacle = state.nearest_obstacle
        if obstacle is None:
            return "none"

        lateral = float(obstacle.lateral_offset_m)
        if lateral > self.config.obstacle_center_band_m:
            return "left"
        if lateral < -self.config.obstacle_center_band_m:
            return "right"
        return "center"

    def _infer_free_side(
        self,
        state: SceneGeometryState,
        obstacle_side: str,
        left_free_width_m: float | None,
        right_free_width_m: float | None,
        required_width_m: float,
    ) -> str:
        left_ok = left_free_width_m is not None and left_free_width_m >= required_width_m
        right_ok = right_free_width_m is not None and right_free_width_m >= required_width_m

        if state.nearest_obstacle is not None:
            if obstacle_side == "left" and right_ok:
                return "right"
            if obstacle_side == "right" and left_ok:
                return "left"
            if obstacle_side == "center":
                if left_ok and right_ok:
                    if left_free_width_m > right_free_width_m + 0.08:
                        return "left"
                    if right_free_width_m > left_free_width_m + 0.08:
                        return "right"
                    return "center"
                if left_ok:
                    return "left"
                if right_ok:
                    return "right"

        if left_ok and right_ok:
            if left_free_width_m > right_free_width_m + 0.08:
                return "left"
            if right_free_width_m > left_free_width_m + 0.08:
                return "right"
            return "center"
        if left_ok:
            return "left"
        if right_ok:
            return "right"

        if not state.gaps:
            return "none"

        widest_gap = max(state.gaps, key=lambda item: item.width_m)
        if widest_gap.width_m < required_width_m:
            return "none"

        center = self._gap_center(widest_gap)
        if center > self.config.gap_center_band_m:
            return "left"
        if center < -self.config.gap_center_band_m:
            return "right"
        return "center"

    def _estimate_side_free_widths(self, state: SceneGeometryState) -> tuple[float | None, float | None]:
        if state.left_wall is None or state.right_wall is None:
            return None, None

        left_boundary = float(state.left_wall.lateral_distance_m)
        right_boundary = -float(state.right_wall.lateral_distance_m)
        obstacle = state.nearest_obstacle
        if obstacle is None:
            center_width = round(left_boundary, 3)
            right_width = round(abs(right_boundary), 3)
            return center_width, right_width

        half_width = max(0.05, float(obstacle.width_m) / 2.0)
        obstacle_start = max(right_boundary, float(obstacle.lateral_offset_m) - half_width)
        obstacle_end = min(left_boundary, float(obstacle.lateral_offset_m) + half_width)
        right_free_width = round(max(0.0, obstacle_start - right_boundary), 3)
        left_free_width = round(max(0.0, left_boundary - obstacle_end), 3)
        return left_free_width, right_free_width

    def _gap_center(self, gap: CorridorGap) -> float:
        return 0.5 * (float(gap.start_m) + float(gap.end_m))

    def _stabilize_action(self, raw_action: str) -> str:
        if raw_action == "stop":
            self._stable_action = "stop"
            self._candidate_action = "stop"
            self._candidate_count = self.config.action_confirm_frames
            return "stop"

        if raw_action == self._stable_action:
            self._candidate_action = raw_action
            self._candidate_count = 0
            return self._stable_action

        if raw_action != self._candidate_action:
            self._candidate_action = raw_action
            self._candidate_count = 1
            return self._stable_action

        self._candidate_count += 1
        if self._candidate_count >= self.config.action_confirm_frames:
            self._stable_action = raw_action
            self._candidate_count = 0
        return self._stable_action
