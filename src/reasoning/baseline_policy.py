from dataclasses import asdict
from math import hypot
from typing import Dict, Iterable, Tuple

from src.geometry.metrics import clearance_between_centers, euclidean_distance
from src.geometry.scene_graph import SceneObject
from src.geometry.scenario_loader import Scenario
from src.reasoning.protocol import ReasoningDecision


Vector2 = Tuple[float, float]


class BaselineYieldPolicy:
    """Simple rule baseline for corridor interaction decisions."""

    def decide(self, scenario: Scenario) -> ReasoningDecision:
        robot = scenario.scene_graph.get_object(scenario.scene_graph.robot_id)
        min_clearance = scenario.thresholds["min_clearance_m"]
        min_ttc = scenario.thresholds["min_ttc_s"]

        facts: Dict[str, float | str] = {
            "scenario_id": scenario.scenario_id,
            "corridor_width_m": scenario.corridor_width_m,
        }

        nearest = self._nearest_relevant_object(robot, scenario.scene_graph.objects.values())
        if nearest is None:
            return ReasoningDecision(
                action="proceed",
                confidence=0.95,
                rationale=["no_dynamic_or_blocking_object_detected"],
                facts=facts,
            )

        clearance = clearance_between_centers(
            robot.position,
            robot.size,
            nearest.position,
            nearest.size,
        )
        ttc = self._estimate_ttc(robot.position, robot.velocity, nearest.position, nearest.velocity)
        lateral_offset = abs(nearest.position[1] - robot.position[1])
        free_width = scenario.corridor_width_m - nearest.size[1]

        facts.update(
            {
                "nearest_object_id": nearest.object_id,
                "nearest_category": nearest.category,
                "distance_m": round(euclidean_distance(robot.position, nearest.position), 3),
                "clearance_m": round(clearance, 3),
                "ttc_s": round(ttc, 3) if ttc != float("inf") else "inf",
                "lateral_offset_m": round(lateral_offset, 3),
                "estimated_free_width_m": round(free_width, 3),
            }
        )

        if clearance < min_clearance:
            if nearest.category == "human":
                return ReasoningDecision(
                    action="stop_and_wait",
                    confidence=0.93,
                    rationale=["human_clearance_below_threshold"],
                    facts=facts,
                )
            return ReasoningDecision(
                action="local_replan",
                confidence=0.9,
                rationale=["static_blockage_clearance_below_threshold"],
                facts=facts,
            )

        if nearest.category != "human" and free_width < robot.size[1] + min_clearance:
            return ReasoningDecision(
                action="local_replan",
                confidence=0.88,
                rationale=["free_width_below_robot_width_plus_margin"],
                facts=facts,
            )

        if ttc < min_ttc:
            return ReasoningDecision(
                action="slow_down",
                confidence=0.84,
                rationale=["time_to_collision_below_threshold"],
                facts=facts,
            )

        if nearest.category == "human" and lateral_offset < 0.25:
            return ReasoningDecision(
                action="yield_right",
                confidence=0.8,
                rationale=["human_aligned_with_robot_path"],
                facts=facts,
            )

        return ReasoningDecision(
            action="proceed",
            confidence=0.78,
            rationale=["clearance_and_ttc_within_limits"],
            facts=facts,
        )

    def _nearest_relevant_object(
        self,
        robot: SceneObject,
        objects: Iterable[SceneObject],
    ) -> SceneObject | None:
        candidates = [obj for obj in objects if obj.object_id != robot.object_id]
        if not candidates:
            return None
        return min(candidates, key=lambda obj: euclidean_distance(robot.position, obj.position))

    def _estimate_ttc(
        self,
        robot_position: Vector2,
        robot_velocity: Vector2,
        target_position: Vector2,
        target_velocity: Vector2,
    ) -> float:
        relative_position = (
            target_position[0] - robot_position[0],
            target_position[1] - robot_position[1],
        )
        relative_velocity = (
            robot_velocity[0] - target_velocity[0],
            robot_velocity[1] - target_velocity[1],
        )
        closing_speed = hypot(relative_velocity[0], relative_velocity[1])
        if closing_speed <= 1e-6:
            return float("inf")

        relative_distance = hypot(relative_position[0], relative_position[1])
        return relative_distance / closing_speed
