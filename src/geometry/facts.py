from math import hypot
from typing import Dict, Iterable, Tuple

from src.geometry.metrics import clearance_between_centers, euclidean_distance
from src.geometry.scene_graph import SceneObject
from src.geometry.scenario_loader import Scenario


Vector2 = Tuple[float, float]


def nearest_relevant_object(robot: SceneObject, objects: Iterable[SceneObject]) -> SceneObject | None:
    candidates = [obj for obj in objects if obj.object_id != robot.object_id]
    if not candidates:
        return None
    return min(candidates, key=lambda obj: euclidean_distance(robot.position, obj.position))


def estimate_ttc(
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


def build_scene_facts(scenario: Scenario) -> Dict[str, float | str]:
    robot = scenario.scene_graph.get_object(scenario.scene_graph.robot_id)
    facts: Dict[str, float | str] = {
        "scenario_id": scenario.scenario_id,
        "corridor_width_m": scenario.corridor_width_m,
        "robot_width_m": robot.size[1],
        "goal_x_m": scenario.goal_position[0],
        "goal_y_m": scenario.goal_position[1],
    }

    nearest = nearest_relevant_object(robot, scenario.scene_graph.objects.values())
    if nearest is None:
        facts["nearest_object_id"] = "none"
        facts["nearest_category"] = "none"
        return facts

    clearance = clearance_between_centers(
        robot.position,
        robot.size,
        nearest.position,
        nearest.size,
    )
    ttc = estimate_ttc(robot.position, robot.velocity, nearest.position, nearest.velocity)
    free_width = scenario.corridor_width_m - nearest.size[1]
    lateral_offset = abs(nearest.position[1] - robot.position[1])
    distance = euclidean_distance(robot.position, nearest.position)

    facts.update(
        {
            "nearest_object_id": nearest.object_id,
            "nearest_category": nearest.category,
            "distance_m": round(distance, 3),
            "clearance_m": round(clearance, 3),
            "ttc_s": round(ttc, 3) if ttc != float("inf") else "inf",
            "estimated_free_width_m": round(free_width, 3),
            "lateral_offset_m": round(lateral_offset, 3),
        }
    )
    return facts
