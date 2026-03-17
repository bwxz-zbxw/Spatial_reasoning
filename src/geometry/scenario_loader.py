from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.geometry.scene_graph import SceneGraph, SceneObject


Vector2 = Tuple[float, float]


@dataclass
class Scenario:
    scenario_id: str
    description: str
    expected_action: str
    corridor_width_m: float
    corridor_length_m: float
    goal_position: Vector2
    thresholds: Dict[str, float]
    scene_graph: SceneGraph


def _to_vector2(values: List[float]) -> Vector2:
    return (float(values[0]), float(values[1]))


def load_scenarios(config_path: Path) -> Dict[str, Scenario]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    scenarios: Dict[str, Scenario] = {}
    for item in data.get("scenarios", []):
        robot = item["robot"]
        graph = SceneGraph(frame_id=item["id"], robot_id=robot["object_id"])
        graph.add_object(
            SceneObject(
                object_id=robot["object_id"],
                category=robot["category"],
                position=_to_vector2(robot["position"]),
                size=_to_vector2(robot["size"]),
                yaw=float(robot.get("yaw", 0.0)),
                velocity=_to_vector2(robot.get("velocity", [0.0, 0.0])),
            )
        )

        for agent in item.get("agents", []):
            graph.add_object(
                SceneObject(
                    object_id=agent["object_id"],
                    category=agent["category"],
                    position=_to_vector2(agent["position"]),
                    size=_to_vector2(agent["size"]),
                    yaw=float(agent.get("yaw", 0.0)),
                    velocity=_to_vector2(agent.get("velocity", [0.0, 0.0])),
                )
            )

        scenarios[item["id"]] = Scenario(
            scenario_id=item["id"],
            description=item["description"],
            expected_action=item["expected_action"],
            corridor_width_m=float(item["environment"]["corridor_width_m"]),
            corridor_length_m=float(item["environment"]["corridor_length_m"]),
            goal_position=_to_vector2(item["goal"]["position"]),
            thresholds={
                key: float(value) for key, value in item.get("thresholds", {}).items()
            },
            scene_graph=graph,
        )

    return scenarios
