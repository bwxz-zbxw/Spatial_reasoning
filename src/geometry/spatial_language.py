from math import hypot
from typing import Iterable, List

from src.perception.observation_protocol import ObservedObject


def euclidean_distance_3d(position: tuple[float, float, float]) -> float:
    return hypot(hypot(position[0], position[1]), position[2])


def lateral_distance(position: tuple[float, float, float]) -> float:
    return abs(position[1])


def forward_distance(position: tuple[float, float, float]) -> float:
    return position[0]


def describe_side(position: tuple[float, float, float]) -> str:
    x, y, _ = position
    forward_label = "front" if x >= 0.25 else "rear" if x <= -0.25 else "side"
    lateral_label = "left" if y >= 0.25 else "right" if y <= -0.25 else "center"

    if forward_label == "side":
        if lateral_label == "center":
            return "center"
        return lateral_label
    if lateral_label == "center":
        return forward_label
    return f"{forward_label}-{lateral_label}"


def filter_by_category(objects: Iterable[ObservedObject], category: str) -> List[ObservedObject]:
    return [obj for obj in objects if obj.category == category]


def nearest_object(objects: Iterable[ObservedObject]) -> ObservedObject | None:
    object_list = list(objects)
    if not object_list:
        return None
    return min(object_list, key=lambda obj: euclidean_distance_3d(obj.position_robot_frame))
