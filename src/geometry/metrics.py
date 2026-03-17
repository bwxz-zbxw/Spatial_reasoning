from math import hypot
from typing import Tuple


Vector2 = Tuple[float, float]


def euclidean_distance(a: Vector2, b: Vector2) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def clearance_between_centers(
    a_position: Vector2,
    a_size: Vector2,
    b_position: Vector2,
    b_size: Vector2,
) -> float:
    center_distance = euclidean_distance(a_position, b_position)
    a_radius = max(a_size) / 2.0
    b_radius = max(b_size) / 2.0
    return center_distance - a_radius - b_radius


def relative_direction(origin: Vector2, target: Vector2) -> Vector2:
    return (target[0] - origin[0], target[1] - origin[1])
