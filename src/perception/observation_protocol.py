from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Vector3 = Tuple[float, float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class ObservedObject:
    object_id: str
    category: str
    position_robot_frame: Vector3
    size: Vector3
    bbox_xyxy: BBox
    confidence: float = 1.0
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class ImageObservation:
    image_id: str
    image_path: str
    reference_frame: str
    robot_pose_hint: Dict[str, float]
    objects: List[ObservedObject] = field(default_factory=list)
