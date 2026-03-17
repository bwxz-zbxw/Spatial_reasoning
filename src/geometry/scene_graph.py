from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


Vector2 = Tuple[float, float]


@dataclass
class SceneObject:
    object_id: str
    category: str
    position: Vector2
    size: Vector2
    yaw: float = 0.0
    velocity: Vector2 = (0.0, 0.0)
    confidence: float = 1.0


@dataclass
class SceneRelation:
    source_id: str
    relation_type: str
    target_id: str
    value: Optional[float] = None


@dataclass
class SceneGraph:
    frame_id: str
    robot_id: str
    objects: Dict[str, SceneObject] = field(default_factory=dict)
    relations: List[SceneRelation] = field(default_factory=list)

    def add_object(self, obj: SceneObject) -> None:
        self.objects[obj.object_id] = obj

    def add_relation(self, relation: SceneRelation) -> None:
        self.relations.append(relation)

    def get_object(self, object_id: str) -> SceneObject:
        return self.objects[object_id]
