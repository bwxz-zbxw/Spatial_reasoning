import json
from pathlib import Path

from src.perception.observation_protocol import ImageObservation, ObservedObject


def load_image_observation(path: Path) -> ImageObservation:
    data = json.loads(path.read_text(encoding="utf-8"))
    objects = [
        ObservedObject(
            object_id=item["object_id"],
            category=item["category"],
            position_robot_frame=tuple(item["position_robot_frame"]),
            size=tuple(item["size"]),
            bbox_xyxy=tuple(item["bbox_xyxy"]),
            confidence=float(item.get("confidence", 1.0)),
            attributes=item.get("attributes", {}),
        )
        for item in data.get("objects", [])
    ]

    return ImageObservation(
        image_id=data["image_id"],
        image_path=data["image_path"],
        reference_frame=data.get("reference_frame", "robot_base"),
        robot_pose_hint=data.get("robot_pose_hint", {}),
        objects=objects,
    )
