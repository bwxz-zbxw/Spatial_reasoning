import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.perception.observation_protocol import ImageObservation, ObservedObject


@dataclass
class SUNRGBDSample:
    sample_dir: Path
    image_path: Path
    depth_path: Path
    intrinsics: np.ndarray
    observation: ImageObservation


class SUNRGBDLoader:
    """Loads one SUNRGBD sample using existing 2D/3D annotations as perception output."""

    def load_sample(self, sample_dir: Path) -> SUNRGBDSample:
        sample_dir = sample_dir.resolve()
        sample_id = sample_dir.name
        image_path = next((sample_dir / "image").glob("*.jpg"))
        depth_path = next((sample_dir / "depth").glob("*.png"))
        intrinsics = self._load_intrinsics(sample_dir / "intrinsics.txt")
        observation = ImageObservation(
            image_id=sample_id,
            image_path=str(image_path),
            reference_frame="robot_base",
            robot_pose_hint={},
            objects=self._load_objects(sample_dir),
        )
        return SUNRGBDSample(
            sample_dir=sample_dir,
            image_path=image_path,
            depth_path=depth_path,
            intrinsics=intrinsics,
            observation=observation,
        )

    def load_depth_meters(self, sample: SUNRGBDSample) -> np.ndarray:
        depth_raw = np.array(Image.open(sample.depth_path), dtype=np.float32)
        depth_m = depth_raw / 1000.0
        invalid = (depth_m <= 0.05) | (depth_m >= 10.0)
        depth_m[invalid] = np.nan
        return depth_m

    def _load_intrinsics(self, path: Path) -> np.ndarray:
        values = [float(item) for item in path.read_text(encoding="utf-8").split()]
        return np.array(values, dtype=np.float32).reshape(3, 3)

    def _load_objects(self, sample_dir: Path) -> list[ObservedObject]:
        anno_path = sample_dir / "annotation2D3D" / "index.json"
        data = json.loads(anno_path.read_text(encoding="utf-8"))
        frame_polygons = data["frames"][0]["polygon"]
        objects_meta = data["objects"]
        observed: list[ObservedObject] = []

        for index, frame_item in enumerate(frame_polygons):
            object_index = frame_item.get("object")
            if object_index is None or object_index >= len(objects_meta):
                continue
            meta = objects_meta[object_index]
            if meta is None or "name" not in meta:
                continue

            category = self._normalize_category(meta["name"])
            bbox = self._polygon_to_bbox(frame_item.get("x", []), frame_item.get("y", []))
            polygon_3d = meta.get("polygon", [])
            position = self._estimate_center_from_polygon(polygon_3d)
            size = self._estimate_size_from_polygon(polygon_3d)
            observed.append(
                ObservedObject(
                    object_id=f"{category}_{index}",
                    category=category,
                    position_robot_frame=position,
                    size=size,
                    bbox_xyxy=bbox,
                    confidence=1.0,
                    attributes={"raw_name": meta["name"]},
                )
            )
        observed.extend(self._load_structural_objects(sample_dir, start_index=len(observed)))
        return observed

    def _load_structural_objects(self, sample_dir: Path, start_index: int) -> list[ObservedObject]:
        anno_path = sample_dir / "annotation3Dfinal" / "index.json"
        data = json.loads(anno_path.read_text(encoding="utf-8"))
        observed: list[ObservedObject] = []
        next_index = start_index
        for meta in data.get("objects", []):
            raw_name = meta.get("name")
            if not raw_name:
                continue
            category = self._normalize_category(raw_name)
            if category not in {"wall", "door", "floor", "ceiling"}:
                continue
            polygon_3d = meta.get("polygon", [])
            if not polygon_3d:
                continue
            observed.append(
                ObservedObject(
                    object_id=f"{category}_{next_index}",
                    category=category,
                    position_robot_frame=self._estimate_center_from_polygon(polygon_3d),
                    size=self._estimate_size_from_polygon(polygon_3d),
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    confidence=1.0,
                    attributes={"raw_name": raw_name, "source": "annotation3Dfinal"},
                )
            )
            next_index += 1
        return observed

    def _normalize_category(self, name: str) -> str:
        base = name.split(":")[0].strip().lower().replace(" ", "_")
        mapping = {
            "side_table": "table",
            "coffee_table": "table",
            "dining_table": "table",
            "desk": "table",
            "night_stand": "cabinet",
            "garbagebin": "garbage_bin",
            "bookshelf": "cabinet",
        }
        return mapping.get(base, base)

    def _polygon_to_bbox(self, x_values: list[float], y_values: list[float]) -> tuple[float, float, float, float]:
        if not x_values or not y_values:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(min(x_values)),
            float(min(y_values)),
            float(max(x_values)),
            float(max(y_values)),
        )

    def _estimate_center_from_polygon(self, polygons: list[dict]) -> tuple[float, float, float]:
        if not polygons:
            return (0.0, 0.0, 0.0)
        polygon = polygons[0]
        x_values = polygon.get("X", [0.0])
        z_values = polygon.get("Z", [0.0])
        y_min = float(polygon.get("Ymin", 0.0))
        y_max = float(polygon.get("Ymax", 0.0))
        x_center = float(sum(x_values) / len(x_values)) if x_values else 0.0
        z_center = float(sum(z_values) / len(z_values)) if z_values else 0.0
        y_center = (y_min + y_max) / 2.0
        # Convert SUNRGBD convention to robot-centric convention: forward=z, left=-x.
        return (z_center, -x_center, y_center)

    def _estimate_size_from_polygon(self, polygons: list[dict]) -> tuple[float, float, float]:
        if not polygons:
            return (0.0, 0.0, 0.0)
        polygon = polygons[0]
        x_values = polygon.get("X", [0.0])
        z_values = polygon.get("Z", [0.0])
        y_min = float(polygon.get("Ymin", 0.0))
        y_max = float(polygon.get("Ymax", 0.0))
        width = float(max(x_values) - min(x_values)) if x_values else 0.0
        depth = float(max(z_values) - min(z_values)) if z_values else 0.0
        height = float(abs(y_max - y_min))
        return (depth, width, height)
