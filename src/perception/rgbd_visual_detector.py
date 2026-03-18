from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

from src.perception.observation_protocol import ObservedObject


@dataclass
class VisualDetectionResult:
    objects: List[ObservedObject]
    source: str
    error: str | None = None


class RGBDVisualDetector:
    """Visual detector for common indoor objects using RGB detection + depth backprojection."""

    def __init__(self, score_threshold: float = 0.55) -> None:
        self.score_threshold = score_threshold
        self._model = None
        self._weights = None
        self._categories = {
            "person": "human",
            "chair": "chair",
            "couch": "sofa",
            "dining table": "table",
            "refrigerator": "fridge",
            "tv": "tv",
            "bed": "bed",
        }

    def detect(
        self,
        image_rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
    ) -> VisualDetectionResult:
        try:
            model = self._get_model()
            import torch

            image_tensor = self._weights.transforms()(Image.fromarray(image_rgb)).to(next(model.parameters()).device)
            with torch.inference_mode():
                prediction = model([image_tensor])[0]
        except Exception as exc:
            return VisualDetectionResult(objects=[], source="detector", error=str(exc))

        labels = prediction["labels"].detach().cpu().tolist()
        scores = prediction["scores"].detach().cpu().tolist()
        boxes = prediction["boxes"].detach().cpu().tolist()
        label_names = self._weights.meta["categories"]

        objects: List[ObservedObject] = []
        for idx, (label_id, score, box) in enumerate(zip(labels, scores, boxes)):
            if score < self.score_threshold:
                continue
            raw_name = label_names[label_id]
            category = self._categories.get(raw_name)
            if category is None:
                continue

            bbox = self._clip_bbox(box, image_rgb.shape[1], image_rgb.shape[0])
            position = self._estimate_position_from_depth(depth_m, intrinsics, bbox)
            if position is None:
                continue
            size = self._estimate_size_from_depth(depth_m, intrinsics, bbox, position[0])

            objects.append(
                ObservedObject(
                    object_id=f"{category}_det_{idx}",
                    category=category,
                    position_robot_frame=position,
                    size=size,
                    bbox_xyxy=bbox,
                    confidence=float(score),
                    attributes={"raw_name": raw_name, "source": "torchvision_detector"},
                )
            )
        return VisualDetectionResult(objects=objects, source="detector", error=None)

    def _get_model(self):
        if self._model is not None and self._weights is not None:
            return self._model

        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        self._weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self._model = fasterrcnn_resnet50_fpn(weights=self._weights)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()
        return self._model

    def _clip_bbox(self, box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        x1 = float(max(0, min(width - 1, x1)))
        y1 = float(max(0, min(height - 1, y1)))
        x2 = float(max(x1 + 1, min(width, x2)))
        y2 = float(max(y1 + 1, min(height, y2)))
        return (x1, y1, x2, y2)

    def _estimate_position_from_depth(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float] | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = depth_m[y1:y2, x1:x2]
        valid = np.isfinite(roi)
        if int(valid.sum()) < 20:
            return None

        z = float(np.nanmedian(roi[valid]))
        center_u = (x1 + x2) / 2.0
        center_v = (y1 + y2) / 2.0
        x_camera = ((center_u - intrinsics[0, 2]) / intrinsics[0, 0]) * z
        y_camera = ((center_v - intrinsics[1, 2]) / intrinsics[1, 1]) * z
        return (z, -float(x_camera), -float(y_camera))

    def _estimate_size_from_depth(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        bbox: tuple[float, float, float, float],
        forward_distance: float,
    ) -> tuple[float, float, float]:
        x1, y1, x2, y2 = bbox
        width_px = max(1.0, x2 - x1)
        height_px = max(1.0, y2 - y1)
        width_m = (width_px / intrinsics[0, 0]) * forward_distance
        height_m = (height_px / intrinsics[1, 1]) * forward_distance
        return (round(forward_distance * 0.3, 3), round(width_m, 3), round(height_m, 3))
