from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.geometry.spatial_language import describe_side, euclidean_distance_3d
from src.perception.observation_protocol import ObservedObject
from src.perception.sunrgbd_loader import SUNRGBDSample
from src.reasoning.query_protocol import SpatialQuery


@dataclass
class QueryAnswer:
    answer: str
    evidence: List[Dict[str, object]]


class SUNRGBDGeometrySolver:
    """RGB-D geometry solver with wall extraction and detector-first object queries."""

    def solve(
        self,
        sample: SUNRGBDSample,
        query: SpatialQuery,
        depth_m: np.ndarray,
        detected_objects: List[ObservedObject] | None = None,
    ) -> QueryAnswer:
        if query.target == "wall":
            return self._solve_wall(sample, query, depth_m)
        return self._solve_object(sample, query, detected_objects or [])

    def _solve_object(
        self,
        sample: SUNRGBDSample,
        query: SpatialQuery,
        detected_objects: List[ObservedObject],
    ) -> QueryAnswer:
        detector_candidates = self._filter_object_candidates(
            [obj for obj in detected_objects if obj.category == query.target],
            query.side_hint,
        )
        annotation_candidates = self._filter_object_candidates(
            [
                obj
                for obj in sample.observation.objects
                if obj.category == query.target
                or obj.attributes.get("raw_name", "").startswith(query.target)
            ],
            query.side_hint,
        )

        candidates = detector_candidates or annotation_candidates
        if not candidates:
            target_name = self._category_name(query.target)
            if query.side_hint in {"left", "right"}:
                side_name = "左侧" if query.side_hint == "left" else "右侧"
                return QueryAnswer(answer=f"当前样本中没有检测到{side_name}的{target_name}。", evidence=[])
            return QueryAnswer(answer=f"当前样本中没有检测到{target_name}。", evidence=[])

        candidates = sorted(candidates, key=lambda obj: euclidean_distance_3d(obj.position_robot_frame))
        selected = candidates[0]

        evidence = [self._object_evidence(selected)]
        evidence[0]["vision_source"] = "detector" if detector_candidates else "annotation_fallback"

        category_name = self._category_name(selected.category)
        answer = (
            f"{category_name}在你的{self._side_name(str(evidence[0]['side']))}，"
            f"直线距离约 {evidence[0]['distance_m']} 米，"
            f"前向距离约 {evidence[0]['forward_distance_m']} 米。"
        )
        return QueryAnswer(answer=answer, evidence=evidence)

    def _solve_wall(self, sample: SUNRGBDSample, query: SpatialQuery, depth_m: np.ndarray) -> QueryAnswer:
        wall_estimates = self._extract_corridor_walls(depth_m, sample.intrinsics)
        evidence = [{"category": "wall", "vision_source": "rgbd_geometry", **item} for item in wall_estimates]

        if not evidence:
            return QueryAnswer(answer="没有可靠检测到墙体。", evidence=[])

        left_candidates = [item for item in evidence if item["side"] == "left"]
        right_candidates = [item for item in evidence if item["side"] == "right"]

        if query.side_hint == "left":
            if not left_candidates:
                return QueryAnswer(answer="没有可靠检测到左墙。", evidence=evidence)
            wall = min(left_candidates, key=lambda item: item["lateral_distance_m"])
            return QueryAnswer(
                answer=(
                    f"左墙在你的左侧，横向距离约 {wall['lateral_distance_m']} 米，"
                    f"前向距离约 {wall['forward_distance_m']} 米。"
                ),
                evidence=evidence,
            )

        if query.side_hint == "right":
            if not right_candidates:
                return QueryAnswer(answer="没有可靠检测到右墙。", evidence=evidence)
            wall = min(right_candidates, key=lambda item: item["lateral_distance_m"])
            return QueryAnswer(
                answer=(
                    f"右墙在你的右侧，横向距离约 {wall['lateral_distance_m']} 米，"
                    f"前向距离约 {wall['forward_distance_m']} 米。"
                ),
                evidence=evidence,
            )

        parts: List[str] = []
        if left_candidates:
            left_wall = min(left_candidates, key=lambda item: item["lateral_distance_m"])
            parts.append(f"左墙横向距离约 {left_wall['lateral_distance_m']} 米")
        if right_candidates:
            right_wall = min(right_candidates, key=lambda item: item["lateral_distance_m"])
            parts.append(f"右墙横向距离约 {right_wall['lateral_distance_m']} 米")
        if left_candidates and right_candidates:
            corridor_width = round(
                min(left_candidates, key=lambda item: item["lateral_distance_m"])["lateral_distance_m"]
                + min(right_candidates, key=lambda item: item["lateral_distance_m"])["lateral_distance_m"],
                3,
            )
            parts.append(f"估计走廊宽度约 {corridor_width} 米")
        return QueryAnswer(answer="，".join(parts) + "。", evidence=evidence)

    def _extract_corridor_walls(self, depth_m: np.ndarray, intrinsics: np.ndarray) -> List[Dict[str, object]]:
        estimates: List[Dict[str, object]] = []
        for side in ("left", "right"):
            estimate = self._estimate_side_wall(depth_m, intrinsics, side)
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    def _estimate_side_wall(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
        side: str,
    ) -> Dict[str, object] | None:
        height, width = depth_m.shape
        y_start = int(height * 0.28)
        y_end = int(height * 0.78)
        x_start, x_end = (0, int(width * 0.22)) if side == "left" else (int(width * 0.78), width)

        roi = depth_m[y_start:y_end, x_start:x_end]
        valid = np.isfinite(roi)
        if int(valid.sum()) < 40:
            return None

        _, u_coords = np.where(valid)
        z = roi[valid]
        u = u_coords.astype(np.float32) + x_start
        x_camera = ((u - intrinsics[0, 2]) / intrinsics[0, 0]) * z
        side_mask = x_camera < 0 if side == "left" else x_camera > 0
        if int(side_mask.sum()) < 25:
            return None

        lateral = float(np.nanmedian(np.abs(x_camera[side_mask])))
        forward = float(np.nanmedian(z[side_mask]))
        return {
            "side": side,
            "lateral_distance_m": round(lateral, 3),
            "forward_distance_m": round(forward, 3),
            "distance_m": round(float(np.sqrt(lateral**2 + forward**2)), 3),
            "valid_point_count": int(side_mask.sum()),
        }

    def _filter_object_candidates(
        self,
        objects: List[ObservedObject],
        side_hint: str | None,
    ) -> List[ObservedObject]:
        if side_hint not in {"left", "right"}:
            return list(objects)
        filtered = [obj for obj in objects if self._matches_side_hint(obj, side_hint)]
        return filtered

    def _matches_side_hint(self, obj: ObservedObject, side_hint: str) -> bool:
        _, lateral, _ = obj.position_robot_frame
        if side_hint == "left":
            return lateral > 0.0
        if side_hint == "right":
            return lateral < 0.0
        return True

    def _object_evidence(self, obj: ObservedObject) -> Dict[str, object]:
        forward, lateral, vertical = obj.position_robot_frame
        return {
            "object_id": obj.object_id,
            "raw_name": obj.attributes.get("raw_name", obj.category),
            "category": obj.category,
            "side": describe_side((forward, lateral, vertical)),
            "distance_m": round(euclidean_distance_3d(obj.position_robot_frame), 3),
            "forward_distance_m": round(float(forward), 3),
            "lateral_distance_m": round(abs(float(lateral)), 3),
            "bbox_xyxy": obj.bbox_xyxy,
            "confidence": obj.confidence,
        }

    def _category_name(self, category: str) -> str:
        mapping = {
            "table": "桌子",
            "wall": "墙",
            "door": "门",
            "chair": "椅子",
            "sofa": "沙发",
            "cabinet": "柜子",
            "human": "行人",
            "garbage_bin": "垃圾桶",
            "fridge": "冰箱",
            "tv": "电视",
            "bed": "床",
            "unknown": "目标物体",
        }
        return mapping.get(category, category)

    def _side_name(self, side: str) -> str:
        mapping = {
            "left": "左侧",
            "right": "右侧",
            "front": "前方",
            "rear": "后方",
            "front-left": "左前方",
            "front-right": "右前方",
            "rear-left": "左后方",
            "rear-right": "右后方",
            "center": "正中",
        }
        return mapping.get(side, side)
