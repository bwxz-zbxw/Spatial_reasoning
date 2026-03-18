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
    """Combines SUNRGBD annotations and depth-derived wall estimates for spatial answers."""

    def solve(self, sample: SUNRGBDSample, query: SpatialQuery, depth_m: np.ndarray) -> QueryAnswer:
        if query.target == "wall":
            return self._solve_wall(sample, query, depth_m)
        return self._solve_object(sample, query)

    def _solve_object(self, sample: SUNRGBDSample, query: SpatialQuery) -> QueryAnswer:
        candidates = [
            obj for obj in sample.observation.objects
            if obj.category == query.target or obj.attributes.get("raw_name", "").startswith(query.target)
        ]
        if not candidates:
            return QueryAnswer(
                answer=f"当前样本中没有检测到 {query.target}。",
                evidence=[],
            )

        if query.side_hint == "nearest" or len(candidates) > 1:
            candidates = sorted(candidates, key=lambda obj: euclidean_distance_3d(obj.position_robot_frame))
            selected = candidates[0]
        else:
            selected = candidates[0]

        evidence = [self._object_evidence(selected)]
        category_name = self._category_name(selected.category)
        answer = (
            f"{category_name}在你的{self._side_name(evidence[0]['side'])}，"
            f"直线距离约 {evidence[0]['distance_m']} 米，"
            f"前向距离约 {evidence[0]['forward_distance_m']} 米。"
        )
        return QueryAnswer(answer=answer, evidence=evidence)

    def _solve_wall(self, sample: SUNRGBDSample, query: SpatialQuery, depth_m: np.ndarray) -> QueryAnswer:
        annotated_walls = [obj for obj in sample.observation.objects if obj.category == "wall"]
        if annotated_walls:
            return self._solve_wall_from_annotations(annotated_walls, query)

        left = self._estimate_side_wall(depth_m, sample.intrinsics, side="left")
        right = self._estimate_side_wall(depth_m, sample.intrinsics, side="right")

        evidence: List[Dict[str, object]] = []
        if left:
            evidence.append({"category": "wall", **left})
        if right:
            evidence.append({"category": "wall", **right})

        if not evidence:
            return QueryAnswer(answer="没有可靠检测到墙体。", evidence=[])

        if query.side_hint == "left":
            wall = next((item for item in evidence if item["side"] == "left"), None)
            if wall is None:
                return QueryAnswer(answer="没有可靠检测到左墙。", evidence=evidence)
            return QueryAnswer(
                answer=f"左墙在你的左侧，横向距离约 {wall['lateral_distance_m']} 米，前向距离约 {wall['forward_distance_m']} 米。",
                evidence=evidence,
            )

        if query.side_hint == "right":
            wall = next((item for item in evidence if item["side"] == "right"), None)
            if wall is None:
                return QueryAnswer(answer="没有可靠检测到右墙。", evidence=evidence)
            return QueryAnswer(
                answer=f"右墙在你的右侧，横向距离约 {wall['lateral_distance_m']} 米，前向距离约 {wall['forward_distance_m']} 米。",
                evidence=evidence,
            )

        parts = []
        if left:
            parts.append(f"左墙横向距离约 {left['lateral_distance_m']} 米")
        if right:
            parts.append(f"右墙横向距离约 {right['lateral_distance_m']} 米")
        if left and right:
            corridor_width = round(left["lateral_distance_m"] + right["lateral_distance_m"], 3)
            parts.append(f"估计走廊宽度约 {corridor_width} 米")
        return QueryAnswer(answer="，".join(parts) + "。", evidence=evidence)

    def _solve_wall_from_annotations(
        self,
        walls: list[ObservedObject],
        query: SpatialQuery,
    ) -> QueryAnswer:
        evidence = [self._object_evidence(wall) for wall in walls]
        left_candidates = [item for item in evidence if item["side"] in {"left", "front-left", "rear-left"}]
        right_candidates = [item for item in evidence if item["side"] in {"right", "front-right", "rear-right"}]

        if query.side_hint == "left":
            if not left_candidates:
                return QueryAnswer(answer="没有可靠检测到左墙。", evidence=evidence)
            wall = min(left_candidates, key=lambda item: item["lateral_distance_m"])
            return QueryAnswer(
                answer=f"左墙在你的{self._side_name(wall['side'])}，横向距离约 {wall['lateral_distance_m']} 米，前向距离约 {wall['forward_distance_m']} 米。",
                evidence=evidence,
            )

        if query.side_hint == "right":
            if not right_candidates:
                return QueryAnswer(answer="没有可靠检测到右墙。", evidence=evidence)
            wall = min(right_candidates, key=lambda item: item["lateral_distance_m"])
            return QueryAnswer(
                answer=f"右墙在你的{self._side_name(wall['side'])}，横向距离约 {wall['lateral_distance_m']} 米，前向距离约 {wall['forward_distance_m']} 米。",
                evidence=evidence,
            )

        parts = []
        if left_candidates:
            left_wall = min(left_candidates, key=lambda item: item["lateral_distance_m"])
            parts.append(f"左墙横向距离约 {left_wall['lateral_distance_m']} 米")
        if right_candidates:
            right_wall = min(right_candidates, key=lambda item: item["lateral_distance_m"])
            parts.append(f"右墙横向距离约 {right_wall['lateral_distance_m']} 米")
        if left_candidates and right_candidates:
            corridor_width = round(
                min(left_candidates, key=lambda item: item["lateral_distance_m"])["lateral_distance_m"] +
                min(right_candidates, key=lambda item: item["lateral_distance_m"])["lateral_distance_m"],
                3,
            )
            parts.append(f"估计走廊宽度约 {corridor_width} 米")
        return QueryAnswer(answer="，".join(parts) + "。", evidence=evidence)

    def _estimate_side_wall(self, depth_m: np.ndarray, intrinsics: np.ndarray, side: str) -> Dict[str, object] | None:
        height, width = depth_m.shape
        y_start = int(height * 0.30)
        y_end = int(height * 0.75)
        if side == "left":
            x_start, x_end = 0, int(width * 0.20)
            sign = -1
        else:
            x_start, x_end = int(width * 0.80), width
            sign = 1

        roi = depth_m[y_start:y_end, x_start:x_end]
        valid = np.isfinite(roi)
        if int(valid.sum()) < 30:
            return None

        v_coords, u_coords = np.where(valid)
        z = roi[valid]
        u = u_coords.astype(np.float32) + x_start
        x_camera = ((u - intrinsics[0, 2]) / intrinsics[0, 0]) * z
        side_mask = x_camera < 0 if sign < 0 else x_camera > 0
        if int(side_mask.sum()) < 20:
            return None

        lateral = float(np.nanmedian(np.abs(x_camera[side_mask])))
        forward = float(np.nanmedian(z[side_mask]))
        return {
            "side": side,
            "lateral_distance_m": round(lateral, 3),
            "forward_distance_m": round(forward, 3),
            "distance_m": round(float(np.sqrt(lateral ** 2 + forward ** 2)), 3),
            "valid_point_count": int(side_mask.sum()),
        }

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
