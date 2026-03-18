from dataclasses import dataclass
from typing import List

from src.geometry.spatial_language import (
    describe_side,
    euclidean_distance_3d,
    filter_by_category,
    forward_distance,
    lateral_distance,
    nearest_object,
)
from src.perception.observation_protocol import ImageObservation, ObservedObject


@dataclass
class SpatialAnswer:
    question: str
    target_category: str
    answer: str
    evidence: List[dict]


class SpatialQuestionAnswerer:
    """Answers simple spatial questions from image-derived object observations."""

    def answer(self, observation: ImageObservation, question: str) -> SpatialAnswer:
        target_category = self._infer_target_category(question)
        objects = filter_by_category(observation.objects, target_category)
        if not objects:
            return SpatialAnswer(
                question=question,
                target_category=target_category,
                answer=f"没有检测到类别为 {target_category} 的目标，暂时无法回答。",
                evidence=[],
            )

        if "最近" in question:
            selected = [nearest_object(objects)]
        else:
            selected = objects

        evidence = [self._build_evidence(obj) for obj in selected if obj is not None]
        answer = self._compose_answer(target_category, evidence, question)
        return SpatialAnswer(
            question=question,
            target_category=target_category,
            answer=answer,
            evidence=evidence,
        )

    def _infer_target_category(self, question: str) -> str:
        if "墙" in question:
            return "wall"
        if "门" in question:
            return "door"
        if "人" in question or "行人" in question:
            return "human"
        if "桌" in question:
            return "table"
        return "wall"

    def _build_evidence(self, obj: ObservedObject) -> dict:
        position = obj.position_robot_frame
        return {
            "object_id": obj.object_id,
            "category": obj.category,
            "side": describe_side(position),
            "distance_m": round(euclidean_distance_3d(position), 3),
            "lateral_distance_m": round(lateral_distance(position), 3),
            "forward_distance_m": round(forward_distance(position), 3),
            "confidence": obj.confidence,
        }

    def _compose_answer(self, target_category: str, evidence: List[dict], question: str) -> str:
        if not evidence:
            return f"没有足够的 {target_category} 观测结果。"

        if len(evidence) == 1:
            item = evidence[0]
            return (
                f"{self._category_name(target_category)}在你的{self._side_name(item['side'])}，"
                f"直线距离约 {item['distance_m']} 米，"
                f"横向距离约 {item['lateral_distance_m']} 米。"
            )

        parts = []
        for item in evidence:
            parts.append(
                f"{item['object_id']} 在你的{self._side_name(item['side'])}，距离约 {item['distance_m']} 米"
            )
        if "哪边" in question and "多远" in question:
            return f"检测到多个{self._category_name(target_category)}：" + "；".join(parts) + "。"
        return "；".join(parts) + "。"

    def _category_name(self, category: str) -> str:
        mapping = {"wall": "墙", "door": "门", "human": "行人", "table": "桌子"}
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
