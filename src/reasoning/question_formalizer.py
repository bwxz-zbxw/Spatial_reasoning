import re

from src.reasoning.query_protocol import SpatialQuery


class QuestionFormalizer:
    """Formalizes Chinese spatial questions into a structured query."""

    def formalize(self, question: str) -> SpatialQuery:
        return SpatialQuery(
            raw_question=question,
            target=self._infer_target(question),
            attributes=self._infer_attributes(question),
            reference_frame="robot_base",
            side_hint=self._infer_side_hint(question),
            reasoning_source="template",
            metadata={"question_type": "sunrgbd_spatial_query"},
        )

    def _infer_target(self, question: str) -> str:
        mapping = {
            "墙": "wall",
            "桌": "table",
            "门": "door",
            "椅": "chair",
            "沙发": "sofa",
            "柜": "cabinet",
            "人": "human",
            "行人": "human",
            "垃圾桶": "garbage_bin",
            "冰箱": "fridge",
            "台": "table",
        }
        for token, target in mapping.items():
            if token in question:
                return target
        return "wall"

    def _infer_attributes(self, question: str) -> list[str]:
        attributes: list[str] = []
        if re.search("哪边|哪里|左|右|前|后", question):
            attributes.append("side")
        if "多远" in question or "距离" in question:
            attributes.append("distance")
        if "多宽" in question or "宽度" in question:
            attributes.append("width")
        if not attributes:
            attributes = ["side", "distance"]
        return attributes

    def _infer_side_hint(self, question: str) -> str | None:
        if "最近" in question:
            return "nearest"
        if "左边" in question or "左侧" in question:
            return "left"
        if "右边" in question or "右侧" in question:
            return "right"
        return None
