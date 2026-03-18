import json
import os
import re
from pathlib import Path
from typing import Any

from src.reasoning.query_protocol import SpatialQuery


SYSTEM_PROMPT = """你是空间推理系统里的“问题形式化模块”。
你的任务是把中文问题转成 JSON。
只输出 JSON，不要输出解释，不要输出 markdown。

字段要求：
- target: 目标类别。可选值优先使用 wall, table, chair, door, sofa, cabinet, human, garbage_bin, fridge, unknown
- attributes: 需要回答的属性列表，可选值为 side, distance, width
- reference_frame: 固定输出 robot_base
- side_hint: 可选值 left, right, nearest, null

示例：
问题：墙在我的哪边，离我有多远？
输出：
{"target":"wall","attributes":["side","distance"],"reference_frame":"robot_base","side_hint":null}

问题：左边的桌子离我多远？
输出：
{"target":"table","attributes":["distance"],"reference_frame":"robot_base","side_hint":"left"}
"""


class TemplateQuestionFormalizer:
    """Deterministic fallback for spatial question formalization."""

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
        return "unknown"

    def _infer_attributes(self, question: str) -> list[str]:
        attributes: list[str] = []
        if re.search(r"哪边|哪里|左|右|前|后", question):
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


class LLMQuestionFormalizer:
    """Local-model formalizer for Qwen2.5-7B-Instruct with template fallback."""

    def __init__(self) -> None:
        self.fallback = TemplateQuestionFormalizer()
        self._model = None
        self._tokenizer = None
        self._load_error = None
        self.model_path = self._resolve_model_path()

    def formalize(self, question: str) -> SpatialQuery:
        if not self.model_path:
            return self.fallback.formalize(question)

        try:
            self._lazy_load()
            raw_output = self._generate(question)
            payload = self._extract_json(raw_output)
            return SpatialQuery(
                raw_question=question,
                target=str(payload.get("target", self.fallback._infer_target(question))),
                attributes=list(payload.get("attributes", self.fallback._infer_attributes(question))),
                reference_frame=str(payload.get("reference_frame", "robot_base")),
                side_hint=payload.get("side_hint"),
                reasoning_source="llm",
                metadata={
                    "question_type": "sunrgbd_spatial_query",
                    "model_path": str(self.model_path),
                    "raw_model_output": raw_output,
                },
            )
        except Exception as exc:
            self._load_error = str(exc)
            query = self.fallback.formalize(question)
            query.metadata["llm_error"] = self._load_error
            return query

    def _resolve_model_path(self) -> Path | None:
        env_path = os.getenv("QWEN_MODEL_PATH")
        candidates = [
            Path(env_path) if env_path else None,
            Path("/mnt/workspace/.cache/modelscope/models/Qwen/Qwen2.5-7B-Instruct"),
            Path("/mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"),
            Path.home() / ".cache" / "modelscope" / "models" / "Qwen" / "Qwen2.5-7B-Instruct",
            Path.home() / ".cache" / "modelscope" / "hub" / "models" / "Qwen" / "Qwen2.5-7B-Instruct",
        ]
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        return None

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def _generate(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
        )
        new_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
        return self._tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    def _extract_json(self, text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Model output does not contain JSON: {text}")
        payload = json.loads(match.group(0))
        if not isinstance(payload, dict):
            raise ValueError(f"Model output JSON is not an object: {text}")
        return payload


def build_question_formalizer() -> LLMQuestionFormalizer:
    return LLMQuestionFormalizer()
