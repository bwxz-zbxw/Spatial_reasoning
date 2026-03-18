from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from src.geometry.sunrgbd_solver import SUNRGBDGeometrySolver
from src.perception.rgbd_visual_detector import RGBDVisualDetector
from src.perception.sunrgbd_loader import SUNRGBDLoader
from src.reasoning.question_formalizer import build_question_formalizer


@dataclass
class PipelineResult:
    sample_dir: str
    question: str
    formalized_query: Dict[str, Any]
    answer: str
    evidence: list[Dict[str, Any]]
    perception_backend: str


class SUNRGBDPipeline:
    """Question formalization + SUNRGBD geometry solving."""

    def __init__(self) -> None:
        self.formalizer = build_question_formalizer()
        self.loader = SUNRGBDLoader()
        self.solver = SUNRGBDGeometrySolver()
        self.detector = RGBDVisualDetector()

    def run(self, sample_dir: Path, question: str) -> PipelineResult:
        query = self.formalizer.formalize(question)
        sample = self.loader.load_sample(sample_dir)
        depth_m = self.loader.load_depth_meters(sample)
        image_rgb = self.loader.load_rgb_array(sample)
        detection_result = self.detector.detect(image_rgb, depth_m, sample.intrinsics)
        solved = self.solver.solve(sample, query, depth_m, detection_result.objects)
        return PipelineResult(
            sample_dir=str(sample_dir),
            question=question,
            formalized_query=asdict(query),
            answer=solved.answer,
            evidence=solved.evidence,
            perception_backend=detection_result.source if not detection_result.error else f"{detection_result.source}_fallback",
        )
