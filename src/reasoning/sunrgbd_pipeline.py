from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from src.geometry.sunrgbd_solver import SUNRGBDGeometrySolver
from src.perception.sunrgbd_loader import SUNRGBDLoader
from src.reasoning.question_formalizer import QuestionFormalizer


@dataclass
class PipelineResult:
    sample_dir: str
    question: str
    formalized_query: Dict[str, Any]
    answer: str
    evidence: list[Dict[str, Any]]


class SUNRGBDPipeline:
    """Question formalization + SUNRGBD geometry solving."""

    def __init__(self) -> None:
        self.formalizer = QuestionFormalizer()
        self.loader = SUNRGBDLoader()
        self.solver = SUNRGBDGeometrySolver()

    def run(self, sample_dir: Path, question: str) -> PipelineResult:
        query = self.formalizer.formalize(question)
        sample = self.loader.load_sample(sample_dir)
        depth_m = self.loader.load_depth_meters(sample)
        solved = self.solver.solve(sample, query, depth_m)
        return PipelineResult(
            sample_dir=str(sample_dir),
            question=question,
            formalized_query=asdict(query),
            answer=solved.answer,
            evidence=solved.evidence,
        )
