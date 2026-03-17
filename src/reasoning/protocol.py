from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SpatialConstraint:
    name: str
    operator: str
    value: Any
    source: str


@dataclass
class ReasoningRequest:
    task: str
    reference_frame: str
    constraints: List[SpatialConstraint] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningDecision:
    action: str
    confidence: float
    rationale: List[str]
    facts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintEvaluation:
    name: str
    operator: str
    expected_value: Any
    actual_value: Any
    passed: bool
    source: str
