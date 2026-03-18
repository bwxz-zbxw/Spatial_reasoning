from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpatialQuery:
    raw_question: str
    target: str
    attributes: List[str]
    reference_frame: str = "robot_base"
    side_hint: Optional[str] = None
    reasoning_source: str = "template"
    metadata: Dict[str, Any] = field(default_factory=dict)
