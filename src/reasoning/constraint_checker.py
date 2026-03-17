from typing import Any, Dict, List

from src.reasoning.protocol import ConstraintEvaluation, ReasoningRequest


class ConstraintChecker:
    """Evaluates structured constraints against computed scene facts."""

    def evaluate(
        self,
        request: ReasoningRequest,
        facts: Dict[str, Any],
    ) -> List[ConstraintEvaluation]:
        return [
            ConstraintEvaluation(
                name=constraint.name,
                operator=constraint.operator,
                expected_value=constraint.value,
                actual_value=facts.get(constraint.name, "missing"),
                passed=self._compare(facts.get(constraint.name), constraint.operator, constraint.value),
                source=constraint.source,
            )
            for constraint in request.constraints
        ]

    def _compare(self, actual: Any, operator: str, expected: Any) -> bool:
        if actual is None or actual == "missing":
            return False
        if actual == "inf":
            actual = float("inf")

        if operator == ">=":
            return actual >= expected
        if operator == "<=":
            return actual <= expected
        if operator == "==":
            return actual == expected
        raise ValueError(f"Unsupported operator: {operator}")
