from typing import List

from src.geometry.facts import build_scene_facts
from src.geometry.scenario_loader import Scenario
from src.reasoning.constraint_checker import ConstraintChecker
from src.reasoning.constraint_parser import ConstraintParser
from src.reasoning.protocol import ConstraintEvaluation, ReasoningDecision


class ConstrainedYieldPolicy:
    """GCA-style policy: request -> facts -> evaluations -> action."""

    def __init__(self) -> None:
        self.parser = ConstraintParser()
        self.checker = ConstraintChecker()

    def decide(self, scenario: Scenario) -> ReasoningDecision:
        facts = build_scene_facts(scenario)
        request = self.parser.build_request(scenario)
        evaluations = self.checker.evaluate(request, facts)
        failures = [item for item in evaluations if not item.passed]

        if not failures:
            return ReasoningDecision(
                action="proceed",
                confidence=0.9,
                rationale=["all_constraints_satisfied"],
                facts={**facts, "constraint_evaluations": self._serialize(evaluations)},
            )

        action, confidence, rationale = self._resolve_action(failures, request.context)
        return ReasoningDecision(
            action=action,
            confidence=confidence,
            rationale=rationale,
            facts={**facts, "constraint_evaluations": self._serialize(evaluations)},
        )

    def _resolve_action(
        self,
        failures: List[ConstraintEvaluation],
        context: dict,
    ) -> tuple[str, float, List[str]]:
        nearest_category = context.get("nearest_category", "none")
        failed_names = {item.name for item in failures}

        if "clearance_m" in failed_names and nearest_category == "human":
            return "stop_and_wait", 0.93, ["human_clearance_constraint_failed"]

        if "estimated_free_width_m" in failed_names:
            return "local_replan", 0.88, ["passability_constraint_failed"]

        if "ttc_s" in failed_names:
            if nearest_category == "human":
                preferred_side = context.get("preferred_yield_side", "right")
                return f"yield_{preferred_side}", 0.84, ["time_to_collision_constraint_failed"]
            return "slow_down", 0.82, ["time_to_collision_constraint_failed"]

        return "slow_down", 0.75, ["generic_constraint_failure"]

    def _serialize(self, evaluations: List[ConstraintEvaluation]) -> list[dict]:
        return [
            {
                "name": item.name,
                "operator": item.operator,
                "expected_value": item.expected_value,
                "actual_value": item.actual_value,
                "passed": item.passed,
                "source": item.source,
            }
            for item in evaluations
        ]
