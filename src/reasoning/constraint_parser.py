from src.geometry.facts import build_scene_facts
from src.geometry.scenario_loader import Scenario
from src.reasoning.protocol import ReasoningRequest, SpatialConstraint


class ConstraintParser:
    """Builds a structured reasoning request from a scenario."""

    def build_request(self, scenario: Scenario) -> ReasoningRequest:
        facts = build_scene_facts(scenario)
        constraints = [
            SpatialConstraint(
                name="clearance_m",
                operator=">=",
                value=scenario.thresholds["min_clearance_m"],
                source="safety_rule",
            ),
            SpatialConstraint(
                name="ttc_s",
                operator=">=",
                value=scenario.thresholds["min_ttc_s"],
                source="safety_rule",
            ),
        ]

        if facts.get("nearest_category") != "human":
            constraints.append(
                SpatialConstraint(
                    name="estimated_free_width_m",
                    operator=">=",
                    value=round(
                        facts["robot_width_m"] + scenario.thresholds["min_clearance_m"],
                        3,
                    ),
                    source="passability_rule",
                )
            )

        return ReasoningRequest(
            task="hotel_corridor_yielding",
            reference_frame="robot_base",
            constraints=constraints,
            context={
                "preferred_yield_side": "right",
                "nearest_category": facts.get("nearest_category", "none"),
            },
        )
