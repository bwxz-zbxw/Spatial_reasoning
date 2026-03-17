from src.reasoning.protocol import ReasoningRequest, SpatialConstraint


class ConstraintParser:
    """Builds a structured reasoning request from task and scene hints."""

    def build_default_yield_request(self) -> ReasoningRequest:
        return ReasoningRequest(
            task="yield_for_human_in_corridor",
            reference_frame="robot_base",
            constraints=[
                SpatialConstraint(
                    name="minimum_human_clearance_m",
                    operator=">=",
                    value=0.8,
                    source="safety_rule",
                ),
                SpatialConstraint(
                    name="time_to_collision_s",
                    operator=">=",
                    value=1.5,
                    source="safety_rule",
                ),
            ],
            context={"preferred_yield_side": "right"},
        )
