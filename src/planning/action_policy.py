from src.reasoning.protocol import ReasoningDecision


class ActionPolicy:
    """Maps structured decisions to navigation-level commands."""

    def to_nav_command(self, decision: ReasoningDecision) -> dict:
        return {
            "action": decision.action,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
        }
