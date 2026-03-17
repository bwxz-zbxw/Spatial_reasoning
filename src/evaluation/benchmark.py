from dataclasses import asdict
from typing import Any, Dict, List

from src.geometry.scenario_loader import Scenario
from src.reasoning.baseline_policy import BaselineYieldPolicy
from src.reasoning.constrained_policy import ConstrainedYieldPolicy


class PolicyBenchmark:
    """Runs all available policies on a scenario set."""

    def __init__(self) -> None:
        self.policies = {
            "baseline": BaselineYieldPolicy(),
            "constrained": ConstrainedYieldPolicy(),
        }

    def run(self, scenarios: Dict[str, Scenario]) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        totals = {
            policy_name: {"matches": 0, "total": 0}
            for policy_name in self.policies
        }

        for scenario in scenarios.values():
            for policy_name, policy in self.policies.items():
                decision = policy.decide(scenario)
                matched = decision.action == scenario.expected_action
                totals[policy_name]["total"] += 1
                totals[policy_name]["matches"] += int(matched)
                rows.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "policy": policy_name,
                        "expected_action": scenario.expected_action,
                        "predicted_action": decision.action,
                        "matched_expected_action": matched,
                        "confidence": decision.confidence,
                        "rationale": decision.rationale,
                        "facts": decision.facts,
                    }
                )

        summary = {
            policy_name: {
                "matches": data["matches"],
                "total": data["total"],
                "action_match_rate": round(
                    data["matches"] / data["total"], 3
                ) if data["total"] else 0.0,
            }
            for policy_name, data in totals.items()
        }

        return {"summary": summary, "rows": rows}


def format_markdown_report(results: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Policy Benchmark Report",
        "",
        "## Summary",
        "",
        "| Policy | Matches | Total | Action Match Rate |",
        "| --- | ---: | ---: | ---: |",
    ]

    for policy_name, metrics in results["summary"].items():
        lines.append(
            f"| {policy_name} | {metrics['matches']} | {metrics['total']} | {metrics['action_match_rate']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Scenario Results",
            "",
            "| Scenario | Policy | Expected | Predicted | Match | Rationale |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in results["rows"]:
        rationale = ", ".join(row["rationale"])
        match = "yes" if row["matched_expected_action"] else "no"
        lines.append(
            f"| {row['scenario_id']} | {row['policy']} | {row['expected_action']} | {row['predicted_action']} | {match} | {rationale} |"
        )

    return "\n".join(lines) + "\n"
