import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.scenario_loader import load_scenarios
from src.reasoning.baseline_policy import BaselineYieldPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rule baseline on a scenario.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scenarios.yaml"),
        help="Scenario config path.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Scenario id to evaluate.",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.config)
    scenario = scenarios[args.scenario]
    decision = BaselineYieldPolicy().decide(scenario)
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
