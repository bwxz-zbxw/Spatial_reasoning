import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import PolicyBenchmark, format_markdown_report
from src.geometry.scenario_loader import load_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all policies on all scenarios.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scenarios.yaml"),
        help="Scenario config path.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/metrics/policy_benchmark.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("results/metrics/policy_benchmark.md"),
        help="Markdown report output path.",
    )
    args = parser.parse_args()

    scenarios = load_scenarios(args.config)
    results = PolicyBenchmark().run(scenarios)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.md_out.write_text(
        format_markdown_report(results),
        encoding="utf-8",
    )

    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"JSON report written to {args.json_out}")
    print(f"Markdown report written to {args.md_out}")


if __name__ == "__main__":
    main()
