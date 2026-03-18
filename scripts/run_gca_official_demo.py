import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_GCA_ROOT = PROJECT_ROOT / "external" / "gca_official"

if str(OFFICIAL_GCA_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_GCA_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official GCA AgentWorkflow in interactive mode on local images.")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Natural-language spatial reasoning question.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        required=True,
        help="One or more local image paths.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gca_official_none.json",
        help="Path to the interactive GCA config JSON.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Optional fixed session id.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    missing = [str(path) for path in args.image if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image files: {missing}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not OFFICIAL_GCA_ROOT.exists():
        raise FileNotFoundError(f"Official GCA repo not found: {OFFICIAL_GCA_ROOT}")

    os.environ["AGENT_CONFIG_FILE"] = str(args.config.resolve())
    config_data = json.loads(args.config.read_text(encoding="utf-8"))

    from workflow.workflow import AgentWorkflow

    workflow = AgentWorkflow()
    try:
        final_state = await workflow.arun(
            instruction=args.question,
            images=[str(path.resolve()) for path in args.image],
            session_id=args.session_id,
        )
        numeric_result, summary = workflow.get_final_answer(final_state)

        payload = {
            "question": args.question,
            "images": [str(path.resolve()) for path in args.image],
            "numeric_result": _safe_to_jsonable(numeric_result),
            "summary": summary,
            "session_id": final_state.get("session_id"),
            "work_dir": str((PROJECT_ROOT / config_data.get("work_dir", "results")).resolve()),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        workflow.shutdown()


def _safe_to_jsonable(value):
    if value is None:
        return None
    if hasattr(value, "__dict__"):
        try:
            return value.__dict__
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


if __name__ == "__main__":
    asyncio.run(main())
