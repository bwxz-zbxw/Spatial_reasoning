import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.sunrgbd_pipeline import SUNRGBDPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SUNRGBD spatial QA on one sample.")
    parser.add_argument(
        "--sample-dir",
        type=Path,
        required=True,
        help="Path to a SUNRGBD sample directory, e.g. SUNRGBD/kv1/NYUdata/NYU0001",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="墙在我的哪边，离我有多远？",
        help="Chinese spatial question.",
    )
    args = parser.parse_args()

    result = SUNRGBDPipeline().run(args.sample_dir, args.question)
    print(result.answer)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
