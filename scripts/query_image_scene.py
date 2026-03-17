import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.observation_loader import load_image_observation
from src.reasoning.spatial_qa import SpatialQuestionAnswerer


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a spatial question from one image observation.")
    parser.add_argument(
        "--observation",
        type=Path,
        default=Path("data/samples/hotel_corridor_observation.json"),
        help="Image observation JSON path.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="墙在我的哪边，离我有多远？",
        help="Spatial question in Chinese.",
    )
    args = parser.parse_args()

    observation = load_image_observation(args.observation)
    result = SpatialQuestionAnswerer().answer(observation, args.question)
    print(result.answer)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
