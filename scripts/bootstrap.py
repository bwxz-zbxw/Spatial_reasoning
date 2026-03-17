from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"Project root: {root}")
    print("Initial project skeleton is ready.")


if __name__ == "__main__":
    main()
