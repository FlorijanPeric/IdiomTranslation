import argparse

from build_index import build_index
from config import ProjectConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/update local index using remote embedder")
    parser.add_argument("--recreate-collection", action="store_true", help="Drop and recreate collection")
    parser.add_argument(
        "--direction",
        choices=["en_to_sl", "sl_to_en"],
        default="en_to_sl",
        help="Translation direction to train",
    )
    args = parser.parse_args()

    config = ProjectConfig()
    result = build_index(
        config,
        recreate_collection=args.recreate_collection,
        direction=args.direction,
    )
    print(result)


if __name__ == "__main__":
    main()
