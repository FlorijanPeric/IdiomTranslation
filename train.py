import argparse

from build_index import build_index
from config import ProjectConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/update the idiom retrieval index")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate Qdrant collection before indexing",
    )
    args = parser.parse_args()

    config = ProjectConfig()
    result = build_index(config, recreate_collection=args.recreate_collection)
    print(result)


if __name__ == "__main__":
    main()
