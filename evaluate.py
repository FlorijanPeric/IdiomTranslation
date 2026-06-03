import argparse
import csv

from qdrant_client import QdrantClient

from config import ProjectConfig
from embedder_api import embed_texts


SAMPLE_TEST = [
    ("spill the beans", "reveal"),
    ("break the ice", "conversation"),
    ("piece of cake", "easy"),
]


def _load_test_rows(test_csv: str | None) -> list[tuple[str, str]]:
    if not test_csv:
        return SAMPLE_TEST

    rows: list[tuple[str, str]] = []
    with open(test_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query = str(row.get("query") or row.get("idiom") or "").strip()
            expected = str(row.get("expected") or row.get("target") or row.get("keyword") or "").strip()
            if query and expected:
                rows.append((query, expected))

    if not rows:
        raise RuntimeError(f"No valid test rows found in: {test_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate idiom retrieval quality")
    parser.add_argument("--test-csv", help="Optional CSV with query/expected columns")
    args = parser.parse_args()

    test_rows = _load_test_rows(args.test_csv)

    config = ProjectConfig()
    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)

    hits = 0
    for query, expected in test_rows:
        query_vec = embed_texts([query], config)[0]
        result = client.search(
            collection_name=config.qdrant_collection,
            query_vector=query_vec,
            limit=1,
            with_payload=True,
        )

        if not result:
            continue

        payload = result[0].payload or {}
        top_text = " ".join(
            [
                str(payload.get("idiom", "")),
                str(payload.get("target_translation", "")),
                str(payload.get("meaning", "")),
            ]
        ).lower()
        if expected.lower() in top_text:
            hits += 1

    score = hits / len(test_rows)
    print({"hit_at_1": score, "hits": hits, "total": len(test_rows)})


if __name__ == "__main__":
    main()
