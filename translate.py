import argparse

from config import ProjectConfig
from retriever import retrieve_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Query idiom translator index")
    parser.add_argument("--query", "-q", required=True, help="Idiom to translate")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    config = ProjectConfig()
    hits = retrieve_candidates(args.query, config, top_k=args.top_k)

    print(f"Query: {args.query}\n")
    if not hits:
        print("No confident match found.")
        print("Try a clearer idiom phrase or lower RETRIEVAL_MIN_SCORE.")
        return

    for rank, hit in enumerate(hits, start=1):
        print(f"[{rank}] score={hit['score']:.4f}")
        print("idiom:", hit["idiom"])
        print("translation:", hit["translation"])
        print("meaning:", hit["meaning"])
        print()


if __name__ == "__main__":
    main()
