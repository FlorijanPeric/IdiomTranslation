import uuid
import argparse

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import ProjectConfig
from data_loader import load_idiom_dataframe
from embedder_api import embed_in_batches


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    collections = client.get_collections().collections
    if any(collection.name == collection_name for collection in collections):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def build_index(config: ProjectConfig, recreate_collection: bool = False) -> dict:
    idioms_df = load_idiom_dataframe(config.data_root)

    idiom_texts = idioms_df["idiom"].tolist()
    vectors = embed_in_batches(idiom_texts, config)
    if not vectors:
        raise RuntimeError("No vectors generated.")

    client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)
    if recreate_collection:
        collections = client.get_collections().collections
        if any(collection.name == config.qdrant_collection for collection in collections):
            client.delete_collection(collection_name=config.qdrant_collection)

    ensure_collection(client, config.qdrant_collection, vector_size=len(vectors[0]))

    points = []
    for index, row in idioms_df.iterrows():
        stable_key = f"{row['idiom_norm']}|{row['source_file']}"
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                vector=vectors[index],
                payload={
                    "idiom": row["idiom"],
                    "idiom_norm": row["idiom_norm"],
                    "meaning": row["meaning"],
                    "target_translation": row["target_translation"],
                    "source_file": row["source_file"],
                },
            )
        )

    client.upsert(collection_name=config.qdrant_collection, points=points, wait=True)
    return {"indexed_rows": len(points), "collection": config.qdrant_collection}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant idiom index")
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop existing collection and create a new one before indexing",
    )
    args = parser.parse_args()

    config = ProjectConfig()
    result = build_index(config, recreate_collection=args.recreate_collection)
    print(result)


if __name__ == "__main__":
    main()
