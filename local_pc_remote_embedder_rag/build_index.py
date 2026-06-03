import argparse
import uuid

from config import ProjectConfig
from data_loader import load_idiom_rows
from embedder_api import embed_texts
from qdrant_api import QdrantHttpClient


def ensure_collection(client: QdrantHttpClient, collection_name: str, vector_size: int) -> None:
    collections = client.list_collections()
    if collection_name in collections:
        return

    client.create_collection(collection_name=collection_name, vector_size=vector_size)


def build_index(config: ProjectConfig, recreate_collection: bool = False, direction: str = "en_to_sl") -> dict:
    collection_name = config.collection_for_direction(direction)
    idiom_rows = load_idiom_rows(config.data_roots(), direction=direction)
    total_rows = len(idiom_rows)

    # Adjust text labels based on direction
    if direction == "sl_to_en":
        source_label = "slovene_idiom"
        target_label = "english_translation"
    else:
        source_label = "english_idiom"
        target_label = "slovene_translation"

    idiom_texts = []
    for row in idiom_rows:
        searchable_text = "\n".join(
            [
                f"{source_label}: {row.get('idiom', '')}",
                f"meaning: {row.get('meaning', '')}",
                f"{target_label}: {row.get('target_translation', '')}",
            ]
        ).strip()
        idiom_texts.append(searchable_text)

    client = QdrantHttpClient(url=config.qdrant_url, api_key=config.qdrant_api_key, timeout=30)
    if recreate_collection:
        collections = client.list_collections()
        if collection_name in collections:
            client.delete_collection(collection_name=collection_name)

    vectors: list[list[float]] = []
    points_buffer: list[dict] = []
    indexed_rows = 0
    collection_ready = False
    fallback_count = 0

    try:
        for index, row in enumerate(idiom_rows):
            primary_text = idiom_texts[index]
            idiom_only_text = str(row.get("idiom", "")).strip()

            try:
                vector = embed_texts([primary_text], config)[0]
            except Exception as primary_error:
                if not idiom_only_text:
                    raise RuntimeError(
                        f"Embedding failed for row {index} and no idiom fallback exists."
                    ) from primary_error

                try:
                    vector = embed_texts([idiom_only_text], config)[0]
                    fallback_count += 1
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Embedding failed for row {index} idiom='{idiom_only_text}'. "
                        f"Primary error: {primary_error}; Fallback error: {fallback_error}"
                    ) from fallback_error

            vectors.append(vector)

            if not collection_ready:
                ensure_collection(client, collection_name, vector_size=len(vector))
                collection_ready = True

            stable_key = f"{row['idiom_norm']}|{row['source_file']}"
            points_buffer.append(
                {
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                    "vector": vector,
                    "payload": {
                        "idiom": row["idiom"],
                        "idiom_norm": row["idiom_norm"],
                        "meaning": row["meaning"],
                        "target_translation": row["target_translation"],
                        "searchable_text": idiom_texts[index],
                        "source_file": row["source_file"],
                    },
                }
            )

            if len(points_buffer) >= 25:
                client.upsert_points(collection_name=collection_name, points=points_buffer, wait=True)
                indexed_rows += len(points_buffer)
                print(f"Indexed {indexed_rows}/{total_rows} rows ({direction})")
                points_buffer = []

        if points_buffer:
            client.upsert_points(collection_name=collection_name, points=points_buffer, wait=True)
            indexed_rows += len(points_buffer)
            print(f"Indexed {indexed_rows}/{total_rows} rows ({direction})")

    except KeyboardInterrupt:
        if points_buffer and collection_ready:
            client.upsert_points(collection_name=collection_name, points=points_buffer, wait=True)
            indexed_rows += len(points_buffer)
            print(f"Training interrupted. Saved progress: {indexed_rows}/{total_rows} rows ({direction})")
        raise

    if not vectors:
        raise RuntimeError("No vectors generated.")

    return {
        "indexed_rows": indexed_rows,
        "collection": collection_name,
        "direction": direction,
        "fallback_embed_count": fallback_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant index")
    parser.add_argument("--recreate-collection", action="store_true", help="Drop existing collection first")
    parser.add_argument("--direction", choices=["en_to_sl", "sl_to_en"], default="en_to_sl", help="Translation direction")
    args = parser.parse_args()

    config = ProjectConfig()
    result = build_index(config, recreate_collection=args.recreate_collection, direction=args.direction)
    print(result)


if __name__ == "__main__":
    main()
