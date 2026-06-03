from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    source_path = base_dir / "data" / "english_idioms_optimized.csv"
    target_path = base_dir / "data" / "slovene_idioms_reverse.csv"

    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_path}")

    with source_path.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        source_rows = list(reader)

    output_rows: list[dict[str, str]] = []
    seen_idioms: set[str] = set()

    for row in source_rows:
        english_idiom = str(row.get("idiom", "") or "").strip()
        meaning = str(row.get("meaning", "") or "").strip()
        slovene_idiom = str(row.get("slovene_translation", "") or "").strip()
        language = str(row.get("language", "") or "").strip() or "sl"
        domain = str(row.get("domain", "") or "").strip()
        register = str(row.get("register", "") or "").strip()

        if not slovene_idiom:
            continue

        norm = " ".join(slovene_idiom.lower().split())
        if not norm or norm in seen_idioms:
            continue
        seen_idioms.add(norm)

        output_rows.append(
            {
                "idiom": slovene_idiom,
                "meaning": meaning,
                "language": language,
                "domain": domain,
                "register": register,
                "english_idiom": english_idiom,
                "target_translation": english_idiom,
            }
        )

    fieldnames = [
        "idiom",
        "meaning",
        "language",
        "domain",
        "register",
        "english_idiom",
        "target_translation",
    ]

    with target_path.open("w", encoding="utf-8", newline="") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print({
        "source": str(source_path),
        "target": str(target_path),
        "rows_written": len(output_rows),
    })


if __name__ == "__main__":
    main()
