from pathlib import Path
import csv
import re


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)


def detect_columns(fieldnames: list[str]) -> tuple[str | None, str | None, str | None]:
    cols = {column.lower(): column for column in fieldnames}

    idiom_candidates = ["idiom", "idioms", "english_idiom", "expression", "phrase"]
    meaning_candidates = ["meaning", "definition", "explanation", "description"]
    translation_candidates = ["slovene", "slovenian", "translation", "prevod", "target_translation"]

    def pick(candidates: list[str]) -> str | None:
        for candidate in candidates:
            for lower_name, original_name in cols.items():
                if candidate in lower_name:
                    return original_name
        return None

    return pick(idiom_candidates), pick(meaning_candidates), pick(translation_candidates)


def _collect_csv_files(data_roots: list[str]) -> list[Path]:
    found: dict[str, Path] = {}
    skip_dir_names = {".venv", "venv", "__pycache__", ".git", "node_modules"}
    for root_item in data_roots:
        root_path = Path(root_item)
        if root_path.is_file() and root_path.suffix.lower() == ".csv":
            found[str(root_path.resolve())] = root_path
            continue
        if root_path.is_dir():
            for csv_path in root_path.rglob("*.csv"):
                if any(part in skip_dir_names for part in csv_path.parts):
                    continue
                found[str(csv_path.resolve())] = csv_path

    return list(found.values())


def _is_reverse_csv(csv_path: Path) -> bool:
    name = csv_path.name.lower()
    return "reverse" in name or "slovene_idioms" in name


def load_idiom_rows(data_roots: list[str], direction: str = "en_to_sl") -> list[dict]:
    """
    Load idiom rows from CSV files.
    
    direction:
      - "en_to_sl": English idioms -> Slovene translations (standard)
      - "sl_to_en": Slovene idioms -> English translations (reverse)
    """
    csv_files = _collect_csv_files(data_roots)
    print(f"Scanning CSV files under roots: {data_roots} (found={len(csv_files)}, direction={direction})")

    rows: list[dict] = []
    for csv_path in csv_files:
        if _is_reverse_csv(csv_path):
            continue

        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                file_rows = list(reader)
        except Exception:
            try:
                with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter=";")
                    file_rows = list(reader)
            except Exception:
                print(f"Skipping unreadable CSV: {csv_path}")
                continue

        if not file_rows:
            print(f"Skipping empty CSV: {csv_path}")
            continue

        fieldnames = list(file_rows[0].keys())
        idiom_col, meaning_col, translation_col = detect_columns(fieldnames)
        if idiom_col is None:
            print(f"Skipping CSV without idiom-like column: {csv_path}")
            continue

        for raw in file_rows:
            idiom = str(raw.get(idiom_col, "") or "").strip()
            if not idiom:
                continue

            meaning = str(raw.get(meaning_col, "") or "").strip() if meaning_col else ""
            translation = str(raw.get(translation_col, "") or "").strip() if translation_col else ""
            
            # For reverse direction, swap idiom and translation
            if direction == "sl_to_en":
                idiom, translation = translation, idiom
            
            if not idiom or not translation:
                continue
            
            rows.append(
                {
                    "idiom": idiom,
                    "meaning": meaning,
                    "target_translation": translation,
                    "source_file": str(csv_path),
                    "idiom_norm": normalize_text(idiom),
                }
            )

    if not rows:
        raise RuntimeError(f"No usable idiom CSV files found under roots: {data_roots}")

    unique_by_idiom: dict[str, dict] = {}
    for row in rows:
        idiom_norm = row.get("idiom_norm", "")
        if idiom_norm and idiom_norm not in unique_by_idiom:
            unique_by_idiom[idiom_norm] = row

    normalized_rows = list(unique_by_idiom.values())
    print(f"Loaded idioms: {len(normalized_rows)} unique rows (direction={direction})")
    return normalized_rows
