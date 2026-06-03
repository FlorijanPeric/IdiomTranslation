from pathlib import Path
import re
import pandas as pd


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)


def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    cols = {column.lower(): column for column in df.columns}

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


def load_idiom_dataframe(data_root: str) -> pd.DataFrame:
    root = Path(data_root)
    csv_files = [path for path in root.rglob("*.csv")]
    print(f"Scanning CSV files under: {root} (found={len(csv_files)})")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        try:
            tmp = pd.read_csv(csv_path)
        except Exception:
            try:
                tmp = pd.read_csv(csv_path, sep=";")
            except Exception:
                print(f"Skipping unreadable CSV: {csv_path}")
                continue

        if tmp.empty:
            print(f"Skipping empty CSV: {csv_path}")
            continue

        idiom_col, meaning_col, translation_col = detect_columns(tmp)
        if idiom_col is None:
            print(f"Skipping CSV without idiom-like column: {csv_path}")
            continue

        use_cols = [idiom_col]
        if meaning_col:
            use_cols.append(meaning_col)
        if translation_col:
            use_cols.append(translation_col)

        subset = tmp[use_cols].copy()
        rename_map = {idiom_col: "idiom"}
        if meaning_col:
            rename_map[meaning_col] = "meaning"
        if translation_col:
            rename_map[translation_col] = "target_translation"
        subset = subset.rename(columns=rename_map)

        if "meaning" not in subset.columns:
            subset["meaning"] = ""
        if "target_translation" not in subset.columns:
            subset["target_translation"] = ""

        subset["source_file"] = str(csv_path)
        frames.append(subset)

    if not frames:
        raise RuntimeError(f"No usable idiom CSV files found under: {data_root}")

    idioms_df = pd.concat(frames, ignore_index=True)
    idioms_df = idioms_df.dropna(subset=["idiom"]).copy()
    idioms_df["idiom"] = idioms_df["idiom"].astype(str)
    idioms_df["meaning"] = idioms_df["meaning"].astype(str)
    idioms_df["target_translation"] = idioms_df["target_translation"].astype(str)
    idioms_df["idiom_norm"] = idioms_df["idiom"].map(normalize_text)
    idioms_df = idioms_df[idioms_df["idiom_norm"] != ""].copy()
    idioms_df = idioms_df.drop_duplicates(subset=["idiom_norm"]).reset_index(drop=True)

    print(f"Loaded idioms: {len(idioms_df)} unique rows")

    return idioms_df
