import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "career_progression",
    "company_culture",
    "non_salary_benefits",
    "work_meaning_impact",
    "schedule_flexibility",
]


INPUT_PARQUET = Path("data/JOCAS/jocas_small_all_half_sirenised_embed_job_title_drop_stage.parquet")
CHECKPOINT_PATH = Path("data/annotation/prompt_v4/checkpoint_annotations_openai.jsonl")
OUTPUT_PATH = Path("data/annotation/set_labeled_recovered.parquet")


def extract_rome_domain(code: str) -> str:
    """
    Extrait la première lettre du code ROME.
    Ex : 'H1206' → 'H' | NaN/invalide → 'UNKNOWN'
    """
    if not isinstance(code, str) or not re.match(r"^[A-Z]\d{4}$", code.strip()):
        return "UNKNOWN"
    return code.strip()[0]


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    text_col: str = "description",
    rome_col: str = "job_ROME_code",
    min_text_len: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reproduit exactement l'échantillonnage stratifié utilisé dans annotation.py.
    """
    rng = np.random.default_rng(random_state)

    df = df.copy()
    df = df[df[text_col].fillna("").str.len() >= min_text_len].reset_index(drop=True)

    if len(df) < n:
        raise ValueError(
            f"Seulement {len(df)} offres valides après filtrage, impossible d'en tirer {n}."
        )

    df["rome_domain"] = df[rome_col].apply(extract_rome_domain)

    domain_counts = df["rome_domain"].value_counts()
    total = domain_counts.sum()
    quotas = {}
    remaining = n

    strata = domain_counts.index.tolist()

    min_per_stratum = 5
    for domain in strata:
        count = int(domain_counts[domain])
        guaranteed = min(min_per_stratum, count)
        quotas[domain] = guaranteed
        remaining -= guaranteed

    if remaining < 0:
        scale = n / sum(quotas.values())
        quotas = {k: max(1, int(v * scale)) for k, v in quotas.items()}
        remaining = 0

    if remaining > 0:
        proportional_weights = {d: domain_counts[d] / total for d in strata}
        for domain in strata:
            extra = int(remaining * proportional_weights[domain])
            available = int(domain_counts[domain]) - quotas[domain]
            quotas[domain] += min(extra, available)

    sampled_total = sum(quotas.values())
    diff = n - sampled_total

    if diff != 0:
        biggest = max(quotas, key=lambda k: domain_counts[k])
        quotas[biggest] += diff
        quotas[biggest] = max(0, quotas[biggest])

    frames = []
    for domain, quota in quotas.items():
        if quota <= 0:
            continue

        pool = df[df["rome_domain"] == domain]
        k = min(quota, len(pool))

        sampled = pool.sample(
            n=k,
            random_state=int(rng.integers(0, 1_000_000)),
        )

        frames.append(sampled)

    result = pd.concat(frames, ignore_index=True)

    print("\n📊 Distribution de l'échantillon reconstruit par domaine ROME :")
    dist = result["rome_domain"].value_counts().sort_index()
    for domain, count in dist.items():
        pct = count / len(result) * 100
        print(f"   {domain} : {count:4d} offres ({pct:.1f}%)")
    print(f"   TOTAL : {len(result)} offres\n")

    return result

# 2. Recover annotion from startified sampling

def recover_annotations(
    input_parquet: Path,
    checkpoint_path: Path,
    output_path: Path,
    n: int,
    text_col: str = "description",
    rome_col: str = "job_ROME_code",
    seed: int = 42,
    id_col: str | None = None,
) -> pd.DataFrame:

    df = pd.read_parquet(input_parquet)

    df_sample = stratified_sample(
        df,
        n=n,
        text_col=text_col,
        rome_col=rome_col,
        random_state=seed,
    )

    df_sample = df_sample.copy().reset_index(drop=True)

    if id_col is None:
        df_sample["_annotation_id"] = df_sample.index.astype(str)
    else:
        df_sample["_annotation_id"] = df_sample[id_col].astype(str)

    # 2. Charger le checkpoint JSONL
    annotations = {}

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ann = json.loads(line)
            annotations[str(ann["id"])] = ann

    df_annotations = (
        pd.DataFrame(list(annotations.values()))
        .rename(columns={"id": "_annotation_id"})
    )

    # 3. Sécuriser les colonnes
    if "_annotation_id" not in df_annotations.columns:
        df_annotations["_annotation_id"] = pd.Series(dtype=str)

    df_annotations["_annotation_id"] = df_annotations["_annotation_id"].astype(str)

    for feat in FEATURES:
        if feat not in df_annotations.columns:
            df_annotations[feat] = np.nan

    # 4. Merge avec les offres originales
    df_result = df_sample.merge(
        df_annotations,
        on="_annotation_id",
        how="left",
    )

    # 5. Sauvegarde
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_parquet(output_path, index=False)

    print(f"✅ Fichier récupéré sauvegardé : {output_path}")
    print(f"Lignes totales      : {len(df_result)}")
    print(f"Offres annotées     : {df_result[FEATURES[0]].notna().sum()}")
    print(f"Offres sans labels  : {df_result[FEATURES[0]].isna().sum()}")

    return df_result


if __name__ == "__main__":
    recover_annotations(
        input_parquet=Path("data/JOCAS/jocas_small_all_half_sirenised_embed_job_title_drop_stage.parquet"),
        checkpoint_path=Path("data/annotation/prompt_v4/checkpoint_annotations_openai.jsonl"),
        output_path=Path("data/annotation/prompt_v4/training_set_labeled_recovered.parquet"),
        n=10000,
        text_col="description",
        rome_col="job_ROME_code",
        seed=42,
        id_col=None,
    )