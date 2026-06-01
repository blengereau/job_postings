import pandas as pd
import re
import unicodedata
import argparse
import os

def clean_job_title(text):
    if pd.isna(text):
        return pd.NA

    text = str(text).strip().lower()

    text = ''.join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

    text = text.replace("/", " ")
    text = text.replace("-", " ")

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

df["job_title_clean"] = df["job_title"].apply(clean_job_title)

mask_stage = df["job_title_clean"].str.contains(r"\bstage\b", na=False)
df_without_stage = df[~mask_stage]
df_without_stage.to_parquet("data/JOCAS/jocas_small_all_half_sirenised_embed_job_title_drop_stage.parquet")

def main():
    parser = argparse.ArgumentParser("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_cleaned.parquet")
    
    args = parser.parse_args()
    df = pd.read_parquet(args.input)
    print("Nettoyage en cours...")
    df["job_title_clean"] = df["job_title"].apply(clean_job_title)
    mask_stage = df["job_title_clean"].str.contains(r"\bstage\b", na=False)
    df_without_stage = df[~mask_stage]

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_without_stage.to_parquet(args.output)
    print(f"Lignes supprimées (stages) : {len(df) - len(df_without_stage)}")

if __name__ == "__main__":
    main()