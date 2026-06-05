import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input file must be .parquet or .csv")


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Output file must be .parquet or .csv")


def predict_proba_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 1024,
) -> np.ndarray:
    all_probs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits.detach().cpu().numpy()

        all_probs.append(sigmoid(logits))

    return np.vstack(all_probs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict job posting features with a fine-tuned ModernCamemBERT model."
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the inference model folder.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input .parquet or .csv file containing job postings.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .parquet or .csv file with predictions.",
    )
    parser.add_argument(
        "--text-col",
        default="description",
        help="Name of the column containing job posting text. Default: description.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Prediction batch size. Default: 16.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum number of tokens. Default: 1024.",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    labels_path = model_dir / "labels.json"
    thresholds_path = model_dir / "thresholds.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.json in {model_dir}")

    if not thresholds_path.exists():
        raise FileNotFoundError(f"Missing thresholds.json in {model_dir}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    df = load_table(input_path)

    if args.text_col not in df.columns:
        raise ValueError(
            f"Text column '{args.text_col}' not found. Available columns: {list(df.columns)}"
        )

    texts = (
        df[args.text_col]
        .fillna("")
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    probs = predict_proba_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if probs.shape[1] != len(labels):
        raise ValueError(
            f"Model outputs {probs.shape[1]} labels, but labels.json contains {len(labels)} labels."
        )

    result = df.copy()

    for i, label in enumerate(labels):
        threshold = float(thresholds[label])

        result[f"proba_{label}"] = probs[:, i]
        result[f"pred_{label}"] = (probs[:, i] >= threshold).astype(int)

    save_table(result, output_path)

    print(f"Saved predictions to: {output_path}")
    print(f"Number of rows: {len(result):,}")
    print(f"Labels predicted: {len(labels)}")


if __name__ == "__main__":
    main()