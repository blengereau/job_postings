import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import json
import math
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False


# CONFIG

MODEL_NAME = "almanach/moderncamembert-base"
INPUT_PATH = "data/annotation/annoted_set.parquet"
OUTPUT_DIR = "models/moderncamembert_job_features_full_text"

TEXT_COL = "description"

LABELS = [
    "career_progression",
    "company_culture",
    "non_salary_benefits",
    "work_meaning_impact",
    "schedule_flexibility",
]

# ModernCamemBERT / ModernBERT long context.
# Ce n'est pas "infini" : les offres au-delà de 8192 tokens seront tronquées.
MAX_LENGTH = 8192

SEED = 42

NUM_TRAIN_EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

# Avec un gros GPU, tu peux augmenter à 4 ou 8.
# Pour du 8192 tokens, 2 est déjà sérieux.
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8

USE_GRADIENT_CHECKPOINTING = True

USE_POS_WEIGHT = False
MIN_PRECISION_FOR_THRESHOLDS = None
# Mets par exemple 0.75 ou 0.80 si tu veux privilégier la précision
# lors du passage sur les 800k offres.

TOKENIZE_BATCH_SIZE = 16


# REPRODUCIBILITY

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")


# TEXT PREPARATION

def minimal_clean(text):
    if pd.isna(text):
        return ""
    text = str(text).replace("\xa0", " ")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_text(row):
    """
    Version full text : on garde toute l'offre.
    La seule coupe éventuelle se fera ensuite au niveau du tokenizer
    si l'offre dépasse MAX_LENGTH tokens.
    """
    return minimal_clean(row.get(TEXT_COL, ""))


# LOAD DATA

df = pd.read_parquet(INPUT_PATH)

df = df.dropna(subset=LABELS).copy()

for label in LABELS:
    df[label] = df[label].astype(float).round().clip(0, 1).astype(int)

df["text"] = df.apply(build_text, axis=1)
df = df[df["text"].str.len() > 0].copy()

df = df.reset_index(drop=True)
df["offer_id"] = df.index.astype(int)

print(f"Nombre d'offres annotées utilisables : {len(df):,}")
print("\nPrévalence des labels :")
print(df[LABELS].mean().sort_values(ascending=False))


# TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.model_max_length = MAX_LENGTH


# TOKEN LENGTH DIAGNOSTIC

def count_tokens(text):
    return len(
        tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )


print("\nCalcul des longueurs en tokens sur les offres complètes...")
df["n_tokens"] = df["text"].apply(count_tokens)

print("\nStatistiques longueur en tokens, offres complètes :")
token_stats = df["n_tokens"].describe(
    percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
)
print(token_stats)

print("\nCouverture selon MAX_LENGTH :")
for limit in [512, 1024, 2048, 4096, 8192]:
    pct = (df["n_tokens"] <= limit).mean() * 100
    print(f"Offres <= {limit:>4} tokens : {pct:.1f}%")

n_truncated = int((df["n_tokens"] > MAX_LENGTH).sum())
pct_truncated = n_truncated / len(df) * 100

print(
    f"\nOffres qui seront tronquées à {MAX_LENGTH} tokens : "
    f"{n_truncated:,} / {len(df):,} ({pct_truncated:.2f}%)"
)


# SPLIT MULTI-LABEL

def multilabel_split(dataframe, test_size, seed):
    if HAS_ITERSTRAT:
        y = dataframe[LABELS].values.astype(int)
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(np.zeros(len(dataframe)), y))
        return dataframe.iloc[train_idx].copy(), dataframe.iloc[test_idx].copy()

    print(
        "\nWARNING: iterative-stratification non installé. "
        "Fallback vers split random non stratifié.\n"
        "Installe-le avec: pip install iterative-stratification\n"
    )
    return train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )


# 70 / 15 / 15 : train / valid / test
train_df, temp_df = multilabel_split(df, test_size=0.30, seed=SEED)
valid_df, test_df = multilabel_split(temp_df, test_size=0.50, seed=SEED + 1)

print("\nTailles splits :")
print(f"train: {len(train_df):,}")
print(f"valid: {len(valid_df):,}")
print(f"test : {len(test_df):,}")

print("\nPrévalences train :")
print(train_df[LABELS].mean().sort_values(ascending=False))

print("\nPrévalences valid :")
print(valid_df[LABELS].mean().sort_values(ascending=False))

print("\nPrévalences test :")
print(test_df[LABELS].mean().sort_values(ascending=False))


# SAVE SPLITS FOR ERROR ANALYSIS

os.makedirs(OUTPUT_DIR, exist_ok=True)

split_cols = ["offer_id", "text", "n_tokens"] + LABELS

train_df[split_cols].to_parquet(
    os.path.join(OUTPUT_DIR, "train_split.parquet"),
    index=False,
)
valid_df[split_cols].to_parquet(
    os.path.join(OUTPUT_DIR, "valid_split.parquet"),
    index=False,
)
test_df[split_cols].to_parquet(
    os.path.join(OUTPUT_DIR, "test_split.parquet"),
    index=False,
)


# DATASETS

train_ds = Dataset.from_pandas(
    train_df[["offer_id", "text", "n_tokens"] + LABELS],
    preserve_index=False,
)
valid_ds = Dataset.from_pandas(
    valid_df[["offer_id", "text", "n_tokens"] + LABELS],
    preserve_index=False,
)
test_ds = Dataset.from_pandas(
    test_df[["offer_id", "text", "n_tokens"] + LABELS],
    preserve_index=False,
)


# TOKENIZATION

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    labels = np.array(
        [[batch[label][i] for label in LABELS] for i in range(len(batch["text"]))],
        dtype=np.float32,
    )

    tokens["labels"] = labels

    # On garde ces infos pour debug éventuel.
    tokens["offer_id"] = batch["offer_id"]
    tokens["n_tokens_original"] = batch["n_tokens"]

    return tokens


remove_cols = ["text"] + LABELS

train_ds = train_ds.map(
    tokenize,
    batched=True,
    batch_size=TOKENIZE_BATCH_SIZE,
    remove_columns=remove_cols,
)

valid_ds = valid_ds.map(
    tokenize,
    batched=True,
    batch_size=TOKENIZE_BATCH_SIZE,
    remove_columns=remove_cols,
)

test_ds = test_ds.map(
    tokenize,
    batched=True,
    batch_size=TOKENIZE_BATCH_SIZE,
    remove_columns=remove_cols,
)

pad_to_multiple_of = 8 if torch.cuda.is_available() else None

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=pad_to_multiple_of,
)


# MODEL

id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
)

if USE_GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()


# OPTIONAL WEIGHTED LOSS

if USE_POS_WEIGHT:
    pos_counts = train_df[LABELS].sum(axis=0).values.astype(np.float32)
    neg_counts = len(train_df) - pos_counts
    pos_weight = neg_counts / np.maximum(pos_counts, 1.0)

    # Évite des poids absurdes si un label est très rare ou bruité.
    pos_weight = np.clip(pos_weight, 1.0, 10.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)

    print("\npos_weight utilisé :")
    print(dict(zip(LABELS, pos_weight.round(2))))
else:
    pos_weight_tensor = None


class WeightedMultilabelTrainer(Trainer):
    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")

        # Colonnes utiles pour debug / group_by_length,
        # mais pas acceptées par le forward du modèle.
        inputs.pop("offer_id", None)
        inputs.pop("n_tokens_original", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weight is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(logits.device)
            )
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# METRICS

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan


def safe_ap(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except ValueError:
        return np.nan


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(
            labels,
            preds,
            average="micro",
            zero_division=0,
        ),
        "recall_micro": recall_score(
            labels,
            preds,
            average="micro",
            zero_division=0,
        ),
    }

    aucs = []
    aps = []

    for i, label in enumerate(LABELS):
        label_auc = safe_auc(labels[:, i], probs[:, i])
        label_ap = safe_ap(labels[:, i], probs[:, i])

        aucs.append(label_auc)
        aps.append(label_ap)

        metrics[f"f1_{label}"] = f1_score(
            labels[:, i],
            preds[:, i],
            zero_division=0,
        )
        metrics[f"precision_{label}"] = precision_score(
            labels[:, i],
            preds[:, i],
            zero_division=0,
        )
        metrics[f"recall_{label}"] = recall_score(
            labels[:, i],
            preds[:, i],
            zero_division=0,
        )
        metrics[f"auc_{label}"] = label_auc
        metrics[f"ap_{label}"] = label_ap

    metrics["auc_macro"] = float(np.nanmean(aucs))
    metrics["ap_macro"] = float(np.nanmean(aps))

    return metrics


# THRESHOLD TUNING

def find_best_thresholds(labels, probs, min_precision=None):
    thresholds = {}

    for i, label in enumerate(LABELS):
        best_threshold = 0.5
        best_score = -1.0

        for threshold in np.arange(0.05, 0.96, 0.01):
            preds = (probs[:, i] >= threshold).astype(int)

            precision = precision_score(
                labels[:, i],
                preds,
                zero_division=0,
            )
            recall = recall_score(
                labels[:, i],
                preds,
                zero_division=0,
            )
            f1 = f1_score(
                labels[:, i],
                preds,
                zero_division=0,
            )

            if min_precision is not None:
                # Si tu veux limiter les faux positifs :
                # on maximise le recall sous contrainte de précision.
                if precision < min_precision:
                    continue
                score = recall
            else:
                score = f1

            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        thresholds[label] = best_threshold

    return thresholds


def evaluate_with_thresholds(labels, probs, thresholds):
    threshold_array = np.array([thresholds[label] for label in LABELS])
    preds = (probs >= threshold_array).astype(int)

    results = {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(
            labels,
            preds,
            average="micro",
            zero_division=0,
        ),
        "recall_micro": recall_score(
            labels,
            preds,
            average="micro",
            zero_division=0,
        ),
    }

    for i, label in enumerate(LABELS):
        results[f"f1_{label}"] = f1_score(
            labels[:, i],
            preds[:, i],
            zero_division=0,
        )
        results[f"precision_{label}"] = precision_score(
            labels[:, i],
            preds[:, i],
            zero_division=0,
        )
        results[f"recall_{label}"] = recall_score(
            labels[:, i],
            preds[:, i],
            zero_division=0,
        )

    return results


# TRAINING

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

steps_per_epoch = math.ceil(
    len(train_ds) / (PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
)
total_train_steps = steps_per_epoch * NUM_TRAIN_EPOCHS
warmup_steps = int(total_train_steps * WARMUP_RATIO)

print("\nTraining config :")
print(f"MAX_LENGTH: {MAX_LENGTH}")
print(f"per_device_train_batch_size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"gradient_accumulation_steps: {GRADIENT_ACCUMULATION_STEPS}")
print(
    "effective_train_batch_size: "
    f"{PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)
print(f"num_train_epochs: {NUM_TRAIN_EPOCHS}")
print(f"steps_per_epoch: {steps_per_epoch}")
print(f"total_train_steps: {total_train_steps}")
print(f"warmup_steps: {warmup_steps}")
print(f"bf16: {use_bf16}")
print(f"fp16: {use_fp16}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,

    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,

    logging_steps=25,

    load_best_model_at_end=True,
    metric_for_best_model="ap_macro",
    greater_is_better=True,
    save_total_limit=2,
    save_safetensors=True,

    seed=SEED,
    data_seed=SEED,

    bf16=use_bf16,
    fp16=use_fp16,

    group_by_length=True,

    eval_accumulation_steps=10,

    dataloader_num_workers=2,
    dataloader_pin_memory=torch.cuda.is_available(),

    remove_unused_columns=True,
    label_names=["labels"],

    report_to="none",
)

trainer = WeightedMultilabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    pos_weight=pos_weight_tensor,
)

trainer.train()


# VALIDATION: TUNE THRESHOLDS

valid_pred = trainer.predict(valid_ds)
valid_logits = valid_pred.predictions
valid_labels = valid_pred.label_ids
valid_probs = sigmoid(valid_logits)

thresholds = find_best_thresholds(
    valid_labels,
    valid_probs,
    min_precision=MIN_PRECISION_FOR_THRESHOLDS,
)

print("\nSeuils optimaux par label :")
for label, threshold in thresholds.items():
    print(f"{label}: {threshold:.2f}")

valid_threshold_metrics = evaluate_with_thresholds(
    valid_labels,
    valid_probs,
    thresholds,
)

print("\nMetrics validation avec seuils optimisés :")
for k, v in valid_threshold_metrics.items():
    print(f"{k}: {v:.4f}")


# TEST FINAL: NE PAS TUNER SUR CE SET

test_pred = trainer.predict(test_ds)
test_logits = test_pred.predictions
test_labels = test_pred.label_ids
test_probs = sigmoid(test_logits)

test_threshold_metrics = evaluate_with_thresholds(
    test_labels,
    test_probs,
    thresholds,
)

print("\nMetrics test avec seuils issus de la validation :")
for k, v in test_threshold_metrics.items():
    print(f"{k}: {v:.4f}")


# SAVE MODEL + ARTIFACTS

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
    json.dump(thresholds, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
    json.dump(LABELS, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "valid_metrics_thresholded.json"), "w", encoding="utf-8") as f:
    json.dump(valid_threshold_metrics, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "test_metrics_thresholded.json"), "w", encoding="utf-8") as f:
    json.dump(test_threshold_metrics, f, indent=2, ensure_ascii=False)

training_config = {
    "model_name": MODEL_NAME,
    "input_path": INPUT_PATH,
    "output_dir": OUTPUT_DIR,
    "text_col": TEXT_COL,
    "labels": LABELS,
    "max_length": MAX_LENGTH,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO,
    "warmup_steps": warmup_steps,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "gradient_checkpointing": USE_GRADIENT_CHECKPOINTING,
    "use_pos_weight": USE_POS_WEIGHT,
    "min_precision_for_thresholds": MIN_PRECISION_FOR_THRESHOLDS,
    "bf16": use_bf16,
    "fp16": use_fp16,
    "seed": SEED,
}

with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w", encoding="utf-8") as f:
    json.dump(training_config, f, indent=2, ensure_ascii=False)

token_length_summary = {
    "count": int(df["n_tokens"].count()),
    "mean": float(df["n_tokens"].mean()),
    "std": float(df["n_tokens"].std()),
    "min": int(df["n_tokens"].min()),
    "p50": float(df["n_tokens"].quantile(0.50)),
    "p75": float(df["n_tokens"].quantile(0.75)),
    "p90": float(df["n_tokens"].quantile(0.90)),
    "p95": float(df["n_tokens"].quantile(0.95)),
    "p99": float(df["n_tokens"].quantile(0.99)),
    "max": int(df["n_tokens"].max()),
    "max_length": MAX_LENGTH,
    "n_truncated": n_truncated,
    "pct_truncated": pct_truncated,
}

with open(os.path.join(OUTPUT_DIR, "token_length_summary.json"), "w", encoding="utf-8") as f:
    json.dump(token_length_summary, f, indent=2, ensure_ascii=False)

print(f"\nModèle sauvegardé dans : {OUTPUT_DIR}")
print(f"Seuils sauvegardés dans : {os.path.join(OUTPUT_DIR, 'thresholds.json')}")
print(f"Métriques test sauvegardées dans : {os.path.join(OUTPUT_DIR, 'test_metrics_thresholded.json')}")
print(f"Résumé tokens sauvegardé dans : {os.path.join(OUTPUT_DIR, 'token_length_summary.json')}")