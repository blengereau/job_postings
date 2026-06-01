import os
import re
import json
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


MODEL_NAME = "almanach/moderncamembert-base"
INPUT_PATH = "data/annotation/annoted_set.parquet"
OUTPUT_DIR = "models/moderncamembert_job_features"

TEXT_COL = "description"

LABELS = [
    "career_progression",
    "company_culture",
    "non_salary_benefits",
    "work_meaning_impact",
    "schedule_flexibility",
]

MAX_LENGTH = 1024
SEED = 42

USE_POS_WEIGHT = False
MIN_PRECISION_FOR_THRESHOLDS = None
# Mets par exemple 0.70 ou 0.80 si tu veux privilégier fortement la précision
# lors du passage sur les 800k offres.


# REPRODUCIBILITY

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# TEXT PREPARATION

def minimal_clean(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_text(row):
    desc = minimal_clean(row.get(TEXT_COL, ""))

    # Garde début + fin.
    # Début : mission, contexte, impact, culture.
    # Fin : avantages, télétravail, flexibilité, process, conditions.
    if len(desc) <= 4500:
        return desc

    return desc[:1800] + " [...] " + desc[-3200:]


# LOAD DATA

df = pd.read_parquet(INPUT_PATH)

df = df.dropna(subset=LABELS).copy()

for label in LABELS:
    df[label] = df[label].astype(float).round().clip(0, 1).astype(int)

df["text"] = df.apply(build_text, axis=1)
df = df[df["text"].str.len() > 0].copy()

print(f"Nombre d'offres annotées utilisables : {len(df):,}")
print("\nPrévalence des labels :")
print(df[LABELS].mean().sort_values(ascending=False))


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


# DATASETS

train_ds = Dataset.from_pandas(train_df[["text"] + LABELS], preserve_index=False)
valid_ds = Dataset.from_pandas(valid_df[["text"] + LABELS], preserve_index=False)
test_ds = Dataset.from_pandas(test_df[["text"] + LABELS], preserve_index=False)


# TOKENIZATION

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    labels = np.array(
        [[batch[label][i] for label in LABELS] for i in range(len(batch["text"]))],
        dtype=np.float32,
    )

    tokens["labels"] = labels
    return tokens


train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"] + LABELS)
valid_ds = valid_ds.map(tokenize, batched=True, remove_columns=["text"] + LABELS)
test_ds = test_ds.map(tokenize, batched=True, remove_columns=["text"] + LABELS)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
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
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }

    aucs = []
    aps = []

    for i, label in enumerate(LABELS):
        label_auc = safe_auc(labels[:, i], probs[:, i])
        label_ap = safe_ap(labels[:, i], probs[:, i])

        aucs.append(label_auc)
        aps.append(label_ap)

        metrics[f"f1_{label}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)
        metrics[f"precision_{label}"] = precision_score(labels[:, i], preds[:, i], zero_division=0)
        metrics[f"recall_{label}"] = recall_score(labels[:, i], preds[:, i], zero_division=0)
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

            precision = precision_score(labels[:, i], preds, zero_division=0)
            recall = recall_score(labels[:, i], preds, zero_division=0)
            f1 = f1_score(labels[:, i], preds, zero_division=0)

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
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }

    for i, label in enumerate(LABELS):
        results[f"f1_{label}"] = f1_score(labels[:, i], preds[:, i], zero_division=0)
        results[f"precision_{label}"] = precision_score(labels[:, i], preds[:, i], zero_division=0)
        results[f"recall_{label}"] = recall_score(labels[:, i], preds[:, i], zero_division=0)

    return results


# TRAINING

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,

    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,

    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="ap_macro",
    greater_is_better=True,
    save_total_limit=2,

    seed=SEED,
    data_seed=SEED,

    bf16=use_bf16,
    fp16=use_fp16,

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

valid_threshold_metrics = evaluate_with_thresholds(valid_labels, valid_probs, thresholds)

print("\nMetrics validation avec seuils optimisés :")
for k, v in valid_threshold_metrics.items():
    print(f"{k}: {v:.4f}")


# TEST FINAL: NE PAS TUNER SUR CE SET

test_pred = trainer.predict(test_ds)
test_logits = test_pred.predictions
test_labels = test_pred.label_ids
test_probs = sigmoid(test_logits)

test_threshold_metrics = evaluate_with_thresholds(test_labels, test_probs, thresholds)

print("\nMetrics test avec seuils issus de la validation :")
for k, v in test_threshold_metrics.items():
    print(f"{k}: {v:.4f}")


# SAVE

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
    json.dump(thresholds, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
    json.dump(LABELS, f, indent=2, ensure_ascii=False)

print(f"\nModèle sauvegardé dans : {OUTPUT_DIR}")
print(f"Seuils sauvegardés dans : {os.path.join(OUTPUT_DIR, 'thresholds.json')}")