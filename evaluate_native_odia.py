import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.getcwd()

NATIVE_PATH = os.path.join(
    BASE_DIR,
    "01_sentiment_groups_1_8(2).csv"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "odia_sentiment_checkpoints"
)

SEEDS = [42, 43]
MAX_LEN = 128

# Professor dataset labels
NATIVE_LABEL_MAP = {
    "negative": 0,
    "positive": 1
}

# Final benchmark from the actual training run
BENCHMARK_ACC = 0.8222672171237739


# ============================================================
# STEP 1 -- LOAD NATIVE ODIA TEST SET
# ============================================================

print("\n=== STEP 1: Loading native Odia test set ===")

native_df = pd.read_csv(NATIVE_PATH)

print("Columns:", native_df.columns.tolist())
print("Number of rows:", len(native_df))
print("\nLabel counts:")
print(native_df["label"].value_counts())

print("\nFirst 3 rows:")
print(native_df[["text", "label"]].head(3))


# ============================================================
# STEP 2 -- PREPARE LABELS
# ============================================================

native_df = native_df[["text", "label"]].copy()

native_df["label"] = (
    native_df["label"]
    .astype(str)
    .str.strip()
    .map(NATIVE_LABEL_MAP)
)

if native_df["label"].isna().any():
    print("\nERROR: Unknown labels found:")
    print(native_df[native_df["label"].isna()])
    raise ValueError("Some labels could not be mapped.")

native_df["label"] = native_df["label"].astype(int)

print("\nFinal label counts:")
print(native_df["label"].value_counts().sort_index())


# ============================================================
# STEP 3 -- TOKENIZER
# ============================================================

print("\n=== STEP 2: Loading tokenizer ===")

# Use tokenizer from seed 42 checkpoint.
# Both checkpoints were trained from the same base tokenizer.
reference_model_dir = os.path.join(
    CHECKPOINT_DIR,
    "seed_42",
    "best_model_final"
)

tokenizer = AutoTokenizer.from_pretrained(reference_model_dir)


def tokenize_texts(texts):
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=True,
        return_tensors="pt"
    )


# ============================================================
# STEP 4 -- RUN BOTH SEED MODELS
# ============================================================

seed_probabilities = []

texts = native_df["text"].tolist()

for seed in SEEDS:

    model_dir = os.path.join(
        CHECKPOINT_DIR,
        f"seed_{seed}",
        "best_model_final"
    )

    print(f"\n=== Evaluating seed {seed} ===")
    print("Model:", model_dir)

    tokenizer_seed = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    all_probs = []

    batch_size = 64

    for start in range(0, len(texts), batch_size):

        batch_texts = texts[start:start + batch_size]

        inputs = tokenizer_seed(
            batch_texts,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(**inputs)

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )

        all_probs.append(
            probs.cpu().numpy()
        )

    seed_probs = np.concatenate(
        all_probs,
        axis=0
    )

    seed_probabilities.append(seed_probs)

    seed_predictions = np.argmax(
        seed_probs,
        axis=1
    )

    seed_acc = accuracy_score(
        native_df["label"],
        seed_predictions
    )

    seed_f1 = f1_score(
        native_df["label"],
        seed_predictions,
        average="macro"
    )

    print(f"Seed {seed} accuracy: {seed_acc:.4f}")
    print(f"Seed {seed} macro-F1: {seed_f1:.4f}")

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# STEP 5 -- TWO-SEED ENSEMBLE
# ============================================================

print("\n=== STEP 3: Two-seed ensemble ===")

# EXACT SAME PRINCIPLE AS TRAINING SCRIPT:
# average softmax probabilities across seed 42 and seed 43

avg_probs = np.mean(
    seed_probabilities,
    axis=0
)

ensemble_predictions = np.argmax(
    avg_probs,
    axis=1
)

y_true = native_df["label"].to_numpy()

native_acc = accuracy_score(
    y_true,
    ensemble_predictions
)

native_f1 = f1_score(
    y_true,
    ensemble_predictions,
    average="macro"
)

cm = confusion_matrix(
    y_true,
    ensemble_predictions
)


# ============================================================
# STEP 6 -- RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL NATIVE ODIA RESULTS")
print("=" * 60)

print(f"Native Odia accuracy : {native_acc:.4f}")
print(f"Native Odia macro-F1 : {native_f1:.4f}")

print("\nConfusion matrix:")
print(cm)

print("\nClassification report:")
print(
    classification_report(
        y_true,
        ensemble_predictions,
        target_names=["Negative", "Positive"],
        digits=4
    )
)


# ============================================================
# STEP 7 -- TRANSLATIONESE GAP
# ============================================================

translationese_gap = BENCHMARK_ACC - native_acc

print("\n" + "=" * 60)
print("TRANSLATIONESE GAP")
print("=" * 60)

print(f"Benchmark accuracy   : {BENCHMARK_ACC:.4f}")
print(f"Native Odia accuracy : {native_acc:.4f}")
print(
    f"Translationese gap   : {translationese_gap:.4f}"
)

print(
    f"Translationese gap   : "
    f"{translationese_gap * 100:.2f} percentage points"
)


# ============================================================
# STEP 8 -- SAVE RESULTS
# ============================================================

results = {
    "n_rows": len(native_df),
    "benchmark_accuracy": BENCHMARK_ACC,
    "native_accuracy": native_acc,
    "native_macro_f1": native_f1,
    "translationese_gap": translationese_gap,
    "translationese_gap_percentage_points":
        translationese_gap * 100
}

results_df = pd.DataFrame([results])

results_df.to_csv(
    "native_odia_evaluation_results.csv",
    index=False
)

# Save predictions too
output_df = native_df.copy()

output_df["predicted_label"] = np.where(
    ensemble_predictions == 1,
    "Positive",
    "Negative"
)

output_df["prob_negative"] = avg_probs[:, 0]
output_df["prob_positive"] = avg_probs[:, 1]

output_df.to_csv(
    "native_odia_predictions.csv",
    index=False
)

print("\nSaved:")
print("  native_odia_evaluation_results.csv")
print("  native_odia_predictions.csv")

print("\nDONE.")
