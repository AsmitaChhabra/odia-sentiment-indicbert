import os
import json
import numpy as np
import torch

from collections import Counter
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

MODEL_42 = "/mnt/ssd/nlp/Group1/odia_sentiment_checkpoints/seed_42/best_model_final"
MODEL_43 = "/mnt/ssd/nlp/Group1/odia_sentiment_checkpoints/seed_43/best_model_final"

OOD_FILE = "/tmp/or.json"

MAX_LENGTH = 128


# ============================================================
# LOAD OOD DATA
# ============================================================

print("=" * 70)
print("LOADING OOD DATASET")
print("=" * 70)

with open(OOD_FILE, "r", encoding="utf-8") as f:
    raw_data = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

print(f"Total rows: {len(raw_data)}")

# Remove rows with missing labels
data = [
    x for x in raw_data
    if x["LABEL"] in ["Positive", "Negative"]
]

print(f"Rows used for evaluation: {len(data)}")
print(f"Rows excluded because of missing label: {len(raw_data) - len(data)}")

label_counts = Counter(x["LABEL"] for x in data)
print("Label counts:", label_counts)


# ============================================================
# LABEL MAPPING
# SAME AS ORIGINAL TRAINING
#
# Original:
# 0 = Negative
# 1 = Positive
# ============================================================

label2id = {
    "Negative": 0,
    "Positive": 1
}

texts = [x["INDIC REVIEW"] for x in data]
labels = np.array([label2id[x["LABEL"]] for x in data])


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def get_predictions(model_path, model_name):

    print("\n" + "=" * 70)
    print(f"LOADING {model_name}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(device)
    model.eval()

    predictions = []
    probabilities = []

    batch_size = 64

    with torch.no_grad():

        for start in range(0, len(texts), batch_size):

            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            encoded = {
                k: v.to(device)
                for k, v in encoded.items()
            }

            outputs = model(**encoded)

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )

            preds = torch.argmax(
                probs,
                dim=-1
            )

            probabilities.append(
                probs.cpu().numpy()
            )

            predictions.append(
                preds.cpu().numpy()
            )

    probabilities = np.concatenate(probabilities)
    predictions = np.concatenate(predictions)

    # Free model memory before loading next model
    del model
    torch.cuda.empty_cache()

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    print(f"\n{model_name} results")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro-F1 : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            predictions,
            target_names=["Negative", "Positive"],
            digits=4
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            labels,
            predictions
        )
    )

    return probabilities, predictions


# ============================================================
# SEED 42
# ============================================================

probs_42, preds_42 = get_predictions(
    MODEL_42,
    "Seed 42"
)


# ============================================================
# SEED 43
# ============================================================

probs_43, preds_43 = get_predictions(
    MODEL_43,
    "Seed 43"
)


# ============================================================
# ENSEMBLE
# SAME METHOD AS ORIGINAL EXPERIMENT
#
# Average softmax probabilities
# ============================================================

print("\n" + "=" * 70)
print("ENSEMBLE RESULTS")
print("=" * 70)

ensemble_probs = (
    probs_42 + probs_43
) / 2.0

ensemble_preds = np.argmax(
    ensemble_probs,
    axis=1
)

ensemble_accuracy = accuracy_score(
    labels,
    ensemble_preds
)

ensemble_macro_f1 = f1_score(
    labels,
    ensemble_preds,
    average="macro"
)

print(f"Accuracy : {ensemble_accuracy:.4f}")
print(f"Macro-F1 : {ensemble_macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        labels,
        ensemble_preds,
        target_names=["Negative", "Positive"],
        digits=4
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        labels,
        ensemble_preds
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "dataset": "ai4bharat/IndicSentiment",
    "split": "test",
    "language": "Odia",
    "total_rows": len(raw_data),
    "evaluated_rows": len(data),
    "excluded_missing_label": len(raw_data) - len(data),

    "seed_42": {
        "accuracy": float(
            accuracy_score(labels, preds_42)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                preds_42,
                average="macro"
            )
        )
    },

    "seed_43": {
        "accuracy": float(
            accuracy_score(labels, preds_43)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                preds_43,
                average="macro"
            )
        )
    },

    "ensemble": {
        "accuracy": float(ensemble_accuracy),
        "macro_f1": float(ensemble_macro_f1)
    }
}

with open(
    "ood_indicsentiment_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2
    )

print("\nResults saved to:")
print("ood_indicsentiment_results.json")

print("\n" + "=" * 70)
print("OOD EVALUATION COMPLETE")
print("=" * 70)
