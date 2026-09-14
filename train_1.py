"""
Odia Sentiment -- MASTER SCRIPT (overnight run)
Fixes applied: dedup before split, HTML-entity cleanup, tighter early stopping,
2-seed ensemble for the extra accuracy lever, full evaluation at the end.

Run with nohup so it survives your SSH session disconnecting:
    nohup python3 odia_sentiment_master.py > run.log 2>&1 &

Detach safely and check progress anytime with:
    tail -f run.log

In the morning, read final_report.txt for the full writeup-ready summary.

NOTE ON RUNTIME: this trains 2 full seeds sequentially, up to 5 epochs each
(early stopping usually cuts this short). Check the printed row count early
in run.log -- if it still looks too slow, you can kill it (see bottom of
file) and drop to a single seed instead.
"""

import os
import re
import html
import json
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
CHECKPOINT_ROOT = os.path.join(os.getcwd(), "odia_sentiment_checkpoints")
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

SEEDS = [42, 43]               # 2-seed ensemble (was [42, 43, 44])
SPLIT_SEED = 42                # split itself stays fixed across seeds so both models see the same test set
NUM_PROC = 10                   # shared-server courtesy cap, unchanged from original
MAX_LEN = 128

print("CUDA available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0), flush=True)
else:
    print("WARNING: no CUDA GPU detected -- stop and check before continuing.", flush=True)

# =====================================================================
# STEP 1 -- Load, filter to Odia script (unchanged heuristic -- dataset
# and filter approach are given constraints, not something we're touching)
# =====================================================================
print("\n=== STEP 1: Loading + filtering ===", flush=True)
ds = load_dataset("saurabh1003/IndiSentiment140")

TEXT_COL = "text"
LABEL_COL = "sentiment"
LABEL2ID = {0: 0, 4: 1}
NUM_LABELS = len(LABEL2ID)

ODIA_PATTERN = re.compile(r"[\u0B00-\u0B7F]")

def is_odia_text(text, min_ratio=0.3):
    if not text:
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    odia_count = len(ODIA_PATTERN.findall(text))
    return odia_count / len(letters) > min_ratio

odia_ds = ds["train"].filter(lambda ex: is_odia_text(ex[TEXT_COL]), num_proc=NUM_PROC)
print("Rows detected as Odia script:", len(odia_ds), flush=True)

df = odia_ds.to_pandas()[[TEXT_COL, LABEL_COL]].rename(columns={TEXT_COL: "text", LABEL_COL: "label"})
df["label"] = df["label"].map(LABEL2ID)
df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)
print("Before dedup:", len(df), flush=True)

# =====================================================================
# STEP 2 -- DEDUP (the correctness bug fix -- must happen before split,
# or duplicate rows can leak across train/val/test)
# =====================================================================
print("\n=== STEP 2: Deduplication ===", flush=True)
dupes_removed = df.duplicated(subset=["text"]).sum()
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Removed {dupes_removed} exact-duplicate rows.", flush=True)
print("After dedup:", len(df), flush=True)
print("Class balance after dedup:\n", df["label"].value_counts(normalize=True), flush=True)

# =====================================================================
# STEP 3 -- CLEAN HTML-ENTITY ARTIFACTS (confirmed in your sample dump:
# "amp" and "Quot" are literal leftover &amp; / &quot; that survived
# translation as garbage tokens instead of being decoded)
# =====================================================================
print("\n=== STEP 3: Cleaning text ===", flush=True)

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'\bamp\b', '&', text)
    text = re.sub(r'\bQuot\b', '"', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text"] = df["text"].apply(clean_text)
df = df[df["text"].str.len() > 0].reset_index(drop=True)
print("Rows after cleaning (non-empty):", len(df), flush=True)

# =====================================================================
# STEP 4 -- Split ONCE, shared across all seeds (so the ensemble members
# are evaluated on the identical held-out test set -- otherwise the
# "ensemble" comparison is meaningless)
# =====================================================================
print("\n=== STEP 4: Splitting ===", flush=True)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SPLIT_SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SPLIT_SEED)
print("train / val / test sizes:", len(train_df), len(val_df), len(test_df), flush=True)

# =====================================================================
# STEP 5 -- Tokenize ONCE, reused for every seed (saves real time --
# tokenizing the full set multiple times for no reason would waste hours
# overnight)
# =====================================================================
print("\n=== STEP 5: Tokenizing (shared across seeds) ===", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True)).map(tokenize_fn, batched=True, num_proc=NUM_PROC)
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True)).map(tokenize_fn, batched=True, num_proc=NUM_PROC)
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True)).map(tokenize_fn, batched=True, num_proc=NUM_PROC)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# =====================================================================
# STEP 6 -- Train each seed. Hyperparameter fixes vs. the original run:
#   - LR 2e-5 (was 1e-5, likely under-fitting in 3 epochs)
#   - up to 5 epochs, but real EarlyStoppingCallback with patience=5
#     (the original script trained all 3 epochs regardless --
#     load_best_model_at_end only picks the best checkpoint, it does
#     NOT stop training early)
#   - eval_steps 500 (was 2000 -- too coarse to catch the actual best point)
# =====================================================================
print("\n=== STEP 6: Training seeds ===", flush=True)

seed_test_probs = []   # softmax probs per seed, for ensembling
seed_test_accs = []

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---", flush=True)
    set_seed(seed)
    seed_dir = os.path.join(CHECKPOINT_ROOT, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    args = TrainingArguments(
        output_dir=seed_dir,
        learning_rate=3e-5, # 3e-5
        weight_decay=0.01,
        warmup_steps=0.1,       # v5: warmup_ratio was removed; warmup_steps now accepts a float <1 as a ratio
        fp16=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=5, 
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=seed,
        report_to="none",
        logging_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # resume support: if this seed's run was interrupted, pick up from its
    # own last checkpoint rather than restarting from scratch
    last_checkpoint = None
    existing = [d for d in os.listdir(seed_dir) if d.startswith("checkpoint-")]
    if existing:
        last_checkpoint = os.path.join(seed_dir, sorted(existing, key=lambda x: int(x.split("-")[1]))[-1])
        print("Resuming seed", seed, "from checkpoint:", last_checkpoint, flush=True)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # evaluate this seed alone
    single_result = trainer.evaluate(test_ds)
    print(f"Seed {seed} solo test accuracy:", single_result["eval_accuracy"], flush=True)
    seed_test_accs.append(single_result["eval_accuracy"])

    # get raw logits for ensembling
    preds = trainer.predict(test_ds)
    probs = torch.softmax(torch.tensor(preds.predictions), dim=-1).numpy()
    seed_test_probs.append(probs)

    best_model_dir = os.path.join(seed_dir, "best_model_final")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Seed {seed} best model saved to:", best_model_dir, flush=True)

    # free GPU memory before the next seed
    del trainer, model
    torch.cuda.empty_cache()

# =====================================================================
# STEP 7 -- ENSEMBLE: average softmax probabilities across seeds
# (the extra lever available when the model architecture itself is fixed)
# =====================================================================
print("\n=== STEP 7: Ensembling ===", flush=True)
y_true = np.array(test_df["label"].tolist())
avg_probs = np.mean(seed_test_probs, axis=0)
y_pred_ensemble = np.argmax(avg_probs, axis=-1)

ensemble_acc = accuracy_score(y_true, y_pred_ensemble)
ensemble_f1 = f1_score(y_true, y_pred_ensemble, average="macro")

print("Per-seed solo accuracies:", dict(zip(SEEDS, seed_test_accs)), flush=True)
print("ENSEMBLE test accuracy:", ensemble_acc, flush=True)
print("ENSEMBLE f1_macro:", ensemble_f1, flush=True)

# =====================================================================
# STEP 8 -- Full evaluation: confusion matrix, per-class report, and
# concrete misclassified examples for you to eyeball in the morning
# =====================================================================
print("\n=== STEP 8: Error analysis ===", flush=True)
cm = confusion_matrix(y_true, y_pred_ensemble)
report = classification_report(y_true, y_pred_ensemble, digits=3)
print("Confusion matrix:\n", cm, flush=True)
print(report, flush=True)

test_texts = test_df["text"].tolist()
wrong_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred_ensemble[i]][:30]
misclassified_lines = []
for i in wrong_idx:
    line = f"TRUE={y_true[i]} PRED={y_pred_ensemble[i]} TEXT={test_texts[i]}"
    misclassified_lines.append(line)
    print(line, flush=True)

# =====================================================================
# STEP 9 -- Write a single morning-readable report file
# =====================================================================
report_path = os.path.join(os.getcwd(), "final_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("ODIA SENTIMENT -- OVERNIGHT RUN REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Original baseline accuracy (previous script): 0.82\n")
    f.write(f"Rows after Odia filter: {len(odia_ds)}\n")
    f.write(f"Exact duplicates removed before split: {dupes_removed}\n")
    f.write(f"Train / val / test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}\n\n")
    f.write(f"Per-seed solo test accuracy: {dict(zip(SEEDS, seed_test_accs))}\n")
    f.write(f"ENSEMBLE test accuracy: {ensemble_acc:.4f}\n")
    f.write(f"ENSEMBLE f1_macro: {ensemble_f1:.4f}\n\n")
    f.write("Confusion matrix:\n" + str(cm) + "\n\n")
    f.write("Classification report:\n" + report + "\n\n")
    f.write("Sample misclassified examples:\n")
    f.write("\n".join(misclassified_lines) + "\n")

print(f"\nFull report written to {report_path}", flush=True)
print("DONE.", flush=True)

# -----------------------------------------------------------------
# If you need to kill an overnight run from another SSH session:
#   pgrep -f odia_sentiment_master.py
#   kill <pid>
# -----------------------------------------------------------------
