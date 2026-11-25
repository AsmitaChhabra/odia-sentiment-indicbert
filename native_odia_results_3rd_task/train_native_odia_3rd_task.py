"""
Native Odia sentiment -- fine-tune the ORIGINAL pretrained IndicBERTv2 on the
new Native Odia training data, then evaluate once on the untouched test set.

  * The model is loaded fresh from the Hugging Face hub id
    (ai4bharat/IndicBERTv2-MLM-only), never from an old fine-tuned checkpoint.
  * Only the Native Odia training file is used for training. No IndiSentiment140.
  * The validation split comes from the training file only (stratified).
  * Model selection (best epoch) uses validation macro-F1 only.
  * The test file is read and scored only after every seed has finished
    training. Test results never feed back into training or selection.

Run on the Nano server (check the GPU first, see README):

    cd ~/Group1/native_odia_sentiment_experiment
    nohup python3 train_native_odia.py \
        --train_file data/01_sentiment_groups_1_8_train.jsonl \
        --test_file  data/01_sentiment_groups_1_8_test.jsonl \
        > logs/train_native_odia.log 2>&1 &
    tail -f logs/train_native_odia.log
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

ORIGINAL_MODEL = "ai4bharat/IndicBERTv2-MLM-only"
LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Previous experiment, for comparison only: IndiSentiment140-trained 2-seed
# ensemble evaluated on this same 255-sample test set (evaluate_native_odia.py).
PREVIOUS_BENCHMARK_MODEL_ON_TEST = {"accuracy": 0.7765, "macro_f1": 0.7571}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_file", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--out_dir", default=os.path.expanduser(
        "~/Group1/native_odia_sentiment_experiment"))
    p.add_argument("--model_name", default=ORIGINAL_MODEL)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3,
                   help="early-stopping patience in epochs (0 = off)")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--min_free_gb", type=float, default=6.0,
                   help="refuse to start if the GPU has less free memory than this")
    p.add_argument("--group", default="Group 1")
    p.add_argument("--force_download", action="store_true",
                   help="re-download the pretrained weights from the hub instead of using the local HF cache")
    p.add_argument("--overwrite", action="store_true",
                   help="delete this script's previous checkpoints/results before running")
    # escape hatches, only for testing the pipeline
    p.add_argument("--allow_other_model", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--allow_cpu", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def log(*a):
    print(*a, flush=True)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path} line {n}: bad JSON ({e})")
    df = pd.DataFrame(rows)
    missing = {"id", "text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df[["id", "text", "label"]].copy()


def label_dist(df):
    c = Counter(df["label"].map(ID2LABEL))
    return {"total": int(len(df)), "positive": int(c["positive"]), "negative": int(c["negative"])}


def compute_metrics_from_preds(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": {ID2LABEL[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                    "f1": float(f[i]), "support": int(s[i])} for i in (0, 1)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def hf_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")}


def predict_probs(model_dir, texts, max_len, batch_size, device):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)


def main():
    args = parse_args()
    started = datetime.now()
    log(f"Started {started:%Y-%m-%d %H:%M:%S}")
    log("Arguments:", json.dumps(vars(args), indent=2))

    # ------------------------------------------------------------------
    # STEP 0 -- Guards: original model only, GPU free, clean output dirs
    # ------------------------------------------------------------------
    if args.model_name != ORIGINAL_MODEL and not args.allow_other_model:
        sys.exit(f"Refusing to run: model must be the original pretrained "
                 f"{ORIGINAL_MODEL}, got {args.model_name}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log(f"GPU: {torch.cuda.get_device_name(0)} | free {free/1e9:.1f} GB of {total/1e9:.1f} GB")
        try:
            log(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                "--format=csv"], capture_output=True, text=True).stdout)
        except FileNotFoundError:
            pass
        if free / 1e9 < args.min_free_gb:
            sys.exit(f"Refusing to start: only {free/1e9:.1f} GB GPU memory free "
                     f"(need {args.min_free_gb} GB). Another user's job is probably "
                     f"running -- wait for it rather than interrupting it.")
        use_fp16 = True
    elif args.allow_cpu:
        log("WARNING: no GPU, running on CPU (testing only).")
        use_fp16 = False
    else:
        sys.exit("No CUDA GPU detected. Stop and check before continuing.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_root = os.path.join(args.out_dir, "checkpoints")
    res_dir = os.path.join(args.out_dir, "results")
    for d in (ckpt_root, res_dir, os.path.join(args.out_dir, "logs")):
        os.makedirs(d, exist_ok=True)
    seed_dirs = {s: os.path.join(ckpt_root, f"native_seed_{s}") for s in args.seeds}
    existing = [d for d in seed_dirs.values() if os.path.exists(d) and os.listdir(d)]
    if existing:
        if not args.overwrite:
            sys.exit("Refusing to run: these checkpoint folders already exist "
                     f"(no resuming, every run starts fresh): {existing}. "
                     "Pass --overwrite to delete them and start again.")
        for d in existing:
            shutil.rmtree(d)

    # ------------------------------------------------------------------
    # STEP 1 -- Load and validate the Native Odia files
    # ------------------------------------------------------------------
    log("\n=== STEP 1: Loading Native Odia data ===")
    train_all = load_jsonl(args.train_file)
    test_df = load_jsonl(args.test_file)

    for name, df in (("train", train_all), ("test", test_df)):
        df["text"] = df["text"].astype(str).str.strip()
        bad = df["text"].eq("") | df["label"].isna()
        if bad.any():
            sys.exit(f"{name}: {bad.sum()} rows with empty text or label")
        unknown = set(df["label"].astype(str).str.strip().str.lower()) - set(LABEL2ID)
        if unknown:
            sys.exit(f"{name}: unknown labels {unknown}")
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL2ID).astype(int)
        if df["text"].duplicated().any() or df["id"].duplicated().any():
            sys.exit(f"{name}: duplicate texts or ids found")
    if set(train_all["text"]) & set(test_df["text"]):
        sys.exit("Overlapping texts between train and test")
    if set(train_all["id"]) & set(test_df["id"]):
        sys.exit("Overlapping ids between train and test")
    log("Checks passed: no missing values, no duplicates, no train/test overlap.")
    log("Train pool:", label_dist(train_all))
    log("Test      :", label_dist(test_df))

    # ------------------------------------------------------------------
    # STEP 2 -- Validation split from the training pool only
    # (fixed split_seed so every seed sees the same train/val split)
    # ------------------------------------------------------------------
    log("\n=== STEP 2: Train/validation split ===")
    # Group-aware: sentences split from the same YouTube comment share an id
    # prefix (e.g. yt-...-s000, yt-...-s001). Keep them on the same side so the
    # validation score is not inflated by sibling sentences.
    groups = train_all["id"].str.replace(r"-s\d+$", "", regex=True)
    sgkf = StratifiedGroupKFold(n_splits=max(2, round(1 / args.val_size)),
                                shuffle=True, random_state=args.split_seed)
    tr_idx, va_idx = next(sgkf.split(train_all, train_all["label"], groups))
    train_df = train_all.iloc[tr_idx].reset_index(drop=True)
    val_df = train_all.iloc[va_idx].reset_index(drop=True)
    assert not set(groups.iloc[tr_idx]) & set(groups.iloc[va_idx])
    log(f"Comment groups: {groups.nunique()} (sentences of one comment never split across train/val)")
    log("Train:", label_dist(train_df))
    log("Val  :", label_dist(val_df))
    for name, df in (("train_split", train_df), ("val_split", val_df)):
        out = df.copy()
        out["label"] = out["label"].map(ID2LABEL)
        out.to_csv(os.path.join(res_dir, f"{name}.csv"), index=False)

    # ------------------------------------------------------------------
    # STEP 3 -- Tokenise with the ORIGINAL tokenizer
    # ------------------------------------------------------------------
    log("\n=== STEP 3: Tokenising ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, force_download=args.force_download)

    def tok_fn(b):
        return tokenizer(b["text"], truncation=True, max_length=args.max_len)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tok_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tok_fn, batched=True)
    lens = [len(x) for x in train_ds["input_ids"]]
    truncated = sum(l >= args.max_len for l in lens)
    log(f"Token length: median {int(np.median(lens))}, max {max(lens)}, "
        f"truncated at {args.max_len}: {truncated}")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    warmup_steps = int(math.ceil(args.warmup_ratio * steps_per_epoch * args.epochs))

    # ------------------------------------------------------------------
    # STEP 4 -- Train each seed from fresh pretrained weights.
    # The test set is NOT touched here.
    # ------------------------------------------------------------------
    log("\n=== STEP 4: Training ===")
    seed_summaries = {}
    for seed in args.seeds:
        log(f"\n--- Seed {seed}: loading fresh pretrained weights from {args.model_name} ---")
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
            force_download=args.force_download and seed == args.seeds[0])

        targs = TrainingArguments(
            output_dir=seed_dirs[seed],
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_strategy="epoch",
            fp16=use_fp16,
            seed=seed,
            data_seed=seed,
            report_to="none",
        )
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.patience > 0 else []
        trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                          processing_class=tokenizer, data_collator=collator,
                          compute_metrics=hf_compute_metrics, callbacks=callbacks)
        trainer.train()  # never resumes: every run starts from pretrained weights

        best_dir = os.path.join(seed_dirs[seed], "best_model")
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)
        history = trainer.state.log_history
        with open(os.path.join(res_dir, f"training_history_seed_{seed}.json"), "w") as f:
            json.dump(history, f, indent=2)
        evals = [h for h in history if "eval_f1_macro" in h]
        best = max(evals, key=lambda h: h["eval_f1_macro"])
        seed_summaries[seed] = {"best_epoch": best["epoch"], "epochs_run": evals[-1]["epoch"],
                                "best_checkpoint": trainer.state.best_model_checkpoint}
        log(f"Seed {seed}: best val macro-F1 {best['eval_f1_macro']:.4f} at epoch {best['epoch']:.0f} "
            f"(ran {evals[-1]['epoch']:.0f} epochs). Saved to {best_dir}")

        # drop the intermediate checkpoint-* folders, keep only best_model
        for d in os.listdir(seed_dirs[seed]):
            if d.startswith("checkpoint-"):
                shutil.rmtree(os.path.join(seed_dirs[seed], d), ignore_errors=True)
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # STEP 5 -- Training is finished. Reload each saved best model and
    # score validation (for the report) and then, once, the test set.
    # ------------------------------------------------------------------
    log("\n=== STEP 5: Final evaluation (validation, then untouched test) ===")
    y_val, y_test = val_df["label"].to_numpy(), test_df["label"].to_numpy()
    val_probs, test_probs = {}, {}
    for seed in args.seeds:
        best_dir = os.path.join(seed_dirs[seed], "best_model")
        val_probs[seed] = predict_probs(best_dir, val_df["text"].tolist(), args.max_len,
                                        args.eval_batch_size, device)
        test_probs[seed] = predict_probs(best_dir, test_df["text"].tolist(), args.max_len,
                                         args.eval_batch_size, device)
        seed_summaries[seed]["val"] = compute_metrics_from_preds(y_val, val_probs[seed].argmax(1))
        seed_summaries[seed]["test"] = compute_metrics_from_preds(y_test, test_probs[seed].argmax(1))
        log(f"Seed {seed}: val acc {seed_summaries[seed]['val']['accuracy']:.4f} "
            f"F1 {seed_summaries[seed]['val']['macro_f1']:.4f} | "
            f"TEST acc {seed_summaries[seed]['test']['accuracy']:.4f} "
            f"F1 {seed_summaries[seed]['test']['macro_f1']:.4f}")

    ens_test_probs = np.mean([test_probs[s] for s in args.seeds], axis=0)
    ens_val_probs = np.mean([val_probs[s] for s in args.seeds], axis=0)
    ensemble = None
    if len(args.seeds) > 1:
        ensemble = {"val": compute_metrics_from_preds(y_val, ens_val_probs.argmax(1)),
                    "test": compute_metrics_from_preds(y_test, ens_test_probs.argmax(1))}
    test_accs = [seed_summaries[s]["test"]["accuracy"] for s in args.seeds]
    test_f1s = [seed_summaries[s]["test"]["macro_f1"] for s in args.seeds]
    final_name = "ensemble" if ensemble else f"seed {args.seeds[0]}"
    final = ensemble["test"] if ensemble else seed_summaries[args.seeds[0]]["test"]

    # ------------------------------------------------------------------
    # STEP 6 -- Save everything
    # ------------------------------------------------------------------
    results = {
        "group": args.group,
        "task": "Binary sentiment classification (positive/negative) of native Odia YouTube comments",
        "model": ORIGINAL_MODEL if args.model_name == ORIGINAL_MODEL else args.model_name,
        "train_file": os.path.abspath(args.train_file),
        "test_file": os.path.abspath(args.test_file),
        "settings": {k: getattr(args, k) for k in ("seeds", "split_seed", "val_size", "lr", "batch_size",
                                                   "epochs", "patience", "max_len", "weight_decay",
                                                   "warmup_ratio")} | {
            "warmup_steps": warmup_steps, "fp16": use_fp16,
            "model_selection": "best epoch by validation macro-F1"},
        "data": {"train_pool": label_dist(train_all), "train": label_dist(train_df),
                 "validation": label_dist(val_df), "test": label_dist(test_df),
                 "train_tokens_truncated": int(truncated)},
        "per_seed": {str(k): v for k, v in seed_summaries.items()},
        "per_seed_test_mean_std": {"accuracy_mean": float(np.mean(test_accs)),
                                   "accuracy_std": float(np.std(test_accs)),
                                   "macro_f1_mean": float(np.mean(test_f1s)),
                                   "macro_f1_std": float(np.std(test_f1s))},
        "ensemble": ensemble,
        "final_test": {"source": final_name, **final},
        "previous_benchmark_trained_model_on_same_test": PREVIOUS_BENCHMARK_MODEL_ON_TEST,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "versions": {"torch": torch.__version__,
                     "transformers": __import__("transformers").__version__},
    }
    with open(os.path.join(res_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    preds = test_df.copy()
    preds["label"] = preds["label"].map(ID2LABEL)
    for s in args.seeds:
        preds[f"prob_positive_seed_{s}"] = test_probs[s][:, 1]
    preds["prob_positive_final"] = ens_test_probs[:, 1]
    preds["predicted_label"] = np.where(ens_test_probs.argmax(1) == 1, "positive", "negative")
    preds["correct"] = preds["predicted_label"] == preds["label"]
    preds.to_csv(os.path.join(res_dir, "test_predictions.csv"), index=False, encoding="utf-8")

    final_pred = ens_test_probs.argmax(1)
    report = classification_report(y_test, final_pred, target_names=["negative", "positive"], digits=4)
    prev = PREVIOUS_BENCHMARK_MODEL_ON_TEST
    lines = [
        "NATIVE ODIA SENTIMENT -- FINE-TUNING ON NATIVE ODIA DATA",
        "=" * 60,
        f"Group          : {args.group}",
        f"Task           : {results['task']}",
        f"Model          : {results['model']} (fresh pretrained weights)",
        "",
        "Training settings",
        *[f"  {k:<16}: {v}" for k, v in results["settings"].items()],
        "",
        "Data (label counts)",
        *[f"  {k:<11}: {v['total']:>4} total | {v['positive']:>4} positive | {v['negative']:>4} negative"
          for k, v in results["data"].items() if isinstance(v, dict)],
        "",
        "Per-seed results",
        *[f"  seed {s}: best epoch {seed_summaries[s]['best_epoch']:.0f} | "
          f"val acc {seed_summaries[s]['val']['accuracy']:.4f} F1 {seed_summaries[s]['val']['macro_f1']:.4f} | "
          f"test acc {seed_summaries[s]['test']['accuracy']:.4f} F1 {seed_summaries[s]['test']['macro_f1']:.4f}"
          for s in args.seeds],
        f"  mean +/- std test acc {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}, "
        f"macro-F1 {np.mean(test_f1s):.4f} +/- {np.std(test_f1s):.4f}",
        "",
        f"FINAL TEST RESULT ({final_name}, {len(test_df)} samples)",
        f"  Accuracy    : {final['accuracy']:.4f}",
        f"  Macro-F1    : {final['macro_f1']:.4f}",
        f"  Weighted-F1 : {final['weighted_f1']:.4f}",
        f"  Confusion matrix [rows=true neg,pos; cols=pred neg,pos]: {final['confusion_matrix']}",
        "",
        report,
        "Comparison on the same test set",
        f"  Previous: IndiSentiment140-trained ensemble : acc {prev['accuracy']:.4f} | macro-F1 {prev['macro_f1']:.4f}",
        f"  Now     : Native-Odia-trained {final_name:<10}: acc {final['accuracy']:.4f} | macro-F1 {final['macro_f1']:.4f}",
    ]
    with open(os.path.join(res_dir, "final_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n" + "\n".join(lines))
    log(f"\nSaved to {res_dir}: metrics.json, final_report.txt, test_predictions.csv, "
        f"train_split.csv, val_split.csv, training_history_seed_*.json")
    log("DONE.")


if __name__ == "__main__":
    main()
