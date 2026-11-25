# Odia Sentiment Classification using IndicBERTv2

Binary sentiment classification (positive / negative) of Odia-script text using
AI4Bharat's `ai4bharat/IndicBERTv2-MLM-only`, by **Group 1** (DTSC422 project).

The project has three stages:

1. **Benchmark fine-tuning:** fine-tune IndicBERTv2 on the Odia-script subset of
   IndiSentiment140 (Sentiment140 tweets translated into Odia), with a two-seed ensemble.
2. **Out-of-domain evaluation:** test the benchmark-trained model on
   native Odia YouTube comments and on AI4Bharat's IndicSentiment (Odia).
3. **Native Odia fine-tuning:** fine-tune a *fresh* copy of the pretrained
   IndicBERTv2 on the Native Odia training file only, and evaluate it once on
   the unchanged Native Odia test set.

## Results at a glance

| Model (training data) | Evaluated on | Accuracy | Macro-F1 |
|---|---|---|---|
| Stage 1: IndiSentiment140 ensemble | IndiSentiment140 test (152,116) | 0.8223 | 0.8223 |
| Stage 1: IndiSentiment140 ensemble | IndicSentiment Odia test (998) | 0.8948 | 0.8943 |
| Stage 1: IndiSentiment140 ensemble | **Native Odia test (255)** | 0.7765 | 0.7571 |
| **Stage 3: Native Odia ensemble** | **Native Odia test (255)** | **0.9216** | **0.9195** |

Training on native Odia comments improves accuracy on the same 255-sample test
set by about 14.5 percentage points over the benchmark-trained model.

## Repository structure

| File | Stage | Purpose |
|---|---|---|
| `train_1.py` | 1 | Fine-tune on IndiSentiment140 (two seeds + ensemble) |
| `final_results.txt` | 1 | Stage 1 report |
| `evaluate_native_odia.py` | 2 | Evaluate Stage 1 models on the Native Odia test set |
| `native_odia_evaluation_results.csv`, `native_odia_final.log` | 2 | Native Odia evaluation results |
| `evaluate_ood_indicsentiment.py` | 2 | Evaluate Stage 1 models on IndicSentiment (Odia) |
| `ood_indicsentiment_results.json`, `ood_run.log` | 2 | IndicSentiment results |
| `train_native_odia_3rd_task.py` | 3 | Fine-tune fresh IndicBERTv2 on Native Odia data |
| `native_odia_results/` | 3 | Stage 3 report, metrics, predictions and splits |

---

## Stage 1: Benchmark fine-tuning on IndiSentiment140

The model is fine-tuned on an Odia-script subset of the IndiSentiment140 dataset.
Two models are trained independently with random seeds 42 and 43, and their
softmax probabilities on the held-out test set are averaged to produce the
ensemble prediction.

The training pipeline includes:

- Odia-script filtering
- Exact duplicate removal before splitting
- HTML/entity artifact cleaning
- Stratified train/validation/test splitting
- FP16 mixed-precision training
- Early stopping
- Two-seed ensemble evaluation
- Confusion matrix and classification report
- Sample misclassified examples

### Model configuration

| Parameter | Value |
|---|---|
| Model | `ai4bharat/IndicBERTv2-MLM-only` |
| Maximum sequence length | 128 |
| Number of labels | 2 |
| Learning rate | 3e-5 |
| Weight decay | 0.01 |
| Training batch size | 32 |
| Evaluation batch size | 64 |
| Maximum epochs | 5 |
| Early stopping patience | 5 |
| Precision | FP16 |
| Random seeds | 42, 43 |
| Split seed | 42 |

### Dataset

The experiment uses the **IndiSentiment140** dataset:
https://huggingface.co/datasets/saurabh1003/IndiSentiment140

The training data is filtered to keep text in which more than 30% of letters
are Odia Unicode-script characters. The Odia Unicode range used is:

```text
U+0B00 – U+0B7F
```

| Step | Rows |
|---|---|
| Rows detected as Odia script | 1,587,337 |
| Exact duplicates removed before splitting | 66,180 |
| Train / validation / test (80 / 10 / 10, stratified) | 1,216,925 / 152,116 / 152,116 |

### Results

| Model | Test accuracy |
|---|---|
| Seed 42 | 0.8161 |
| Seed 43 | 0.8201 |
| **Ensemble** | **0.8223** (macro-F1 0.8223) |

Full report: `final_results.txt`.

---

## Stage 2: Out-of-domain evaluation

The Stage 1 models are applied, without further training, to two out-of-domain
datasets: IndicSentiment (human-translated Odia product reviews) and native
Odia YouTube comments annotated by Group 1.

| Dataset | Rows | Seed 42 | Seed 43 | Ensemble accuracy | Ensemble macro-F1 |
|---|---|---|---|---|---|
| IndicSentiment, Odia test (`ai4bharat/IndicSentiment`) | 998 | 0.8878 | 0.8868 | 0.8948 | 0.8943 |
| Native Odia test (YouTube comments) | 255 | 0.7843 | 0.7686 | 0.7765 | 0.7571 |

On the native Odia YouTube comments, accuracy is 4.6 percentage points lower than
on the IndiSentiment140 benchmark test set. The main error is negative comments
predicted as positive (44 of 107 negatives).

---

## Stage 3: Fine-tuning on Native Odia data

**Script:** `train_native_odia_3rd_task.py`

Following the Native Odia training protocol:

- The original pretrained `ai4bharat/IndicBERTv2-MLM-only` is loaded fresh from
  the Hugging Face hub for each seed. No earlier fine-tuned or saved model is used.
- Only the Native Odia training file is used for training. No benchmark data is mixed in.
- The validation split is created from the training file only.
- Early stopping and best-epoch selection use validation macro-F1 only.
  No test labels are used for model selection, early stopping or threshold tuning.
- The Native Odia test file is unchanged and is scored only after all training is complete.

### Data

Both files are native Odia YouTube comments annotated by Group 1.

| Split | Total | Positive | Negative |
|---|---|---|---|
| Training file (pool) | 887 | 473 | 414 |
| → Train | 764 | 418 | 346 |
| → Validation | 123 | 55 | 68 |
| Test (unchanged) | 255 | 148 | 107 |

Checks run before training: no missing text or labels, no duplicate texts or
IDs within either file, and no overlapping texts or IDs between training and
test.

The validation split is stratified by label and grouped by comment: some
comments were split into several sentences (IDs ending `-s000`, `-s001`, …),
and all sentences of one comment are kept on the same side of the split.

### Training settings

| Parameter | Value |
|---|---|
| Model | `ai4bharat/IndicBERTv2-MLM-only` (fresh pretrained weights) |
| Maximum sequence length | 128 (no training text was truncated; longest is 95 tokens) |
| Learning rate | 3e-5 |
| Weight decay | 0.01 |
| Warmup | 10% of steps (49 steps) |
| Training batch size | 16 |
| Evaluation batch size | 64 |
| Maximum epochs | 10 |
| Early stopping patience | 3 epochs |
| Model selection | Best epoch by validation macro-F1 |
| Precision | FP16 |
| Random seeds | 42, 43 |
| Split seed | 42 |

Batch size and epoch count were adjusted from Stage 1 for the much smaller
dataset: batch 32 for 5 epochs would give only about 120 update steps.

### Results

| Model | Best epoch | Val accuracy | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|
| Seed 42 | 2 | 0.8455 | 0.8429 | 0.8980 | 0.8969 |
| Seed 43 | 5 | 0.8618 | 0.8612 | 0.9216 | 0.9195 |
| **Ensemble** | — | — | — | **0.9216** | **0.9195** |

Mean ± std across seeds on the test set: accuracy 0.9098 ± 0.0118, macro-F1
0.9082 ± 0.0113.

**Ensemble on the Native Odia test set (255 samples)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.9065 | 0.9065 | 0.9065 | 107 |
| Positive | 0.9324 | 0.9324 | 0.9324 | 148 |

| | Predicted negative | Predicted positive |
|---|---|---|
| **True negative** | 97 | 10 |
| **True positive** | 10 | 138 |

Compared with the Stage 1 benchmark-trained ensemble on the same test set,
accuracy rises from 0.7765 to 0.9216 and macro-F1 from 0.7571 to 0.9195.
Negatives recognised correctly rise from 63 to 97 of 107.

**Note on test vs. validation scores.** Test scores are higher than validation
scores. The validation set is small (123 samples), and the supplied test set
shares source videos with the training data (85 of 98 test videos also appear
in training), although no texts overlap. The test score may therefore be
slightly optimistic for comments on unseen topics.

### Outputs (`native_odia_results/`)

| File | Contents |
|---|---|
| `final_report.txt` | Readable report of settings, data counts and metrics |
| `metrics.json` | All settings and metrics in machine-readable form |
| `test_predictions.csv` | Per-sample test predictions and probabilities for each seed |
| `train_split.csv`, `val_split.csv` | Exact train/validation split with IDs |
| `training_history_seed_*.json` | Per-epoch training and validation logs |

### How to run

```bash
cd ~/Group1/native_odia_sentiment_experiment
nvidia-smi        # check the GPU is free enough; the script needs at least 6 GB
nohup python3 train_native_odia_3rd_task.py \
  --train_file data/01_sentiment_groups_1_8_train.jsonl \
  --test_file  data/01_sentiment_groups_1_8_test.jsonl \
  --out_dir    "$PWD" \
  > logs/train_native_odia_3rd_task.log 2>&1 &
tail -f logs/train_native_odia_3rd_task.log
```

The script refuses to run with any model other than the original pretrained
checkpoint, never resumes from earlier checkpoints, and will not overwrite an
earlier run unless `--overwrite` is given. Add `--force_download` to
re-download the pretrained weights instead of using the local cache.

Requirements: Python 3, PyTorch with CUDA, `transformers` (run with 5.17.0),
`datasets`, `scikit-learn`, `pandas`, `numpy`.
