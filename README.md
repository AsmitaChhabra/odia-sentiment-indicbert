# Odia Sentiment Classification using IndicBERTv2

Binary sentiment classification of Odia-script text using AI4Bharat's
IndicBERTv2 model and a two-seed ensemble approach.

## Overview

This project fine-tunes the multilingual IndicBERTv2 model for sentiment
classification on an Odia-script subset of the IndiSentiment140 dataset.

The experiment uses two independently trained models with random seeds 42 and
43. Their softmax probabilities on the held-out test set are averaged to
produce the final ensemble prediction.

The training pipeline also includes:

- Odia-script filtering
- Exact duplicate removal before splitting
- HTML/entity artifact cleaning
- Stratified train/validation/test splitting
- FP16 mixed-precision training
- Early stopping
- Two-seed ensemble evaluation
- Confusion matrix and classification report
- Sample misclassified examples

---

## Model

**Pre-trained model:** `ai4bharat/IndicBERTv2-MLM-only`

IndicBERTv2 is a multilingual model designed for Indian languages. The model
is fine-tuned here for binary sentiment classification.

### Model Configuration

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

---

## Dataset

The experiment uses the **IndiSentiment140** dataset:

https://huggingface.co/datasets/saurabh1003/IndiSentiment140

The original training data is filtered to identify text containing a
sufficient proportion of Odia Unicode-script characters.

The Odia Unicode range used by the script is:

```text
U+0B00 – U+0B7F
