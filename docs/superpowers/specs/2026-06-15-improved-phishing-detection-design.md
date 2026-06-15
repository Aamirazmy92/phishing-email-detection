# Improved Fraudulent Behaviour Prevention System — Design Spec

**Date:** 2026-06-15
**Project:** Final Year Project — NLP-based phishing email detection
**Location:** `D:\Uni\FINAL YEAR\Code\FYP Code`

## Goal

Improve the existing phishing email detection project with a focus on **academic
rigor**, while keeping everything simple enough to explain verbally to a tutor.
No deep learning. Uses the existing `CEAS_08.csv` dataset only.

## Problem with the current version

The current code (`enhanced_fraud_detection.py` / `testing.py`):

- Puts data loading, training, and the GUI in **one script** that retrains the
  model **every time** the app launches.
- Reports only a **single accuracy number** — no precision/recall/F1, no
  confusion matrix, no cross-validation.
- Re-implements text preprocessing inside the GUI separately from training,
  risking train/inference inconsistency.
- Requires a live **Gmail login** to demo, which is awkward and risky in a
  presentation.

## Target architecture

Split the project into three clear parts plus saved artifacts:

```
preprocessing.py      Shared text-cleaning, imported by BOTH training and GUI
train_evaluate.py     Trains & compares models, saves results + best model
gui_app.py            Loads the saved model, runs the Tkinter scanner
models/               best_model.joblib + vectorizer.joblib
results/              metrics.csv + charts (PNG) for the dissertation
```

**One-line justification:** "I separated model development from deployment, so
the app loads a trained model instead of retraining every launch."

## Components

### `preprocessing.py`
- Single `preprocess_text(text)` function (lowercase, URL/email masking,
  punctuation/digit removal, stopword removal).
- Imported by both `train_evaluate.py` and `gui_app.py` so training and
  inference use **identical** cleaning. Fixes a real correctness risk in the
  current code.

### `train_evaluate.py` (where the rigor lives)
1. Load `CEAS_08.csv`, build the `text` column, clean it via `preprocessing`.
2. **Stratified** train/test split (test_size=0.2) with a fixed `random_state`
   — reproducible and preserves the phishing/legit class ratio.
3. Fit a single `TfidfVectorizer` on the training set only (no test leakage).
4. Train **three simple, interpretable models** on the same features:
   - Multinomial Naïve Bayes
   - Logistic Regression
   - Linear SVM (`LinearSVC`)
5. For each model report **accuracy, precision, recall, F1-score** plus a
   **confusion matrix**.
6. Run **5-fold cross-validation** on each model and report mean ± std.
7. Save artifacts:
   - `results/metrics.csv` — comparison table.
   - `results/model_comparison.png` — bar chart of F1 across models.
   - `results/confusion_matrix_<model>.png` — heatmap per model.
   - `models/best_model.joblib` and `models/vectorizer.joblib` (best by F1).
8. Print a short summary to the console.

**Tutor-facing justifications:**
- Precision/recall/F1 over accuracy: "accuracy alone misleads on imbalanced
  classes; recall shows how many phishing emails I actually caught."
- Confusion matrix: "shows false positives — a legitimate email wrongly flagged
  — which matters for user trust."
- Cross-validation: "proves the score wasn't a lucky split."
- Model comparison: "I justified my model choice with evidence, not assumption."

### `gui_app.py`
- Loads `models/best_model.joblib` + `models/vectorizer.joblib` (fast startup,
  no retraining). If the model files are missing, shows a clear message telling
  the user to run `train_evaluate.py` first.
- Keeps the existing Tkinter Gmail IMAP scanner (login → scan up to 5 unread →
  classify each).
- Adds a **"Paste an email to test"** text box + button so the app can be demoed
  **without a Gmail login** — safer for a live presentation.
- Displays the predicted label for each email.

## Data flow

```
CEAS_08.csv
  -> preprocessing.preprocess_text
  -> TfidfVectorizer (fit on train only)
  -> train NB / LogReg / SVM
  -> evaluate (metrics + confusion matrix + CV)
  -> save best model + vectorizer + figures
        |
        v
gui_app.py loads model + vectorizer
  -> user pastes email OR scans Gmail inbox
  -> preprocessing.preprocess_text (same cleaning)
  -> vectorizer.transform -> model.predict
  -> show Phishing / Legitimate
```

## Error handling

- **Missing CSV** in `train_evaluate.py`: clear message naming the expected file
  and folder.
- **Missing `label` or text columns**: explicit `ValueError` describing what the
  CSV must contain.
- **Missing model files** in `gui_app.py`: message box telling the user to run
  training first (no crash).
- **Gmail login / empty inbox**: existing try/except message boxes retained.

## Testing / verification

- A small built-in sanity check in `train_evaluate.py`: after training, predict
  on 2–3 hand-written example emails (one obvious phishing, one obvious
  legitimate) and print the predictions to confirm the pipeline works
  end-to-end.
- Manual: run `train_evaluate.py`, confirm `results/` and `models/` are
  populated; run `gui_app.py`, paste a test email, confirm a prediction shows.

## Explicitly out of scope (to stay simple)

- No deep learning / BERT / transformers (despite being listed in
  `requirements.txt`).
- No new datasets — `CEAS_08.csv` only.
- No changes to the Gmail IMAP logic beyond loading the saved model.
- Suspicious-keyword highlighting / confidence scores (the CV's "XAI" claim) are
  noted as a possible later extension but are NOT part of this rigor-focused
  scope.

## Dependencies

Already installed: `pandas`, `scikit-learn`, `nltk`. Additionally used:
`matplotlib` and `seaborn` (for charts) and `joblib` (ships with scikit-learn).
`requirements.txt` will be trimmed to what is actually used.
