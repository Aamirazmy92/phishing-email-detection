# Fraudulent Behaviour Prevention System (NLP + ML)

Phishing email detector. Compares three interpretable machine-learning models
and runs a Tkinter scanner that loads the best one.

## Setup

```
py -m pip install -r requirements.txt
```

## Usage

1. Train and evaluate (creates `models/` and `results/`):

   ```
   py train_evaluate.py
   ```

2. Run the scanner:

   ```
   py gui_app.py
   ```

   Paste an email to test offline, or log in with a Gmail **app password** to
   scan unread inbox messages.

## How it works

1. `train_evaluate.py` loads `CEAS_08.csv`, cleans the text, and splits it into
   train/test sets (stratified, fixed seed for reproducibility).
2. It trains and compares **Naive Bayes**, **Logistic Regression** and
   **Linear SVM** on TF-IDF features, reporting accuracy, precision, recall and
   F1 plus 5-fold cross-validation.
3. The best model (by F1) and the TF-IDF vectorizer are saved to `models/`.
4. `gui_app.py` loads that saved model so the app starts instantly instead of
   retraining each launch.

## Files

- `preprocessing.py` — shared text cleaning (used by training and GUI)
- `data_loader.py` — loads/validates `CEAS_08.csv`
- `evaluation.py` — model registry + metric calculation
- `train_evaluate.py` — trains the three models, saves metrics, charts and the
  best model
- `gui_app.py` — Tkinter scanner that loads the saved model
- `results/` — `metrics.csv` and charts (confusion matrices + model comparison)
  for the report
- `tests/` — unit tests (`py -m pytest`)

## Tests

```
py -m pytest
```
