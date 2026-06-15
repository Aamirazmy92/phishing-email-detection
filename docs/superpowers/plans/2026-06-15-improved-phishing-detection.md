# Improved Phishing Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the phishing email detector into shared-preprocessing + training/evaluation + GUI modules, add rigorous multi-model evaluation with charts, and load a saved model in the GUI.

**Architecture:** Three Python modules — `preprocessing.py` (shared text cleaning), `train_evaluate.py` (trains & compares Naïve Bayes / Logistic Regression / Linear SVM, saves metrics, charts and the best model), and `gui_app.py` (loads the saved model, offers a paste-an-email tester plus the existing Gmail scan). Artifacts live in `models/` and `results/`.

**Tech Stack:** Python 3.13, scikit-learn, pandas, nltk, matplotlib, seaborn, joblib, tkinter, pytest.

**Working directory for all commands:** `D:\Uni\FINAL YEAR\Code\FYP Code`

---

### Task 0: Project setup

**Files:**
- Create: `.gitignore`
- Modify: `requirements.txt`

- [ ] **Step 1: Initialize Git and install test/dev deps**

Run (from the project folder):
```bash
git init
py -m pip install pytest matplotlib seaborn joblib
```
Expected: repo initialized; packages install successfully (pandas/scikit-learn/nltk already present).

- [ ] **Step 2: Create `.gitignore`**

```
venv/
__pycache__/
*.pyc
models/
results/
.pytest_cache/
```

- [ ] **Step 3: Replace `requirements.txt` with what is actually used**

```
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore requirements.txt
git commit -m "chore: init repo, pin actual dependencies"
```

---

### Task 1: Shared preprocessing module

**Files:**
- Create: `preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_preprocessing.py`:
```python
from preprocessing import preprocess_text


def test_lowercases_and_removes_stopwords():
    assert preprocess_text("The QUICK brown fox") == "quick brown fox"


def test_masks_urls():
    assert "url" in preprocess_text("visit http://evil.com now")


def test_masks_emails():
    assert "email" in preprocess_text("contact me at bob@example.com")


def test_removes_digits_and_punctuation():
    assert preprocess_text("Order #12345!!!") == "order"


def test_handles_non_string_input():
    assert preprocess_text(None) == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'preprocessing'`

- [ ] **Step 3: Write minimal implementation**

`preprocessing.py`:
```python
import re
import string

import nltk

nltk.download("stopwords", quiet=True)
_STOP_WORDS = set(nltk.corpus.stopwords.words("english"))


def preprocess_text(text):
    """Lowercase, mask URLs/emails, strip punctuation/digits, drop stopwords."""
    text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    return " ".join(t for t in tokens if t not in _STOP_WORDS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/test_preprocessing.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add shared text preprocessing module with tests"
```

---

### Task 2: Dataset loading helpers

**Files:**
- Create: `data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_data_loader.py`:
```python
import pandas as pd
import pytest

from data_loader import build_text_column, load_dataset


def test_build_text_column_from_subject_body():
    df = pd.DataFrame({"subject": ["Hi"], "body": ["there"]})
    assert build_text_column(df).iloc[0] == "Hi there"


def test_build_text_column_from_text():
    df = pd.DataFrame({"text": ["already combined"]})
    assert build_text_column(df).iloc[0] == "already combined"


def test_build_text_column_missing_raises():
    df = pd.DataFrame({"unrelated": [1]})
    with pytest.raises(ValueError):
        build_text_column(df)


def test_load_dataset_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("does_not_exist.csv")


def test_load_dataset_missing_label_raises(tmp_path):
    csv = tmp_path / "no_label.csv"
    csv.write_text("subject,body\nhi,there\n")
    with pytest.raises(ValueError):
        load_dataset(str(csv))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -m pytest tests/test_data_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_loader'`

- [ ] **Step 3: Write minimal implementation**

`data_loader.py`:
```python
import os

import pandas as pd

from preprocessing import preprocess_text

DATA_FILE = "CEAS_08.csv"


def build_text_column(df):
    if "subject" in df.columns and "body" in df.columns:
        return df["subject"].fillna("") + " " + df["body"].fillna("")
    if "text" in df.columns:
        return df["text"].fillna("")
    raise ValueError(
        "CSV must include 'subject' and 'body' columns, or a 'text' column."
    )


def load_dataset(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{path}' not found. Place CEAS_08.csv in this folder."
        )
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(
            "CSV must include a 'label' column (1 = phishing, 0 = legitimate)."
        )
    df["text"] = build_text_column(df)
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -m pytest tests/test_data_loader.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add data_loader.py tests/test_data_loader.py
git commit -m "feat: add dataset loading with column validation and tests"
```

---

### Task 3: Evaluation helper

**Files:**
- Create: `evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

`tests/test_evaluation.py`:
```python
from sklearn.naive_bayes import MultinomialNB

from evaluation import evaluate_model, get_models


def test_get_models_returns_three_named_models():
    models = get_models()
    assert set(models.keys()) == {
        "Naive Bayes",
        "Logistic Regression",
        "Linear SVM",
    }


def test_evaluate_model_returns_metrics_and_matrix():
    # Tiny separable dataset: feature 0 -> class 0, feature 1 -> class 1
    X = [[1, 0], [1, 0], [0, 1], [0, 1]]
    y = [0, 0, 1, 1]
    model = MultinomialNB().fit(X, y)
    metrics, cm = evaluate_model(model, X, y)
    assert metrics["accuracy"] == 1.0
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert cm.shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_evaluation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation'`

- [ ] **Step 3: Write minimal implementation**

`evaluation.py`:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def get_models():
    """The three simple, interpretable models being compared."""
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }


def evaluate_model(model, X_test, y_test):
    """Return (metrics dict, confusion matrix) for a fitted model."""
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    return metrics, confusion_matrix(y_test, preds)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_evaluation.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add evaluation.py tests/test_evaluation.py
git commit -m "feat: add model registry and evaluation helper with tests"
```

---

### Task 4: Training & evaluation script (charts + saved model)

**Files:**
- Create: `train_evaluate.py`

This is a script (produces files + console output), verified by running rather than unit tests — it depends on the full 64 MB dataset.

- [ ] **Step 1: Write the script**

`train_evaluate.py`:
```python
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # no GUI needed for saving figures
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split

from data_loader import load_dataset
from evaluation import evaluate_model, get_models
from preprocessing import preprocess_text

RANDOM_STATE = 42
MODELS_DIR = "models"
RESULTS_DIR = "results"


def _slug(name):
    return name.replace(" ", "_").lower()


def main():
    df = load_dataset()
    X, y = df["clean_text"], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer()
    X_train_v = vectorizer.fit_transform(X_train)  # fit on train only (no leakage)
    X_test_v = vectorizer.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    best = None  # (name, f1, model)
    for name, model in get_models().items():
        model.fit(X_train_v, y_train)
        metrics, cm = evaluate_model(model, X_test_v, y_test)
        cv = cross_val_score(model, X_train_v, y_train, cv=5, scoring="f1")
        metrics["cv_f1_mean"] = cv.mean()
        metrics["cv_f1_std"] = cv.std()
        metrics["model"] = name
        rows.append(metrics)

        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Phish"], yticklabels=["Legit", "Phish"],
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{_slug(name)}.png"))
        plt.close()

        if best is None or metrics["f1"] > best[1]:
            best = (name, metrics["f1"], model)

    results_df = pd.DataFrame(rows)[
        ["model", "accuracy", "precision", "recall", "f1", "cv_f1_mean", "cv_f1_std"]
    ]
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=results_df, x="model", y="f1")
    plt.title("Model Comparison (F1-score)")
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"))
    plt.close()

    best_name, best_f1, best_model = best
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))

    print("\n=== Model comparison ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name} (F1={best_f1:.3f}) saved to {MODELS_DIR}/")
    print(f"Charts and metrics.csv saved to {RESULTS_DIR}/")

    # Sanity check: predict on two obvious examples
    examples = [
        "URGENT: Your account is suspended. Click http://verify-now.com to "
        "confirm your password immediately.",
        "Hi team, attaching the meeting notes from yesterday. Let me know if "
        "you have any questions. Thanks.",
    ]
    preds = best_model.predict(
        vectorizer.transform([preprocess_text(e) for e in examples])
    )
    print("\n=== Sanity check ===")
    for example, pred in zip(examples, preds):
        label = "Phishing" if pred == 1 else "Legitimate"
        print(f"[{label}] {example[:55]}...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script end-to-end**

Run: `py train_evaluate.py`
Expected: a comparison table printed; `models/best_model.joblib`, `models/vectorizer.joblib`, `results/metrics.csv`, `results/model_comparison.png`, and three `results/confusion_matrix_*.png` files created; sanity check prints `[Phishing]` for the first example and `[Legitimate]` for the second.

- [ ] **Step 3: Verify artifacts exist**

Run: `py -c "import os; print([f for f in os.listdir('results')]); print(os.path.exists('models/best_model.joblib'))"`
Expected: lists the 5 result files and prints `True`.

- [ ] **Step 4: Commit**

```bash
git add train_evaluate.py
git commit -m "feat: add training script with model comparison, charts, saved model"
```

---

### Task 5: GUI that loads the saved model

**Files:**
- Create: `gui_app.py`
- Test: `tests/test_gui_classify.py`

- [ ] **Step 1: Write the failing test for the classify helper**

`tests/test_gui_classify.py`:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from gui_app import classify


def test_classify_returns_label_string():
    texts = ["win free money now click link", "project meeting notes attached"]
    labels = [1, 0]
    vec = TfidfVectorizer().fit(texts)
    model = MultinomialNB().fit(vec.transform(texts), labels)
    result = classify("win free money now", model, vec)
    assert result in ("Phishing", "Legitimate")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_gui_classify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gui_app'`

- [ ] **Step 3: Write the implementation**

`gui_app.py`:
```python
import email
import imaplib
import os
import tkinter as tk
from email.header import decode_header
from tkinter import messagebox, scrolledtext

import joblib

from preprocessing import preprocess_text

MODEL_PATH = os.path.join("models", "best_model.joblib")
VECTORIZER_PATH = os.path.join("models", "vectorizer.joblib")


def load_artifacts():
    """Return (model, vectorizer) or (None, None) if not yet trained."""
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def classify(text, model, vectorizer):
    """Classify raw email text. Returns 'Phishing' or 'Legitimate'."""
    cleaned = preprocess_text(text)
    pred = model.predict(vectorizer.transform([cleaned]))[0]
    return "Phishing" if int(pred) == 1 else "Legitimate"


class EmailScannerApp:
    def __init__(self, root, model, vectorizer):
        self.root = root
        self.model = model
        self.vectorizer = vectorizer
        root.title("Phishing Email Scanner")
        root.geometry("520x540")

        tk.Label(root, text="Phishing Email Scanner", font=("Arial", 16)).pack(pady=8)

        # --- Offline tester ---
        tk.Label(root, text="Paste an email to test (no login needed):").pack()
        self.email_box = scrolledtext.ScrolledText(root, width=60, height=8)
        self.email_box.pack(padx=10, pady=4)
        tk.Button(root, text="Classify pasted email",
                  command=self.classify_pasted).pack(pady=4)
        self.result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=4)

        # --- Gmail scan ---
        tk.Label(root, text="--- or scan a Gmail inbox ---").pack(pady=6)
        tk.Label(root, text="Email:").pack()
        self.email_entry = tk.Entry(root, width=40)
        self.email_entry.pack()
        tk.Label(root, text="App password:").pack()
        self.password_entry = tk.Entry(root, show="*", width=40)
        self.password_entry.pack()
        tk.Button(root, text="Login and Scan Inbox",
                  command=self.login_and_scan).pack(pady=8)

    def classify_pasted(self):
        text = self.email_box.get("1.0", tk.END).strip()
        if not text:
            self.result_label.config(text="Please paste some email text first.")
            return
        label = classify(text, self.model, self.vectorizer)
        colour = "red" if label == "Phishing" else "green"
        self.result_label.config(text=f"Prediction: {label}", fg=colour)

    def login_and_scan(self):
        email_user = self.email_entry.get()
        email_pass = self.password_entry.get()
        try:
            imap = imaplib.IMAP4_SSL("imap.gmail.com")
            imap.login(email_user, email_pass)
            imap.select("inbox")
            _, messages = imap.search(None, "UNSEEN")
            mail_ids = messages[0].split()
            results = []
            for num in mail_ids[:5]:
                _, data = imap.fetch(num, "(RFC822)")
                msg = email.message_from_bytes(data[0][1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="ignore")
                label = classify(f"{subject} {body}", self.model, self.vectorizer)
                results.append(f"Subject: {subject}\nPrediction: {label}\n")
            imap.logout()
            self.show_results(results or ["No unread emails found."])
        except Exception as exc:
            messagebox.showerror("Login Failed", f"Error: {exc}")

    def show_results(self, results):
        win = tk.Toplevel(self.root)
        win.title("Scan Results")
        box = scrolledtext.ScrolledText(win, width=80, height=20)
        box.pack(padx=10, pady=10)
        box.insert(tk.END, "\n\n".join(results))
        box.config(state=tk.DISABLED)


def main():
    model, vectorizer = load_artifacts()
    if model is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Model not found",
            "No trained model found. Please run 'py train_evaluate.py' first.",
        )
        return
    root = tk.Tk()
    EmailScannerApp(root, model, vectorizer)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_gui_classify.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Manually launch the GUI**

Run: `py gui_app.py`
Expected: window opens; pasting the phishing example from Task 4 and clicking "Classify pasted email" shows "Prediction: Phishing" in red; a normal message shows "Legitimate" in green. Close the window.

- [ ] **Step 6: Commit**

```bash
git add gui_app.py tests/test_gui_classify.py
git commit -m "feat: GUI loads saved model, adds offline paste-email tester"
```

---

### Task 6: Full test run + README note

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the whole test suite**

Run: `py -m pytest -v`
Expected: all tests pass (preprocessing 5, data_loader 5, evaluation 2, gui_classify 1).

- [ ] **Step 2: Write `README.md`**

```markdown
# Fraudulent Behaviour Prevention System (NLP + ML)

Phishing email detector. Compares three interpretable models and runs a
Tkinter scanner that loads the best one.

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
   Paste an email to test offline, or log in with a Gmail **app password**
   to scan unread inbox messages.

## Files
- `preprocessing.py` — shared text cleaning (used by training and GUI)
- `data_loader.py` — loads/validates `CEAS_08.csv`
- `evaluation.py` — model registry + metric calculation
- `train_evaluate.py` — trains NB / Logistic Regression / Linear SVM,
  saves metrics, charts and the best model
- `gui_app.py` — Tkinter scanner that loads the saved model
- `results/` — metrics.csv and charts for the report
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and usage"
```

---

## Definition of done
- `py -m pytest -v` passes (13 tests).
- `py train_evaluate.py` produces `metrics.csv`, a comparison chart, three confusion-matrix charts, and saved model + vectorizer; sanity check labels the two examples correctly.
- `py gui_app.py` opens, classifies a pasted email, and shows a clear error if the model is missing.
