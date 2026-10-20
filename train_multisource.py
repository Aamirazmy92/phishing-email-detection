"""Train on several corpora and measure generalisation to unseen sources.

1. Leave-one-source-out: train on every source except one, test on the one
   left out. This is the realistic estimate for email from somewhere new.
2. Train the final model on all sources and save it for the GUI.

Run:  py train_multisource.py [--final-only]   (--final-only skips the slow ablation)
"""
import os
import shutil
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

from data_loader import load_dataset
from features import EmailFeaturizer

SOURCES = {
    "CEAS_08": "CEAS_08.csv",
    "SpamAssassin": "datasets/SpamAssasin.csv",
    "Enron": "datasets/Enron.csv",
    "Nazario": "datasets/Nazario.csv",
    "Nigerian_Fraud": "datasets/Nigerian_Fraud.csv",
}
MODELS = {
    "Linear SVM": lambda: LinearSVC(class_weight="balanced"),
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
}
RESULTS_DIR, MODELS_DIR = "results", "models"


def load_all():
    frames = []
    for name, path in SOURCES.items():
        df = load_dataset(path)
        df = df.reindex(columns=["clean_text", "label", "subject", "body", "sender"]).copy()
        df["source"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["clean_text"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates("clean_text")  # also removes cross-source copies
    print(f"{len(df)} emails after dropping {before - len(df)} duplicates")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0), "\n")
    return df


VARIANTS = {
    "text only": (),
    "text + url": ("url",),
    "text + url + content": ("url", "content"),
    "text + url + content + sender": ("url", "content", "sender"),
}
# Chosen from the ablation (results/leave_one_source_out.csv): the extra feature
# groups did not improve held-out performance, and sender features hurt it.
FINAL_VARIANT = "text only"


def make_vectorizer(groups=()):
    return EmailFeaturizer(groups=groups, sublinear_tf=True, ngram_range=(1, 2),
                           min_df=3, max_features=300_000)


def score(y, pred):
    out = {"n": len(y), "recall": recall_score(y, pred, zero_division=0)}
    if y.nunique() == 2:  # precision/F1/FPR only make sense with both classes
        out.update(
            accuracy=accuracy_score(y, pred),
            precision=precision_score(y, pred, zero_division=0),
            f1=f1_score(y, pred, zero_division=0),
            false_positive_rate=float(((pred == 1) & (y == 0)).sum() / (y == 0).sum()),
        )
    return out


def main():
    df = load_all()
    rows = []
    for held in ([] if "--final-only" in sys.argv else SOURCES):
        train, test = df[df.source != held], df[df.source == held]
        for vname, groups in VARIANTS.items():
            vec = make_vectorizer(groups)
            Xtr = vec.fit_transform(train)
            Xte = vec.transform(test)
            for mname, make in MODELS.items():
                pred = make().fit(Xtr, train.label).predict(Xte)
                rows.append({"held_out_source": held, "features": vname, "model": mname,
                             **score(test.label, pred)})
                print(f"held out {held:15s} {vname:30s} {mname:20s}",
                      {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if rows:  # --final-only skips the ablation and must not overwrite its results
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(RESULTS_DIR, "leave_one_source_out.csv"), index=False)
        print("\n", res.round(3).to_string(index=False))

    # Final model on everything. Keep the old CEAS-only model as a baseline.
    os.makedirs(MODELS_DIR, exist_ok=True)
    for f in ("best_model.joblib", "vectorizer.joblib"):
        src, dst = os.path.join(MODELS_DIR, f), os.path.join(MODELS_DIR, "baseline_ceas_" + f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    vec = make_vectorizer(VARIANTS[FINAL_VARIANT])
    X = vec.fit_transform(df)
    model = MODELS["Linear SVM"]().fit(X, df.label)
    joblib.dump(model, os.path.join(MODELS_DIR, "best_model.joblib"))
    joblib.dump(vec, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    print(f"\nSaved multi-source Linear SVM ({FINAL_VARIANT}) to {MODELS_DIR}/ (old model kept as baseline_ceas_*)")


if __name__ == "__main__":
    main()
