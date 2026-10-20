"""Evaluate the saved model on a different dataset (out-of-distribution test).

Usage: py cross_dataset_eval.py other.csv
The CSV needs a 'label' column (1 = phishing) and either 'subject'+'body' or 'text'.
Some public sets use text labels; "phishing"/"spam" -> 1 and "legitimate"/"ham"/"safe" -> 0.
"""
import sys

import joblib
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_dataset

POS = {"phishing", "phishing email", "spam", "1"}
NEG = {"legitimate", "ham", "safe email", "safe", "0"}


def main(path):
    df = load_dataset(path)
    df["label"] = df["label"].map(
        lambda v: 1 if str(v).strip().lower() in POS else
        0 if str(v).strip().lower() in NEG else None)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    model = joblib.load("models/best_model.joblib")
    vec = joblib.load("models/vectorizer.joblib")
    preds = model.predict(vec.transform(df["clean_text"]))
    print(f"{len(df)} emails from {path}")
    print(confusion_matrix(df["label"], preds))
    print(classification_report(df["label"], preds, target_names=["Legit", "Phish"], zero_division=0))


if __name__ == "__main__":
    main(sys.argv[1])
