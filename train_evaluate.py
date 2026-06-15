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
