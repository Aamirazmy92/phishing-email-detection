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
