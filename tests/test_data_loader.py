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
