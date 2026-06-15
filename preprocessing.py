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
