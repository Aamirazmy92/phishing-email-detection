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
