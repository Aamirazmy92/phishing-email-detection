from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from explain import explain

TEXTS = ["verify your account password now", "urgent account suspended verify",
         "meeting notes attached thanks", "lunch tomorrow thanks see you"]
LABELS = [1, 1, 0, 0]


def _fit(model):
    vec = TfidfVectorizer().fit(TEXTS)
    return model.fit(vec.transform(TEXTS), LABELS), vec


def test_explain_all_model_types():
    for m in (LogisticRegression(), LinearSVC(), MultinomialNB()):
        model, vec = _fit(m)
        r = explain("Please verify your account password", model, vec)
        assert r["label"] == "Phishing"
        assert 0.5 <= r["confidence"] <= 1
        words = [w for w, _ in r["phishing_terms"]]
        assert "verify" in words or "password" in words


def test_explain_legit_and_unknown_words():
    model, vec = _fit(LogisticRegression())
    assert explain("meeting notes thanks", model, vec)["label"] == "Legitimate"
    r = explain("zzzz qqqq", model, vec)  # no known words -> no crash
    assert r["phishing_terms"] == [] and r["legit_terms"] == []
