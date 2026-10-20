"""Explain a prediction of a linear text classifier (LinearSVC / LogReg / NB)."""
import numpy as np

from preprocessing import preprocess_text


def _weights(model):
    if hasattr(model, "coef_"):
        return model.coef_[0]
    if hasattr(model, "feature_log_prob_"):  # Naive Bayes: log-odds per word
        return model.feature_log_prob_[1] - model.feature_log_prob_[0]
    raise TypeError(f"Cannot explain {type(model).__name__}")


def _score(model, X):
    if hasattr(model, "decision_function"):
        return float(model.decision_function(X)[0])
    p = model.predict_proba(X)[0, 1]
    return float(np.log(p / (1 - p)))


def explain(text, model, vectorizer, top_k=6, subject="", sender=None):
    """Return dict: label, confidence (0-1), phishing_terms, legit_terms.

    Each term list is [(feature, contribution)]; features are words or named
    URL/sender signals such as has_ip_url. Each entry is (name, contribution), strongest first. Contribution is
    weight * tf-idf value, so it sums (plus bias) to the model's raw score.
    Confidence is a logistic squashing of that score: a readable indicator of
    how far from the decision boundary the email is, not a calibrated probability.
    """
    if hasattr(vectorizer, "transform_raw"):  # EmailFeaturizer: text + URL/sender features
        X = vectorizer.transform_raw(subject, text, sender)
    else:  # plain TF-IDF vectorizer
        X = vectorizer.transform([preprocess_text(text)])
    score = _score(model, X)
    contrib = X.multiply(_weights(model)).tocoo()
    names = vectorizer.get_feature_names_out()
    terms = sorted(
        ((names[j], float(v)) for j, v in zip(contrib.col, contrib.data)),
        key=lambda t: t[1],
    )
    phish = [t for t in reversed(terms) if t[1] > 0][:top_k]
    legit = [t for t in terms if t[1] < 0][:top_k]
    is_phish = score > 0
    return {
        "label": "Phishing" if is_phish else "Legitimate",
        "confidence": float(1 / (1 + np.exp(-abs(score)))),
        "phishing_terms": phish,
        "legit_terms": legit,
    }


def format_explanation(result):
    lines = [f"{result['label']} (confidence {result['confidence']:.0%})"]
    if result["phishing_terms"]:
        lines.append("Pushes toward phishing: " +
                     ", ".join(w for w, _ in result["phishing_terms"]))
    if result["legit_terms"]:
        lines.append("Pushes toward legitimate: " +
                     ", ".join(w for w, _ in result["legit_terms"]))
    return "\n".join(lines)
