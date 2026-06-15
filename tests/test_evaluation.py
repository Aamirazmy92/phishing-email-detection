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
