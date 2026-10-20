import pandas as pd

from features import EmailFeaturizer, extract_features


def test_url_signals():
    f = extract_features(
        "Account limited",
        "Verify at http://192.168.4.4/login or <a href='http://evil.example.ru/x'>https://paypal.com</a> "
        "or http://bit.ly/abc",
        "PayPal <help@paypa1-mail.com>",
    )
    assert f["has_ip_url"] == 1 and f["has_shortener"] == 1
    assert f["anchor_mismatch"] == 1 and f["has_html"] == 1
    assert f["sender_domain_in_urls"] == 0 and f["sender_domain_digits"] == 1
    assert f["max_subdomain_depth"] == 1  # evil.example.ru has one subdomain; the IP URL adds none


def test_plain_email_has_no_signals():
    f = extract_features("Lunch?", "Are you free tomorrow? Thanks")
    assert f["n_urls"] == 0 and f["has_ip_url"] == 0 and f["has_sender"] == 0


def test_handles_missing_values():
    assert extract_features(float("nan"), None, None)["n_urls"] == 0


def test_featurizer_roundtrip():
    df = pd.DataFrame({
        "subject": ["verify now", "meeting notes", "verify account", "lunch plans"],
        "body": ["http://1.2.3.4/x click", "see attached", "http://bit.ly/z urgent", "tomorrow ok"],
        "sender": ["a <a@x1.com>", "b <b@work.org>", None, None],
    })
    fz = EmailFeaturizer(groups=("url", "content", "sender"))
    X = fz.fit_transform(df)
    assert X.shape[0] == 4 and X.shape[1] == len(fz.get_feature_names_out())
    one = fz.transform_raw("verify now", "http://1.2.3.4/x click", None)
    assert one.shape == (1, X.shape[1])
