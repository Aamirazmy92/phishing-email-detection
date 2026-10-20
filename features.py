"""Hand-built URL, content and sender features to complement TF-IDF text.

Everything is computed from the raw email (before text cleaning), because the
cleaning step replaces URLs and addresses with placeholders.
"""
import math
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from preprocessing import preprocess_text

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.I)
ANCHOR_RE = re.compile(r"<a\s[^>]*href=[\"']?([^\"'\s>]+)[^>]*>(.*?)</a>", re.I | re.S)
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly"}
FREEMAIL = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "mail.com",
            "yahoo.co.uk", "live.com", "msn.com", "protonmail.com"}
URGENCY = re.compile(
    r"\b(urgent|immediately|verify|suspend\w*|confirm|expire\w*|limited|locked|"
    r"within \d+ hours|act now|final notice|security alert|unusual activity|"
    r"update your|click (?:here|below)|password|account)\b", re.I)
ADDR_RE = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")

URL_FEATURES = [
    "n_urls", "n_distinct_url_domains", "has_ip_url", "has_shortener", "has_at_in_url",
    "max_subdomain_depth", "frac_http_not_https", "max_url_len", "anchor_mismatch",
]
CONTENT_FEATURES = ["has_html", "urgency_hits", "n_exclaim", "upper_ratio", "digit_ratio", "log_len"]
SENDER_FEATURES = [
    "has_sender", "sender_freemail", "sender_has_display_name",
    "sender_domain_in_urls", "sender_domain_digits",
]
GROUPS = {"url": URL_FEATURES, "content": CONTENT_FEATURES, "sender": SENDER_FEATURES}


def _domain(url):
    u = url if "//" in url else "http://" + url
    try:
        return (urlparse(u).hostname or "").lower().strip(".")
    except ValueError:
        return ""


def _registered(domain):
    parts = domain.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain


def extract_features(subject, body, sender=None):
    """Return dict of numeric features for one raw email."""
    subject = "" if subject is None or (isinstance(subject, float) and math.isnan(subject)) else str(subject)
    body = "" if body is None or (isinstance(body, float) and math.isnan(body)) else str(body)
    text = f"{subject} {body}"
    urls = URL_RE.findall(text)
    domains = [d for d in (_domain(u) for u in urls) if d]
    f = {k: 0.0 for k in URL_FEATURES + CONTENT_FEATURES + SENDER_FEATURES}

    f["n_urls"] = len(urls)
    f["n_distinct_url_domains"] = len({_registered(d) for d in domains})
    f["has_ip_url"] = float(any(IP_RE.match(d) for d in domains))
    f["has_shortener"] = float(any(_registered(d) in SHORTENERS for d in domains))
    f["has_at_in_url"] = float(any("@" in u.split("//", 1)[-1].split("/", 1)[0] for u in urls))
    f["max_subdomain_depth"] = max((0 if IP_RE.match(d) else max(len(d.split(".")) - 2, 0) for d in domains), default=0)
    f["frac_http_not_https"] = (sum(u.lower().startswith("http://") for u in urls) / len(urls)) if urls else 0.0
    f["max_url_len"] = max((len(u) for u in urls), default=0)
    mismatch = 0
    for href, label in ANCHOR_RE.findall(body):
        shown = URL_RE.search(re.sub(r"<[^>]+>", "", label))
        if shown and _registered(_domain(shown.group(0))) != _registered(_domain(href)):
            mismatch = 1
            break
    f["anchor_mismatch"] = float(mismatch)

    letters = [c for c in text if c.isalpha()]
    f["has_html"] = float(bool(re.search(r"<(html|body|a\s|table|div|br)\b", body, re.I)))
    f["urgency_hits"] = len(URGENCY.findall(text))
    f["n_exclaim"] = text.count("!")
    f["upper_ratio"] = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    f["digit_ratio"] = (sum(c.isdigit() for c in text) / len(text)) if text else 0.0
    f["log_len"] = math.log1p(len(text))

    if sender is not None and isinstance(sender, str) and sender.strip():
        m = ADDR_RE.search(sender)
        dom = m.group(1).lower() if m else ""
        f["has_sender"] = 1.0
        f["sender_freemail"] = float(dom in FREEMAIL)
        f["sender_has_display_name"] = float(bool(re.sub(r"<[^>]*>|[\"'\s]", "", sender.split("<")[0]).strip()
                                                  and "<" in sender))
        f["sender_domain_in_urls"] = float(bool(dom) and any(_registered(d) == _registered(dom) for d in domains))
        f["sender_domain_digits"] = float(bool(re.search(r"\d", dom.split(".")[0] if dom else "")))
    return f


class EmailFeaturizer:
    """TF-IDF text features + optional hand-built feature groups, one matrix.

    transform() takes a DataFrame with columns subject, body and (optionally)
    sender, so it can be pickled and reused by the GUI.
    """

    def __init__(self, groups=("url", "content", "sender"), **tfidf_kwargs):
        self.groups = tuple(groups)
        self.tfidf = TfidfVectorizer(**tfidf_kwargs)
        self.scaler = MaxAbsScaler()
        self.numeric_names = [n for g in self.groups for n in GROUPS[g]]

    def _numeric(self, df):
        sender = df["sender"] if "sender" in df else [None] * len(df)
        rows = [extract_features(s, b, snd) for s, b, snd in zip(df["subject"], df["body"], sender)]
        return np.log1p(pd.DataFrame(rows)[self.numeric_names].to_numpy(dtype=float))

    def _clean(self, df):
        if "clean_text" in df:
            return df["clean_text"]
        return [preprocess_text(f"{s} {b}") for s, b in zip(df["subject"].fillna(""), df["body"].fillna(""))]

    def fit_transform(self, df):
        X = self.tfidf.fit_transform(self._clean(df))
        if not self.numeric_names:
            return X
        return sparse.hstack([X, sparse.csr_matrix(self.scaler.fit_transform(self._numeric(df)))]).tocsr()

    def transform(self, df):
        X = self.tfidf.transform(self._clean(df))
        if not self.numeric_names:
            return X
        return sparse.hstack([X, sparse.csr_matrix(self.scaler.transform(self._numeric(df)))]).tocsr()

    def get_feature_names_out(self):
        return np.concatenate([self.tfidf.get_feature_names_out(), np.array(self.numeric_names, dtype=object)])

    def transform_raw(self, subject, body, sender=None):
        """Featurise one raw email (used by the GUI/explainer)."""
        return self.transform(pd.DataFrame({"subject": [subject], "body": [body], "sender": [sender]}))
