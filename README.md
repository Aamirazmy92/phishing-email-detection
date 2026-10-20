# Fraudulent Behaviour Prevention System (NLP + ML)

A phishing email detector built with interpretable machine-learning models
(Naive Bayes, Logistic Regression, Linear SVM) on TF-IDF text features plus
hand-built URL, content and sender features. A Tkinter app classifies a pasted
email and **explains** each verdict.

> **Status:** academic prototype. See [Limitations](#limitations).

## Setup

```
py -m pip install -r requirements.txt
```

### Data (not included in the repo)

Download the files from the Zenodo record
[Phishing Email Curated Datasets](https://zenodo.org/records/8339691)
(Champa, Rabbi and Zibran, CC-BY-4.0) and place them like this:

```
CEAS_08.csv                    # project root
datasets/SpamAssasin.csv
datasets/Enron.csv
datasets/Nazario.csv
datasets/Nigerian_Fraud.csv
```

Label convention: `1` = phishing/spam, `0` = legitimate.

## Usage

| Step | Command | What it does |
|---|---|---|
| Baseline | `py train_evaluate.py` | Trains and compares the three models on CEAS only; writes `results/metrics.csv` and charts |
| Multi-source | `py train_multisource.py` | Leave-one-source-out evaluation over all five corpora and feature sets; saves the final model to `models/` |
| Other data | `py cross_dataset_eval.py other.csv` | Scores the saved model on any labelled CSV |
| App | `py gui_app.py` | Desktop app: paste an email and see the verdict, confidence and the words behind it |
| Tests | `py -m pytest` | Unit tests |

## Findings

**A single-corpus score is misleading.** On CEAS-08 alone, Linear SVM reaches
F1 0.996 after removing duplicate emails. But the model learned the corpus's
topics, not phishing: its strongest "legitimate" words are `python`, `perl`,
`opensuse`, `patch` (mailing lists) and its strongest "phishing" words are
`pills`, `watches`, `replica` (2008 pharma spam). On other corpora it fails:

| Test set | CEAS-only model |
|---|---|
| Nazario (real phishing), recall | 16% |
| Nigerian_Fraud, recall | 68% |
| SpamAssassin, spam recall / accuracy | 38% / 75% |
| Enron, accuracy | 72% |

**Training on several sources helps, evaluated on a source it never saw**
(Linear SVM, text features, leave-one-source-out):

| Held-out source | Metric | Score |
|---|---|---|
| CEAS-08 | F1 / false-positive rate | 0.91 / 11% |
| SpamAssassin | F1 / false-positive rate | 0.88 / 8% |
| Enron | accuracy / false-positive rate | 82% / 26% |
| Nazario | recall | 49% |
| Nigerian_Fraud | recall | 99% |

Full tables (both models, all feature sets) are in `results/leave_one_source_out.csv`.

### Do URL and sender features help? Not on unseen sources

`features.py` adds URL signals (IP-address links, shorteners, `@` in a URL,
subdomain depth, link-text/target mismatch), content signals (HTML, urgency
words, capitals) and sender signals. Leave-one-source-out, Linear SVM:

| Held-out source (metric) | Text only | + URL | + URL + content | + all incl. sender |
|---|---|---|---|---|
| CEAS-08 (F1) | 0.911 | 0.916 | 0.911 | 0.899 |
| SpamAssassin (F1) | 0.880 | 0.852 | 0.866 | 0.845 |
| Enron (F1) | 0.834 | 0.817 | 0.830 | 0.806 |
| Nazario (recall) | 0.492 | 0.519 | 0.514 | 0.461 |
| Nigerian_Fraud (recall) | 0.993 | 0.990 | 0.989 | 0.988 |

The extra features give at most small, inconsistent changes, and the sender
group makes things worse (Enron has no sender field, so "has a sender" mostly
identifies the source rather than the class). The final saved model therefore
uses **text only**. Likely reasons: the corpora are old and their URLs are
rarely the distinguishing signal, and the labels are spam-vs-ham rather than
phishing-specific.

## Explainability

Every prediction in the app lists the features pushing it toward phishing or
legitimate (words, or named signals such as `has_ip_url`). Contributions are
`weight x feature value` for the linear model. The confidence figure is a
logistic squashing of the distance from the decision boundary, **not** a
calibrated probability.

## Project layout

- `preprocessing.py` - text cleaning shared by training and the app
- `data_loader.py` - loads and validates a CSV
- `features.py` - URL / content / sender feature extraction and `EmailFeaturizer`
- `evaluation.py` - model registry and metrics
- `train_evaluate.py` - single-corpus baseline
- `train_multisource.py` - multi-source training and leave-one-source-out evaluation
- `cross_dataset_eval.py` - evaluate the saved model on another dataset
- `explain.py` - per-prediction explanations
- `gui_app.py` - Tkinter app
- `tests/` - unit tests

## Limitations

- The corpora are mostly from the 2000s. Current phishing (brand impersonation,
  QR codes, business-email-compromise) is underrepresented.
- Enron and SpamAssassin are labelled spam vs. legitimate, not phishing specifically.
- No dataset here has a Reply-To header, so sender/Reply-To mismatch is not used.
  Enron has no sender at all, so sender features exist only for the other sources.
- Text-only models are easy to evade by rewording; no adversarial testing was done.
- The app classifies one pasted email at a time; there is no mailbox integration.

## License

MIT - see `LICENSE`.
