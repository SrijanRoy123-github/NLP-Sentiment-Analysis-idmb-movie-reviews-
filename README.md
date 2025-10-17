# IMDB Sentiment Analysis — Dual Pipeline (TF‑IDF + BiLSTM)

> **One‑liner:** IMDB sentiment classifier with **NLTK + spaCy** preprocessing, **TF‑IDF + Logistic Regression** and **Keras BiLSTM** dual models, trained with **5‑fold cross‑validation** and a brief **negation/sarcasm error analysis**.

## Overview
This project builds a robust movie review sentiment classifier on the **IMDB 50K** dataset using two complementary approaches:
1. **Classical ML** — TF‑IDF (word + character n‑grams) → Logistic Regression  
2. **Deep Learning** — Tokenizer → padded sequences → **BiLSTM**

The text pipeline uses **NLTK + spaCy** for tokenization, cleaning, and lemmatization (with graceful fallbacks if `en_core_web_sm` isn’t available). We keep **negation tokens** (e.g., *not, never, n't*) to preserve sentiment cues.  
With 5‑fold CV, results are typically around **Accuracy ≈ 0.90** and **F1 ≈ 0.88** (dataset/version dependent).

## Features
- Clean preprocessing (HTML/URLs removal, punctuation normalization, lemmatization)
- Dual track models: **TF‑IDF+LR** and **BiLSTM**
- **5‑fold CV** for each track
- **Safe inference helper** that uses `.transform()` (avoids `min_df` issues)
- Brief **error analysis** focusing on **negation** and simple **sarcasm** cues

## Tech Stack
- **Python**, **pandas**, **NumPy**
- **scikit‑learn** (TF‑IDF, Logistic Regression, metrics)
- **TensorFlow / Keras** (BiLSTM)
- **NLTK** (stopwords, WordNet lemmatizer) & **spaCy** (tokenization/lemmatization)
- (Optional) **GPU** for faster BiLSTM on Kaggle/Colab

## Repository Structure (suggested)
```
.
├── notebooks/
│   └── imdb_sentiment_dual_pipeline_fixed.ipynb   # Main notebook (dual pipeline + CV + inference)
├── data/
│   └── IMDB Dataset.csv                            # (Optional local copy; Kaggle path supported)
└── README.md
```

> **Download this notebook:** `notebooks/imdb_sentiment_dual_pipeline_fixed.ipynb` (provided alongside this README).

## Dataset
Use Kaggle’s **“IMDB Dataset of 50K Movie Reviews”**. In Kaggle, add it via *Add Data* (path will be `/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv`).  
Locally, place `IMDB Dataset.csv` next to the notebook or under `./data/` and update the path in the notebook if needed.

## Quick Start

### A) Run on Kaggle (recommended)
1. Create a new notebook and **Add Data** → *IMDB Dataset of 50K Movie Reviews*.
2. Upload `notebooks/imdb_sentiment_dual_pipeline_fixed.ipynb` (this repo’s notebook).
3. (Optional) **Enable GPU** for faster BiLSTM.
4. **Run All** cells.

### B) Run locally
```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
# Windows: py -m venv .venv && .venv\Scripts\activate

pip install -U numpy pandas scikit-learn nltk spacy tensorflow
python -m spacy download en_core_web_sm  # optional; notebook will fallback if missing

python - <<'PY'
import nltk
for p in ["stopwords","wordnet","punkt"]:
    nltk.download(p)
PY
```

Then open the notebook and **Run All**.

## Training & Cross‑Validation
- **TF‑IDF+LR**: word n‑grams (1–2) + char n‑grams (3–5), `C=~2.0`, `solver="saga"`, class_weight balanced.
- **BiLSTM**: vocab size 30k, max length 200, embedding 128, BiLSTM(64), global max pool, dense(64) + dropout.

You can adjust `max_features`, `ngram_range`, `C`, `MAX_LEN`, LSTM units, and epochs to trade off speed vs. accuracy.

## Inference
The notebook defines a **safe inference** helper that cleans text, uses **pre‑fit** TF‑IDF vectorizers with `.transform()`, and predicts with both pipelines:

```python
samples = [
    "I absolutely loved this movie, the performances were brilliant.",
    "Not good. I didn't enjoy it at all — boring and predictable."
]
cleaned, y_tfidf, y_lstm = predict_texts(samples)
```

- `y_tfidf`: predictions from TF‑IDF + Logistic Regression
- `y_lstm`: predictions from the BiLSTM (if the model is in scope)

## Metrics
- 5‑fold CV reported for both pipelines (mean Accuracy and F1).  
- Typical results: **Accuracy ≈ 0.90**, **F1 ≈ 0.88** (your exact numbers may differ based on environment and randomness).

## Error Analysis (Negation & Sarcasm)
We sample misclassified reviews and flag simple **negation** presence and **sarcasm** heuristics (quotes, “yeah right”, etc.). This helps pinpoint common failure modes.

## Reproducibility
We set a global `SEED=42` for numpy/tensorflow splits, but deep learning still has inherent non‑determinism across environments. For stricter reproducibility, pin library versions and set TF deterministic flags.

## Roadmap
- Persist artifacts (TF‑IDF vectorizers, LR model, tokenizer, BiLSTM weights) for deployment
- Add stronger deep baselines (e.g., DistilBERT)
- Expand error analysis (contrastive test sets, SHAP/attribution)
- Lightweight API demo (FastAPI/Streamlit) for interactive inference

## License
MIT — feel free to use and modify for your projects.

## Acknowledgements
- **IMDB 50K Movie Reviews** dataset (Kaggle)
- NLTK, spaCy, scikit‑learn, TensorFlow/Keras communities
