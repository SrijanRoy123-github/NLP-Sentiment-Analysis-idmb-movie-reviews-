I built an IMDB movie review sentiment classifier using a **dual-pipeline approach** that combines classical machine learning and deep learning. The project uses **NLTK and spaCy** for robust text preprocessing (cleaning, tokenization, lemmatization) while deliberately preserving **negation tokens** to retain sentiment cues. I train and evaluate two complementary models: **TF-IDF (word + character n-grams) with Logistic Regression** and a **Keras BiLSTM** operating on padded token sequences. Both pipelines are validated using **5-fold cross-validation**, achieving typical performance of **~0.90 accuracy and ~0.88 F1 score** (dataset and environment dependent). I also include a **safe inference helper** that avoids TF-IDF fitting issues and a brief **error analysis** highlighting common failure modes related to negation and simple sarcasm, making the repository a clean, reproducible reference for practical sentiment analysis workflows.


## Acknowledgements
- **IMDB 50K Movie Reviews** dataset (Kaggle)
- NLTK, spaCy, scikit‑learn, TensorFlow/Keras communities
