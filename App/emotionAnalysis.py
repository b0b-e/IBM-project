"""Retrain and verify script

This script trains a TF-IDF + Logistic Regression model on the Dataset
and saves model artifacts to the sibling `Model/` directory.

Structure:
  - Imports
  - Configuration (paths + constants)
  - Utility text-processing functions
  - Data loading and preprocessing
  - Feature extraction (TF-IDF)
  - Model training and evaluation
  - Saving artifacts and optional quick verification
"""

##########################################################################
# Imports
##########################################################################
import os
import re
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


##########################################################################
# Configuration
##########################################################################
# Resolve dataset and model directories relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'Dataset'
MODEL_DIR = PROJECT_ROOT / 'Model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Vectorizer / model hyperparameters
MAX_FEAT = 12000


##########################################################################
# Utility functions: text cleaning and small NLP helpers
##########################################################################
def clean_text(s: str) -> str:
    """Basic cleaning: remove URLs, normalize quotes, keep letters/numbers/hyphens."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r"’", "'", s)  # normalize curly apostrophes
    s = re.sub(r"[^A-Za-z0-9\s'\-]", ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()


def expand_contractions(text: str) -> str:
    """Replace common contractions with expanded forms (small set)."""
    contractions = {
        "can't": "can not", "won't": "will not", "n't": " not",
        "i'm": "i am", "it's": "it is", "that's": "that is",
        "i've": "i have", "i'd": "i would", "you're": "you are"
    }
    txt = text
    for k, v in contractions.items():
        txt = re.sub(r'\b' + re.escape(k) + r'\b', v, txt)
    return txt


def handle_negations(text: str) -> str:
    """Transform negations to tokens like `not_word` for a lightweight negation feature."""
    text = expand_contractions(text)
    words = text.split()
    new_words = []
    neg = False
    # Use double-quoted string for won't to avoid accidental parsing issues
    negation_tokens = {'not', 'no', 'never', 'cannot', 'cant', "won't"}
    for w in words:
        lw = w.lower()
        if neg:
            new_words.append("not_" + lw)
            neg = False
            continue
        if lw in negation_tokens or lw.endswith("n't") or lw == "not":
            neg = True
            continue
        new_words.append(lw)
    return " ".join(new_words)


##########################################################################
# Main flow: data load -> preprocess -> train -> save
##########################################################################
def main():
    # --- Load raw data ---
    print("Loading data...")
    train = pd.read_csv(str(DATA_DIR / 'train.txt'), sep=';', names=['text', 'emotion'], encoding='utf-8', engine='python')
    test = pd.read_csv(str(DATA_DIR / 'test.txt'), sep=';', names=['text', 'emotion'], encoding='utf-8', engine='python')
    val = pd.read_csv(str(DATA_DIR / 'val.txt'), sep=';', names=['text', 'emotion'], encoding='utf-8', engine='python')

    df = pd.concat([train, test, val]).reset_index(drop=True)
    print("Total samples:", len(df))
    print(df['emotion'].value_counts())

    # --- Preprocess text ---
    print("Cleaning text and applying negation handling...")
    df['text'] = df['text'].astype(str).apply(clean_text)
    df['text'] = df['text'].apply(handle_negations)

    # quick sanity examples
    print("\nSample transformations:")
    for sample in ["not good", "not happy", "no way this is fine", "that's insane", "i'm not sure"]:
        print(sample, "->", handle_negations(clean_text(sample)))

    # --- Encode labels ---
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])
    print("\nClasses:", list(le.classes_))

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    # --- Vectorize ---
    print("\nVectorizing with TF-IDF (1-2 grams, sublinear_tf)...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1, 2), sublinear_tf=True, norm='l2')
    X_tr_vec = vectorizer.fit_transform(X_train)
    X_te_vec = vectorizer.transform(X_test)

    # check for negation markers in vocabulary
    vocab = set(vectorizer.get_feature_names_out())
    checks = ['not_good', 'not_happy', 'no_way', 'never_again', 'not_bad']
    print("\nNegation tokens present in vocabulary:")
    for token in checks:
        # Avoid nested braces inside f-strings; use format for clarity
        print("  {}: {}".format(token, 'YES' if token in vocab else 'NO'))

    # --- Train model ---
    print("\nTraining LogisticRegression (max_iter=2000, class_weight='balanced') ...")
    model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga')
    model.fit(X_tr_vec, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_te_vec)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # --- Save artifacts ---
    print("\nSaving model artifacts to", MODEL_DIR)
    joblib.dump(model, str(MODEL_DIR / 'emotion_model.pkl'))
    joblib.dump(vectorizer, str(MODEL_DIR / 'tfidf_vectorizer.pkl'))
    joblib.dump(le, str(MODEL_DIR / 'label_encoder.pkl'))
    print("Saved.")

        # --- Quick verification ---
    print("\n--- Manual sanity check ---")
    test_samples = [
        "not good",
        "not happy today",
        "I am not excited",
        "no way this is fine",
        "never again",
        "I can't believe this happened!",
        "I am terrified of tomorrow",
        "I love spending time with you",
        "I’m furious right now",
        "Everything feels hopeless",
        "That's insane, how can that happen?"
    ]

    cleaned = [handle_negations(clean_text(t)) for t in test_samples]
    X_check = vectorizer.transform(cleaned)
    preds = model.predict(X_check)
    probs = model.predict_proba(X_check)

    for text, pred_idx, prob_row in zip(test_samples, preds, probs):
        label = le.inverse_transform([pred_idx])[0]
        conf = round(max(prob_row) * 100, 2)
        print(f"{text:<45} -> {label.upper():<10} ({conf}% confidence)")


if __name__ == '__main__':
    main()

print("All done.")