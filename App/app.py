from flask import Flask, render_template, request
import pickle
import joblib
from pathlib import Path

import re

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r"’", "'", s)
    s = re.sub(r"[^A-Za-z0-9\s'\-]", ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def expand_contractions(text):
    contractions = {
        "can't": "can not", "won't": "will not", "n't": " not",
        "i'm": "i am", "it's": "it is", "that's": "that is",
        "i've": "i have", "i'd": "i would", "you're": "you are"
    }
    txt = text
    for k, v in contractions.items():
        txt = re.sub(r'\b' + re.escape(k) + r'\b', v, txt)
    return txt

def handle_negations(text):
    text = expand_contractions(text)
    words = text.split()
    new_words = []
    neg = False
    negation_tokens = {'not', 'no', 'never', 'cannot', 'cant'}
    for w in words:
        lw = w.lower()
        if neg:
            new_words.append("not_" + lw)
            neg = False
            continue
        if lw in negation_tokens or lw.endswith("n't"):
            neg = True
            continue
        new_words.append(lw)
    return " ".join(new_words)

def preprocess(text):
    text = clean_text(text)
    text = handle_negations(text)
    return text


app = Flask(__name__)

import os
print("Loading model from:", os.path.abspath("../Model/emotion_model.pkl"))


# Resolve paths: Model directory is the sibling 'Model' folder of this App folder
# Use parents[1] to robustly resolve one level above the App folder (the project root)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(__file__).resolve().parents[1] / 'Model'

def load_pickle(filename):
    """Try joblib.load (preferred for sklearn artifacts) then fallback to pickle.load."""
    path = MODEL_DIR / filename
    try:
        return joblib.load(path)
    except Exception as e:
        # fallback to pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

try:
    model = load_pickle('emotion_model.pkl')
    vectorizer = load_pickle('tfidf_vectorizer.pkl')
    label_encoder = load_pickle('label_encoder.pkl')
except Exception as e:
    # Print to console for local debugging; templates will show a friendly error if used before models are present
    print('Error loading model files:', e)
    model = None
    vectorizer = None
    label_encoder = None

# Map emotions to emojis
EMOJI_MAP = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😡',
    'fear': '😨',
    'love': '❤️',
    'surprise': '😲'
}


@app.route('/', methods=['GET'])
def index():
    """Render the main page. Optional query values filled when prediction available."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    text = text.strip()
    if not text:
        return render_template('index.html', error='Please enter some text to analyze.')

    if model is None:
        return render_template('index.html', error='Model files not found or failed to load on server.')

    try:
        # Some models expect raw text (e.g., pipeline). Try vectorizing first; if it fails, pass raw text.
        try:
            processed = preprocess(text)
            X = vectorizer.transform([processed])

            pred = model.predict(X)
        except Exception:
            pred = model.predict([text])

        # pred can be ndarray-like. Normalize to a single label string.
        label = None
        try:
            # If label_encoder exists, use it to decode numeric labels
            if label_encoder is not None:
                # ensure pred is a 1D array-like
                label = label_encoder.inverse_transform(pred)[0]
            else:
                # model may already return string labels
                label = pred[0] if hasattr(pred, '__getitem__') else str(pred)
        except Exception:
            # Fallback: convert first element to string
            try:
                label = str(pred[0])
            except Exception:
                label = str(pred)

        display_label = str(label)
        emoji = EMOJI_MAP.get(display_label.lower(), '')

        # Attempt to get confidence score via predict_proba if available.
        confidence_str = ''
        try:
            if hasattr(model, 'predict_proba'):
                # If we vectorized above, X is available; otherwise recreate input for predict_proba
                try:
                    probs = model.predict_proba(X)
                except Exception:
                    probs = model.predict_proba([text])
                # take the highest class probability for the first (and only) sample
                top_prob = float(probs[0].max())
                confidence_pct = round(top_prob * 100.0, 2)
                confidence_str = f" ({confidence_pct:.2f}% confidence)"
        except Exception:
            # If predict_proba exists but fails for some reason, gracefully ignore confidence
            confidence_str = ''

        # Build a friendly prediction string that includes the emoji and confidence
        prediction_text = f"Emotion: {display_label} {emoji}{confidence_str}"

        return render_template('index.html', prediction=prediction_text, emoji=emoji, text=text)

    except Exception as e:
        return render_template('index.html', error=f'Prediction error: {e}')


if __name__ == '__main__':
    # Debug mode enabled for local testing
    app.run(debug=True)
