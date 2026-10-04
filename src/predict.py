"""
Inference and Prediction Module for Fake News Detection System.

This module:
1. Loads the serialized champion model and TF-IDF vectorizer.
2. Implements the core 'predict_news(text)' function.
3. Performs rigorous input validation and error handling.
4. Preprocesses new input with the exact same pipeline used in training.
5. Returns human-readable classifications (FAKE NEWS / REAL NEWS) along with
   calibrated model confidence percentages.
"""

import os
import sys
import argparse
import joblib
import numpy as np

# Ensure src can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import preprocess_text

# Default paths relative to project root
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, 'models', 'fake_news_model.pkl')
_DEFAULT_VEC_PATH = os.path.join(_BASE_DIR, 'models', 'tfidf_vectorizer.pkl')

_CACHED_MODEL = None
_CACHED_VECTORIZER = None


def load_artifacts(model_path: str = _DEFAULT_MODEL_PATH, vectorizer_path: str = _DEFAULT_VEC_PATH):
    """
    Loads and caches model and vectorizer to avoid redundant disk I/O.
    """
    global _CACHED_MODEL, _CACHED_VECTORIZER

    if _CACHED_MODEL is None or _CACHED_VECTORIZER is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                f"Please run 'python src/train_model.py' to generate trained weights."
            )
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"TF-IDF Vectorizer artifact not found at '{vectorizer_path}'. "
                f"Please run 'python src/train_model.py' to fit and save the vectorizer."
            )

        _CACHED_MODEL = joblib.load(model_path)
        _CACHED_VECTORIZER = joblib.load(vectorizer_path)

    return _CACHED_MODEL, _CACHED_VECTORIZER


def predict_news(
    text: str,
    model_path: str = _DEFAULT_MODEL_PATH,
    vectorizer_path: str = _DEFAULT_VEC_PATH
) -> dict:
    """
    Analyzes a raw news article or headline and classifies it as FAKE or REAL.

    Parameters:
    ----------
    text : str
        The raw article body or headline to evaluate.
    model_path : str, optional
        Path to the trained classifier .pkl file.
    vectorizer_path : str, optional
        Path to the fitted TF-IDF vectorizer .pkl file.

    Returns:
    -------
    dict:
        {
            "prediction": "FAKE NEWS" | "REAL NEWS",
            "class_id": 0 | 1,
            "confidence": float (0.0 to 1.0),
            "confidence_percentage": str (e.g. "94.2%"),
            "probabilities": {"Fake": float, "Real": float},
            "cleaned_text": str,
            "word_count": int,
            "warning": str or None,
            "error": str or None
        }
    """
    # 1. Input Validation
    if text is None or not isinstance(text, str):
        return {
            "prediction": "INVALID INPUT",
            "class_id": -1,
            "confidence": 0.0,
            "confidence_percentage": "0.0%",
            "probabilities": {"Fake": 0.0, "Real": 0.0},
            "cleaned_text": "",
            "word_count": 0,
            "warning": None,
            "error": "Input must be a non-empty string."
        }

    raw_text = text.strip()
    if not raw_text:
        return {
            "prediction": "EMPTY INPUT",
            "class_id": -1,
            "confidence": 0.0,
            "confidence_percentage": "0.0%",
            "probabilities": {"Fake": 0.0, "Real": 0.0},
            "cleaned_text": "",
            "word_count": 0,
            "warning": None,
            "error": "Please provide a valid news article or headline."
        }

    words = raw_text.split()
    word_count = len(words)
    warning = None
    if word_count < 10:
        warning = (
            f"Input is short ({word_count} words). While the model will still score it, "
            f"very short headlines provide limited linguistic context and higher uncertainty."
        )

    # 2. Text Preprocessing (Consistent with training)
    cleaned = preprocess_text(raw_text)
    if not cleaned:
        return {
            "prediction": "INSUFFICIENT TOKENS",
            "class_id": -1,
            "confidence": 0.0,
            "confidence_percentage": "0.0%",
            "probabilities": {"Fake": 0.0, "Real": 0.0},
            "cleaned_text": "",
            "word_count": word_count,
            "warning": None,
            "error": "After removing stop words and non-alphabetic characters, no meaningful tokens remained."
        }

    # 3. Vectorization
    model, vectorizer = load_artifacts(model_path, vectorizer_path)
    X_tfidf = vectorizer.transform([cleaned])

    # 4. Model Prediction & Confidence
    pred_class = int(model.predict(X_tfidf)[0])

    # Probability extraction (CalibratedClassifierCV, LogisticRegression, and MultinomialNB support predict_proba)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_tfidf)[0]
        prob_fake = float(probs[0])
        prob_real = float(probs[1])
        confidence = float(probs[pred_class])
    elif hasattr(model, 'decision_function'):
        df_score = model.decision_function(X_tfidf)[0]
        # Sigmoid conversion
        prob_real = float(1 / (1 + np.exp(-df_score)))
        prob_fake = 1.0 - prob_real
        confidence = prob_real if pred_class == 1 else prob_fake
    else:
        prob_fake = 1.0 if pred_class == 0 else 0.0
        prob_real = 1.0 if pred_class == 1 else 0.0
        confidence = 1.0

    label_str = "REAL NEWS" if pred_class == 1 else "FAKE NEWS"

    return {
        "prediction": label_str,
        "class_id": pred_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.1f}%",
        "probabilities": {
            "Fake": round(prob_fake, 4),
            "Real": round(prob_real, 4)
        },
        "cleaned_text": cleaned,
        "word_count": word_count,
        "warning": warning,
        "error": None
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a News Article")
    parser.add_argument('--text', type=str, required=True, help="News article or headline to test")
    parser.add_argument('--model_path', type=str, default=_DEFAULT_MODEL_PATH)
    parser.add_argument('--vectorizer_path', type=str, default=_DEFAULT_VEC_PATH)
    args = parser.parse_args()

    result = predict_news(args.text, args.model_path, args.vectorizer_path)
    if result["error"]:
        print(f"Error: {result['error']}")
        return

    print("=" * 60)
    print("NEWS PREDICTION RESULT")
    print("=" * 60)
    print(f"Input Text:     {args.text[:120]}..." if len(args.text) > 120 else f"Input Text:     {args.text}")
    print(f"Word Count:     {result['word_count']} words")
    print(f"Cleaned Text:   {result['cleaned_text'][:120]}...")
    print(f"Prediction:     >>> {result['prediction']} <<<")
    print(f"Confidence:     {result['confidence_percentage']}")
    print(f"Probabilities:  Fake: {result['probabilities']['Fake']:.2%}, Real: {result['probabilities']['Real']:.2%}")
    if result['warning']:
        print(f"Notice:         {result['warning']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
