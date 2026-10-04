"""
Model Training Module for Fake News Detection System.

This module:
1. Loads and preprocesses news data.
2. Performs stratified train/test split (80/20) to ensure balanced representation.
3. Extracts TF-IDF features (unigrams and bigrams) without data leakage.
4. Trains three distinct machine learning classifiers:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Linear Support Vector Machine (with probability calibration)
5. Evaluates model performance using Accuracy, Precision, Recall, and F1-score.
6. Automatically selects the best-performing model based on test F1-score.
7. Serializes the champion model and vectorizer with joblib.
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add parent directory to path to import src modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_preprocessing import load_and_combine_data, clean_dataset, preprocess_text


def build_models():
    """
    Initializes the dictionary of candidate models.
    LinearSVC is wrapped in CalibratedClassifierCV to provide reliable probability estimates.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.1,
            fit_prior=True
        ),
        "Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, dual=False),
            method='sigmoid',
            cv=3
        )
    }


def train_and_evaluate(
    data_path: str = None,
    fake_path: str = 'data/Fake.csv',
    true_path: str = 'data/True.csv',
    sample_size: int = 15000,
    max_features: int = 10000,
    models_dir: str = 'models',
    random_state: int = 42
):
    """
    Full training pipeline from data loading to artifact persistence.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir_path = os.path.join(base_dir, models_dir)
    os.makedirs(models_dir_path, exist_ok=True)

    # 1. Load Data
    print("=" * 60)
    print("STEP 1: DATA LOADING & PREPARATION")
    print("=" * 60)
    
    # Check if pre-processed data exists, otherwise load raw files
    proc_path = os.path.join(base_dir, 'data', 'processed_data.csv')
    if data_path and os.path.exists(data_path):
        print(f"Loading provided dataset from: {data_path}")
        df = pd.read_csv(data_path)
    elif os.path.exists(proc_path):
        print(f"Found existing processed dataset at: {proc_path}")
        df = pd.read_csv(proc_path)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            print(f"Subsampled to {sample_size} records.")
    else:
        fake_p = os.path.join(base_dir, fake_path)
        true_p = os.path.join(base_dir, true_path)
        print(f"Loading raw files: {fake_p} & {true_p}")
        df = load_and_combine_data(fake_p, true_p, sample_size=sample_size, random_state=random_state)
        df = clean_dataset(df)
        print("Preprocessing raw text...")
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        df = df[df['cleaned_text'].str.strip().str.len() > 5].copy()

    # Use 'cleaned_text' if present, otherwise 'text'
    text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    X = df[text_column].astype(str)
    y = df['label'].astype(int)

    print(f"Final training corpus size: {len(df)} samples")
    print(f"Class counts: Fake (0) = {(y == 0).sum()}, Real (1) = {(y == 1).sum()}")

    # 2. Train / Test Split (Stratified 80/20)
    print("\n" + "=" * 60)
    print("STEP 2: STRATIFIED TRAIN / TEST SPLIT (80/20)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )
    print(f"Training set:   {X_train.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")

    # 3. TF-IDF Feature Extraction (No Data Leakage)
    print("\n" + "=" * 60)
    print("STEP 3: TF-IDF FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Configuring TfidfVectorizer (max_features={max_features}, ngram_range=(1,2), sublinear_tf=True)...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
        norm='l2'
    )

    # Fit strictly on X_train to prevent test data leakage
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
    print(f"TF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}")

    # 4. Model Training & Comparative Evaluation
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    models = build_models()
    results = {}
    classification_reports = {}
    confusion_matrices = {}

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train_tfidf, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test_tfidf)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1 Score": round(float(f1), 4),
            "Confusion Matrix": cm
        }
        classification_reports[name] = classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)'], output_dict=True)

        print(f"{name} Results:")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1 Score:  {f1 * 100:.2f}%")

    # 5. Display Model Comparison Table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY TABLE")
    print("=" * 60)
    summary_df = pd.DataFrame(results).T[['Accuracy', 'Precision', 'Recall', 'F1 Score']]
    print(summary_df.to_string())

    # 6. Select Champion Model
    best_model_name = max(results, key=lambda k: (results[k]['F1 Score'], results[k]['Accuracy']))
    best_model = models[best_model_name]
    print(f"\nChampion Model Selected: '{best_model_name}' (F1 Score: {results[best_model_name]['F1 Score']})")

    # 7. Persist Artifacts
    print("\n" + "=" * 60)
    print("STEP 5: PERSISTING MODEL ARTIFACTS")
    print("=" * 60)
    model_save_path = os.path.join(models_dir_path, 'fake_news_model.pkl')
    vec_save_path = os.path.join(models_dir_path, 'tfidf_vectorizer.pkl')
    metrics_save_path = os.path.join(models_dir_path, 'model_metrics.json')

    joblib.dump(best_model, model_save_path)
    joblib.dump(vectorizer, vec_save_path)

    metrics_payload = {
        "champion_model": best_model_name,
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "ngram_range": [1, 2],
        "models": results,
        "detailed_reports": classification_reports
    }

    with open(metrics_save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"Saved Champion Model:    {model_save_path}")
    print(f"Saved TF-IDF Vectorizer: {vec_save_path}")
    print(f"Saved Metrics & Summary: {metrics_save_path}")
    print("\n[SUCCESS] Pipeline training complete!")

    return best_model, vectorizer, results


def main():
    parser = argparse.ArgumentParser(description="Train Fake News ML Models")
    parser.add_argument('--data_path', type=str, default=None, help='Path to pre-processed CSV')
    parser.add_argument('--sample_size', type=int, default=15000, help='Number of records to use (e.g. 15000 or 0 for all)')
    parser.add_argument('--max_features', type=int, default=10000, help='Max TF-IDF features')
    args = parser.parse_args()

    sample_val = args.sample_size if args.sample_size > 0 else None
    train_and_evaluate(data_path=args.data_path, sample_size=sample_val, max_features=args.max_features)


if __name__ == '__main__':
    main()
