"""
Script to generate notebooks/fake_news_detection.ipynb with complete data science workflow.
"""

import os
import json

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
notebook_path = os.path.join(base_dir, 'notebooks', 'fake_news_detection.ipynb')

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Fake News Detection Using Machine Learning\n",
            "### End-to-End Data Science & NLP Pipeline\n",
            "\n",
            "**Author:** Kota Adbuth Kumar  \n",
            "**Objective:** Build, evaluate, and deploy a machine learning classification system capable of analyzing news articles and detecting stylistic, lexical, and semantic patterns distinguishing **FAKE NEWS (0)** from **REAL NEWS (1)**.\n",
            "\n",
            "---\n",
            "\n",
            "### Machine Learning Workflow:\n",
            "```\n",
            "1. Data Collection & Loading (Fake.csv, True.csv)\n",
            "   ↓\n",
            "2. Data Cleaning & Schema Harmonization\n",
            "   ↓\n",
            "3. Exploratory Data Analysis (EDA & WordClouds)\n",
            "   ↓\n",
            "4. NLP Preprocessing (Tokenization, Negation-preserving Lemmatization)\n",
            "   ↓\n",
            "5. TF-IDF Feature Extraction (Unigram + Bigram)\n",
            "   ↓\n",
            "6. Model Training (Logistic Regression, Multinomial NB, Linear SVM)\n",
            "   ↓\n",
            "7. Model Evaluation & Benchmark Comparison (Accuracy, Precision, Recall, F1)\n",
            "   ↓\n",
            "8. Confusion Matrix & Error Tradeoff Deep Dive\n",
            "   ↓\n",
            "9. Model Serialization & Interactive Inference Test\n",
            "```"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Environment Setup & Library Imports"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import re\n",
            "import json\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from collections import Counter\n",
            "from wordcloud import WordCloud\n",
            "\n",
            "import nltk\n",
            "from nltk.corpus import stopwords\n",
            "from nltk.stem import WordNetLemmatizer\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.feature_extraction.text import TfidfVectorizer\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.naive_bayes import MultinomialNB\n",
            "from sklearn.svm import LinearSVC\n",
            "from sklearn.calibration import CalibratedClassifierCV\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, precision_score, recall_score, f1_score,\n",
            "    classification_report, confusion_matrix\n",
            ")\n",
            "\n",
            "# Download NLTK assets\n",
            "for res in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:\n",
            "    try:\n",
            "        nltk.data.find(f'corpora/{res}' if res in ['stopwords', 'wordnet'] else f'tokenizers/{res}')\n",
            "    except LookupError:\n",
            "        nltk.download(res, quiet=True)\n",
            "\n",
            "# Visual styling\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "plt.rcParams['figure.figsize'] = (9, 5)\n",
            "print('Libraries imported and NLTK assets ready!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Loading & Exploration\n",
            "We load the two datasets:\n",
            "- `Fake.csv`: Labeled as `0`\n",
            "- `True.csv`: Labeled as `1`"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fake_path = '../data/Fake.csv'\n",
            "true_path = '../data/True.csv'\n",
            "\n",
            "fake_df = pd.read_csv(fake_path)\n",
            "true_df = pd.read_csv(true_path)\n",
            "\n",
            "print(f'Fake news raw shape: {fake_df.shape}')\n",
            "print(f'True news raw shape: {true_df.shape}')\n",
            "\n",
            "# Assign ground truth target labels\n",
            "fake_df['label'] = 0\n",
            "true_df['label'] = 1\n",
            "\n",
            "# Concatenate title and text into a unified text feature\n",
            "fake_df['full_text'] = fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')\n",
            "true_df['full_text'] = true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')\n",
            "\n",
            "# Combine and shuffle with fixed random seed\n",
            "df = pd.concat([fake_df[['full_text', 'label']], true_df[['full_text', 'label']]], ignore_index=True)\n",
            "df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
            "df.rename(columns={'full_text': 'text'}, inplace=True)\n",
            "\n",
            "print(f'Combined shuffled dataset shape: {df.shape}')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Data Cleaning & Hygiene\n",
            "Check for missing values, duplicates, and non-informative rows."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Missing Values:\\n', df.isnull().sum())\n",
            "print(f'Duplicate records: {df.duplicated(subset=[\"text\"]).sum()}')\n",
            "\n",
            "# Remove nulls, short strings, and duplicates\n",
            "df = df.dropna(subset=['text', 'label'])\n",
            "df['text'] = df['text'].astype(str).str.strip()\n",
            "df = df[df['text'].str.len() > 15]\n",
            "df = df.drop_duplicates(subset=['text']).reset_index(drop=True)\n",
            "\n",
            "print(f'Remaining clean records: {len(df):,}')\n",
            "print('Class balance:\\n', df['label'].value_counts(normalize=True).round(3))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Exploratory Data Analysis (EDA)\n",
            "We inspect:\n",
            "1. Class distribution\n",
            "2. Article character length and word count distributions\n",
            "3. Top frequent vocabulary in Fake vs. Real news"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df['word_count'] = df['text'].apply(lambda x: len(x.split()))\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Class count\n",
            "sns.countplot(x='label', data=df, ax=axes[0], palette=['#EF4444', '#10B981'])\n",
            "axes[0].set_title('Fake vs Real News Distribution', fontsize=13, fontweight='bold')\n",
            "axes[0].set_xticklabels(['Fake (0)', 'Real (1)'])\n",
            "\n",
            "# Word count distribution (clipped to 1500 for visualization clarity)\n",
            "sns.histplot(data=df[df['word_count'] < 1500], x='word_count', hue='label', \n",
            "             kde=True, ax=axes[1], palette=['#EF4444', '#10B981'], bins=35)\n",
            "axes[1].set_title('Word Count Distribution by Class', fontsize=13, fontweight='bold')\n",
            "axes[1].legend(['Real (1)', 'Fake (0)'])\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Word Clouds: Real vs. Fake Articles"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fake_sample_text = ' '.join(df[df['label'] == 0]['text'].sample(2000, random_state=42))\n",
            "real_sample_text = ' '.join(df[df['label'] == 1]['text'].sample(2000, random_state=42))\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
            "\n",
            "wc_f = WordCloud(width=800, height=450, background_color='white', colormap='Reds', max_words=80).generate(fake_sample_text)\n",
            "axes[0].imshow(wc_f, interpolation='bilinear')\n",
            "axes[0].axis('off')\n",
            "axes[0].set_title('WordCloud: Fake News Vocabulary', fontsize=15, fontweight='bold')\n",
            "\n",
            "wc_r = WordCloud(width=800, height=450, background_color='white', colormap='Greens', max_words=80).generate(real_sample_text)\n",
            "axes[1].imshow(wc_r, interpolation='bilinear')\n",
            "axes[1].axis('off')\n",
            "axes[1].set_title('WordCloud: Real News Vocabulary', fontsize=15, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Robust Text Preprocessing Pipeline\n",
            "\n",
            "Why is text preprocessing necessary?\n",
            "- Computers cannot directly calculate loss functions on raw character strings.\n",
            "- Uncleaned text contains noise (URLs, HTML tags, punctuation, inconsistent casing).\n",
            "- Standard stopword lists inadvertently remove negations (`not`, `no`, `never`). For fake news, saying `\"minister did commit fraud\"` versus `\"minister did not commit fraud\"` has opposite truth claims. We preserve negations to protect semantic meaning."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "NEGATIONS = {'not', 'no', 'never', 'nor', 'neither', 'barely', 'scarcely', 'hardly'}\n",
            "CUSTOM_STOPWORDS = set(stopwords.words('english')) - NEGATIONS\n",
            "LEMMATIZER = WordNetLemmatizer()\n",
            "\n",
            "def preprocess_text(text: str) -> str:\n",
            "    if not isinstance(text, str) or not text.strip():\n",
            "        return ''\n",
            "    # Lowercase\n",
            "    t = text.lower()\n",
            "    # Strip URLs\n",
            "    t = re.sub(r'https?://\\S+|www\\.\\S+', ' ', t)\n",
            "    # Strip HTML tags\n",
            "    t = re.sub(r'<.*?>', ' ', t)\n",
            "    # Strip publisher dateline artifacts (e.g. reuters)\n",
            "    t = re.sub(r'^[a-z\\s]+reuters\\s*[-–—]?\\s*', ' ', t)\n",
            "    # Remove punctuation & non-alphabetic chars\n",
            "    t = re.sub(r'[^a-z\\s]', ' ', t)\n",
            "    tokens = t.split()\n",
            "    # Lemmatize and filter\n",
            "    cleaned = [LEMMATIZER.lemmatize(w) for w in tokens if w not in CUSTOM_STOPWORDS and len(w) > 2]\n",
            "    return ' '.join(cleaned)\n",
            "\n",
            "# Demonstrate preprocessing\n",
            "sample_raw = 'BREAKING: Official claims NASA did not discover extraterrestrial bacteria! See https://example.com/live <p>Report</p>'\n",
            "print('Raw:    ', sample_raw)\n",
            "print('Cleaned:', preprocess_text(sample_raw))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Preprocessing Dataset & Stratified Train/Test Split"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# For rapid demonstration in notebook, we use 15,000 balanced records\n",
            "subset_df = df.sample(n=min(15000, len(df)), random_state=42).copy()\n",
            "print('Applying preprocessing to subset...')\n",
            "subset_df['cleaned_text'] = subset_df['text'].apply(preprocess_text)\n",
            "subset_df = subset_df[subset_df['cleaned_text'].str.strip().str.len() > 5].reset_index(drop=True)\n",
            "\n",
            "X = subset_df['cleaned_text']\n",
            "y = subset_df['label']\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y,\n",
            "    test_size=0.20,\n",
            "    random_state=42,\n",
            "    stratify=y\n",
            ")\n",
            "\n",
            "print(f'Training samples: {len(X_train):,}')\n",
            "print(f'Test samples:     {len(X_test):,}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. TF-IDF Feature Extraction\n",
            "\n",
            "**What is TF-IDF?**\n",
            "- **Term Frequency (TF):** Measures how frequently a word occurs in a document.\n",
            "- **Inverse Document Frequency (IDF):** Penalizes words that appear ubiquitously across all documents (like `said`, `would`), elevating discriminative keywords (like `conspiracy`, `spokesperson`).\n",
            "- **N-grams (1, 2):** Captures single words (unigrams) as well as pairs of consecutive words (bigrams like `white house`, `fake news`, `climate change`).\n",
            "\n",
            "> **Crucial Anti-Leakage Rule:** We `fit_transform` ONLY on `X_train`, and `transform` on `X_test`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "vectorizer = TfidfVectorizer(\n",
            "    max_features=10000,\n",
            "    ngram_range=(1, 2),\n",
            "    stop_words='english',\n",
            "    sublinear_tf=True,\n",
            "    norm='l2'\n",
            ")\n",
            "\n",
            "# Fit strictly on training set\n",
            "X_train_tfidf = vectorizer.fit_transform(X_train)\n",
            "X_test_tfidf = vectorizer.transform(X_test)\n",
            "\n",
            "print('TF-IDF Train matrix shape:', X_train_tfidf.shape)\n",
            "print('TF-IDF Test matrix shape: ', X_test_tfidf.shape)\n",
            "print('Sample vocabulary features:', vectorizer.get_feature_names_out()[:15])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Model Training & Comparison\n",
            "We train and compare three classical Machine Learning architectures:\n",
            "1. **Logistic Regression:** Linear probabilistic boundary via sigmoid function.\n",
            "2. **Multinomial Naive Bayes:** Probabilistic model based on Bayes' theorem under naive feature independence assumption.\n",
            "3. **Linear Support Vector Machine (Linear SVM):** Finds maximum-margin hyperplane separating classes in high-dimensional TF-IDF space (wrapped in `CalibratedClassifierCV` for valid probability outputs)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
            "    'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),\n",
            "    'Linear SVM': CalibratedClassifierCV(\n",
            "        estimator=LinearSVC(C=1.0, random_state=42, dual=False),\n",
            "        cv=3\n",
            "    )\n",
            "}\n",
            "\n",
            "results = {}\n",
            "predictions = {}\n",
            "\n",
            "for name, clf in models.items():\n",
            "    clf.fit(X_train_tfidf, y_train)\n",
            "    y_pred = clf.predict(X_test_tfidf)\n",
            "    predictions[name] = y_pred\n",
            "    \n",
            "    acc = accuracy_score(y_test, y_pred)\n",
            "    prec = precision_score(y_test, y_pred, average='weighted')\n",
            "    rec = recall_score(y_test, y_pred, average='weighted')\n",
            "    f1 = f1_score(y_test, y_pred, average='weighted')\n",
            "    \n",
            "    results[name] = {\n",
            "        'Accuracy': round(acc, 4),\n",
            "        'Precision': round(prec, 4),\n",
            "        'Recall': round(rec, 4),\n",
            "        'F1 Score': round(f1, 4)\n",
            "    }\n",
            "\n",
            "# Summary Table\n",
            "benchmark_df = pd.DataFrame(results).T\n",
            "display(benchmark_df)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Model Evaluation Visualizations & Confusion Matrix"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Bar chart of model performance\n",
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            "benchmark_df.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', alpha=0.9)\n",
            "ax.set_ylim(0.85, 1.01)\n",
            "ax.set_title('Performance Comparison of ML Models', fontsize=14, fontweight='bold', pad=12)\n",
            "ax.set_ylabel('Metric Score')\n",
            "plt.xticks(rotation=0)\n",
            "plt.legend(loc='lower right')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Confusion Matrix Deep Dive\n",
            "We inspect the confusion matrix of the champion model (Linear SVM):\n",
            "- **True Negative (TN):** Fake news correctly identified as Fake.\n",
            "- **False Positive (FP):** Fake news falsely classified as Real (misinformation slips through).\n",
            "- **False Negative (FN):** Real news falsely classified as Fake (legitimate news discredited).\n",
            "- **True Positive (TP):** Real news correctly identified as Real."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "best_model_name = 'Linear SVM'\n",
            "cm = confusion_matrix(y_test, predictions[best_model_name])\n",
            "\n",
            "fig, ax = plt.subplots(figsize=(6, 5))\n",
            "sns.heatmap(\n",
            "    cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n",
            "    xticklabels=['Predicted Fake (0)', 'Predicted Real (1)'],\n",
            "    yticklabels=['Actual Fake (0)', 'Actual Real (1)']\n",
            ")\n",
            "plt.title(f'Confusion Matrix: {best_model_name}', fontsize=13, fontweight='bold', pad=12)\n",
            "plt.ylabel('Actual Label')\n",
            "plt.xlabel('Predicted Label')\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print('Classification Report:\\n')\n",
            "print(classification_report(y_test, predictions[best_model_name], target_names=['Fake (0)', 'Real (1)']))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 10. Serializing Model & Vectorizer"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "os.makedirs('../models', exist_ok=True)\n",
            "joblib.dump(models[best_model_name], '../models/fake_news_model.pkl')\n",
            "joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')\n",
            "print('Model and Vectorizer persisted to ../models/ directory!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 11. Interactive Prediction Function & Testing"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "loaded_model = joblib.load('../models/fake_news_model.pkl')\n",
            "loaded_vec = joblib.load('../models/tfidf_vectorizer.pkl')\n",
            "\n",
            "def predict_article(text: str):\n",
            "    cleaned = preprocess_text(text)\n",
            "    if not cleaned:\n",
            "        return 'Insufficient text content.'\n",
            "    vec_text = loaded_vec.transform([cleaned])\n",
            "    pred = loaded_model.predict(vec_text)[0]\n",
            "    probs = loaded_model.predict_proba(vec_text)[0]\n",
            "    \n",
            "    label = 'REAL NEWS' if pred == 1 else 'FAKE NEWS'\n",
            "    confidence = probs[pred]\n",
            "    return {\n",
            "        'Prediction': label,\n",
            "        'Confidence': f'{confidence:.1%}',\n",
            "        'Probabilities': {'Fake': round(probs[0], 3), 'Real': round(probs[1], 3)}\n",
            "    }\n",
            "\n",
            "# Test 1: Real News Excerpt\n",
            "sample_real = 'GENEVA - The World Health Organization reported a notable decline in seasonal influenza cases following enhanced public vaccination programs.'\n",
            "print('Sample Real Article Test:', predict_article(sample_real))\n",
            "\n",
            "# Test 2: Fake News Excerpt\n",
            "sample_fake = 'BREAKING BOMBSHELL: Alien mothership discovered frozen in Antarctic glaciers, world governments suppress evidence!'\n",
            "print('Sample Fake Article Test:', predict_article(sample_fake))"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.13"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=2)

print(f"Jupyter Notebook successfully written to: {notebook_path}")
