# Fake News Detection Using Machine Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Data Science and Natural Language Processing (NLP) system that analyzes the linguistic structure, vocabulary patterns, and stylistic signals in news articles and headlines to predict whether an article is **FAKE NEWS (0)** or **REAL NEWS (1)**.

---

## Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Technology Stack](#3-technology-stack)
- [4. Dataset & Ground Truth](#4-dataset--ground-truth)
- [5. Machine Learning Pipeline](#5-machine-learning-pipeline)
- [6. Installation & Environment Setup](#6-installation--environment-setup)
- [7. How to Train the Models](#7-how-to-train-the-models)
- [8. Running the Streamlit Web Application](#8-running-the-streamlit-web-application)
- [9. Visualizations & Evaluation Artifacts](#9-visualizations--evaluation-artifacts)
- [10. Empirical Results & Benchmark Comparison](#10-empirical-results--benchmark-comparison)
- [11. Confusion Matrix & Error Tradeoffs](#11-confusion-matrix--error-tradeoffs)
- [12. Limitations & Ethical Considerations](#12-limitations--ethical-considerations)
- [13. Future Enhancements & Portfolio Roadmap](#13-future-enhancements--portfolio-roadmap)

---

## 1. Project Overview

The viral propagation of digitally fabricated news poses significant challenges to public discourse, democratic processes, public health, and financial markets. 

While manual fact-checking by journalistic institutions remains indispensable, human verification cannot scale to match the velocity of online content generation. This project implements a **computational machine learning pipeline** that learns discriminating stylistic, syntactic, and semantic signatures between authentic journalism and fabricated sensationalism.

The application follows production best practices:
- **Clean modular software architecture** separating preprocessing, model training, evaluation, and inference.
- **Strict anti-leakage feature extraction** where vectorizers are fitted solely on training splits.
- **Calibrated probability outputs** so predictions reflect genuine statistical confidence.
- **Interactive Streamlit Web Interface** with input validation, sample demonstrations, and preprocessing inspection.

---

## 2. Key Features

- **End-to-End Workflow:** From raw CSV ingestion to interactive web deployment.
- **Robust Text Preprocessing:** Regex-based cleaning, HTML/URL stripping, publisher dateline neutralization, negation-preserving stopword filtering, and WordNet lemmatization.
- **N-Gram TF-IDF Vectorization:** Captures unigrams and bigrams (e.g., `climate change`, `white house`) with sublinear term-frequency scaling.
- **Multi-Model Benchmark:** Trains and rigorously compares **Logistic Regression**, **Multinomial Naive Bayes**, and **Linear Support Vector Machine (Linear SVM)**.
- **Probability Calibration:** Wraps maximum-margin SVM in Platt scaling (`CalibratedClassifierCV`) to deliver reliable confidence scores alongside binary labels.
- **Interactive Streamlit Web Application:** Allows users to paste live news text, test pre-loaded demonstrations, view probability distributions, and review extracted token sequences.

---

## 3. Technology Stack

| Layer | Tools & Libraries |
| :--- | :--- |
| **Language** | Python 3 (Python 3.10 – 3.13) |
| **Data Manipulation** | Pandas, NumPy |
| **Natural Language Processing** | NLTK (Tokenizers, WordNet Lemmatizer, Stopwords) |
| **Feature Extraction** | Scikit-Learn `TfidfVectorizer` |
| **Machine Learning Algorithms** | Logistic Regression, Multinomial Naive Bayes, Linear Support Vector Classifier (`LinearSVC`) |
| **Calibration & Evaluation** | `CalibratedClassifierCV`, Scikit-Learn Metrics (`classification_report`, `confusion_matrix`) |
| **Visualization & Reporting** | Matplotlib, Seaborn, WordCloud |
| **Model Persistence** | Joblib |
| **Web Deployment** | Streamlit |

---

## 4. Dataset & Ground Truth

The system is trained on the benchmark **ISOT Fake and Real News Dataset** (University of Victoria / Kaggle):
- **`Fake.csv`**: Contains fabricated articles flagged by Politifact and independent watchdog organizations.
- **`True.csv`**: Contains factual, verified real-world news dispatches collected from Reuters.

### Ground Truth Label Encoding
```text
Fake News = 0
Real News = 1
```

### Dataset Schema
| Column | Description | Type |
| :--- | :--- | :--- |
| `title` | Article headline | String |
| `text` | Main body text | String |
| `label` | Target classification (`0` = Fake, `1` = Real) | Integer |

*Note: The dataset loader in `src/data_preprocessing.py` supports configurable column mappings to effortlessly ingest external datasets.*

---

## 5. Machine Learning Pipeline

```text
               +--------------------------------------+
               | Raw Datasets (Fake.csv and True.csv) |
               +--------------------------------------+
                                   |
                                   v
               +--------------------------------------+
               |    Data Hygiene & Merging            |
               | (Null removal, Deduplication, Merge) |
               +--------------------------------------+
                                   |
                                   v
               +--------------------------------------+
               |    NLP Preprocessing Engine          |
               | - Lowercase & URL / HTML stripping   |
               | - Dateline & punctuation removal     |
               | - Negation-preserving stopwords      |
               | - WordNet Lemmatization              |
               +--------------------------------------+
                                   |
                                   v
               +--------------------------------------+
               |   Stratified Train / Test Split      |
               |           (80% Train, 20% Test)      |
               +--------------------------------------+
                                   |
                                   v
               +--------------------------------------+
               |   TF-IDF N-Gram Vectorizer           |
               | (Fitted exclusively on X_train)      |
               +--------------------------------------+
                                   |
                                   v
               +--------------------------------------+
               |      Model Training & Tuning         |
               |  1. Logistic Regression              |
               |  2. Multinomial Naive Bayes          |
               |  3. Calibrated Linear SVM            |
               +--------------------------------------+
                                   |
                                   v
               +--------------------------------------+
               |   Comparative Evaluation & Audit     |
               |  (Accuracy, Precision, Recall, F1)   |
               +--------------------------------------+
                                   |
                                   v
               +--------------------------------------+
               |     Champion Model Selection         |
               |    (models/fake_news_model.pkl)      |
               +--------------------------------------+
                                   |
                                   v
               +--------------------------------------+
               |    Streamlit Web Application         |
               |     (Interactive Live Inference)     |
               +--------------------------------------+
```

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### Step 2: Create and Activate Virtual Environment

**On Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade Pip & Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. How to Train the Models

### Run Data Preprocessing (Optional Standalone)
Preprocesses raw articles and outputs `data/processed_data.csv`:
```bash
python src/data_preprocessing.py --sample_size 16000
```

### Train All Candidate Models
Trains Logistic Regression, Multinomial Naive Bayes, and Calibrated Linear SVM, displays the comparison table, and serializes the champion model:
```bash
python src/train_model.py --sample_size 15000 --max_features 10000
```

### Generate Detailed Evaluation Reports & Plots
Produces confusion matrix heatmaps and comparison charts in `reports/`:
```bash
python src/evaluate_model.py
```

### Test CLI Single Article Inference
```bash
python src/predict.py --text "WASHINGTON - Officials at the Department of Energy announced new funding for renewable solar grids across five states."
```

---

## 8. Running the Streamlit Web Application

Launch the browser-based dashboard:
```bash
streamlit run app.py
```

The web dashboard will automatically open in your browser at `http://localhost:8501`.

### App Capabilities:
1. **Interactive Text Box:** Paste any news body or headline.
2. **One-Click Demos:** Click **"Load Example Fake News"** or **"Load Example Real News"** to test model behavior instantly.
3. **Linguistic Token Inspection:** Expand the token inspector to see what words survived preprocessing.
4. **Confidence Gauge:** Visual progress bars displaying the class probabilities.
5. **Input Validation:** Clear advisories when text is too short or empty.

---

## 9. Visualizations & Evaluation Artifacts

All charts generated during execution are stored in `reports/`:

| Visualization | Artifact File | Description |
| :--- | :--- | :--- |
| **Model Comparison** | `reports/model_comparison.png` | Grouped bar chart comparing Accuracy, Precision, Recall, and F1. |
| **Confusion Matrix** | `reports/confusion_matrix.png` | Annotated heatmap detailing TP, TN, FP, and FN distribution. |
| **Class Balance** | `reports/class_distribution.png` | Distribution of Fake vs. Real articles in the corpus. |
| **Word Counts** | `reports/word_count_distribution.png` | Kernel Density Estimation of article length across classes. |
| **Word Frequency** | `reports/top_words_comparison.png` | Top 12 most frequent terms in Fake vs. Real news. |
| **WordClouds** | `reports/wordcloud_fake_vs_real.png` | High-resolution side-by-side lexical cloud. |

---

## 10. Empirical Results & Benchmark Comparison

The following results were empirically obtained during pipeline execution on **3,057 held-out test articles** (from a balanced corpus of 15,282 articles):

| Model Architecture | Accuracy | Precision | Recall | F1 Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Support Vector Machine (Linear SVM)** | **98.76%** | **98.76%** | **98.76%** | **0.9876** | 🏆 **Champion** |
| **Logistic Regression** | 97.91% | 97.91% | 97.91% | 0.9791 | Runner-Up |
| **Multinomial Naive Bayes** | 94.67% | 94.68% | 94.67% | 0.9467 | Baseline |

### Why Linear SVM Outperformed Other Models:
1. **High-Dimensional Sparsity:** TF-IDF feature matrices are high-dimensional ($10,000$ dimensions) and sparse. Linear SVMs find the optimal maximum-margin hyperplane separating classes with minimal risk of overfitting.
2. **Robustness to Collinear N-grams:** Unlike Naive Bayes (which presumes feature conditional independence), SVM naturally weights interacting unigram-bigram patterns without independence distortions.
3. **Calibrated Decision Boundary:** By calibrating with Platt scaling (`CalibratedClassifierCV`), we retain SVM's geometric margin advantages while gaining reliable class probabilities.

---

## 11. Confusion Matrix & Error Tradeoffs

### Test Set Breakdown (Champion Model: Linear SVM)
Total Test Samples: **3,057**

```text
                      PREDICTED
                 Fake (0)      Real (1)
ACTUAL Fake (0)    1,439          23          (Total Fake: 1,462)
ACTUAL Real (1)      15        1,580          (Total Real: 1,595)
```

- **True Negatives (TN = 1,439):** Fabricated articles correctly flagged as Fake.
- **True Positives (TP = 1,580):** Authentic journalism correctly verified as Real.
- **False Positives (FP = 23):** Fake articles incorrectly tagged as Real *(Type I error)*.
- **False Negatives (FN = 15):** Real journalism erroneously flagged as Fake *(Type II error)*.

### Societal & Operational Impact of Errors:
- **Impact of False Positives (FP):** Allows malicious falsehoods (e.g., fraudulent investment schemes or harmful medical claims) to circulate with algorithmic approval.
- **Impact of False Negatives (FN):** Falsely discredits established, legitimate journalism, eroding user trust in factual reporting and exposing platforms to censorship criticism.
- **Recommended Industry Triage:** Systems should automate decisions only above high confidence thresholds (e.g., $>85\%$) and route ambiguous scores ($40\% - 60\%$) to professional human fact-checkers.

---

## 12. Limitations & Ethical Considerations

1. **Stylistic Pattern Matching $\neq$ Fact Verification:**
   The model learns writing styles, sensationalist vocabulary, and syntactic patterns. It **cannot check factual truth** against real-world evidence or databases. A completely fabricated lie written in sober, academic prose may deceive the model.
2. **Dataset & Temporal Bias:**
   Training corpora reflect specific time periods and political topics. Newly emerging topics, novel slang, or unseen event contexts may degrade model performance.
3. **Publisher Shortcut Artifacts:**
   Datasets often contain source signatures (e.g., `(Reuters)`). While our preprocessing strips common datelines, real-world deployment requires continuous data scrubbing to prevent models from learning publisher names rather than linguistic signals.
4. **No Proof of Truth:**
   Predictions should always be presented as an advisory likelihood, never as authoritative censorship criteria.

---

## 13. Future Enhancements & Portfolio Roadmap

- [ ] **Transformer Ensembles:** Fine-tune pre-trained BERT, RoBERTa, or DeBERTa architectures for context-aware sentence embeddings.
- [ ] **Fact-Checking API Integration:** Link predictions to Google Fact Check Tools API or ClaimReview schemas for automated ground-truth claim verification.
- [ ] **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) or LIME in the Streamlit UI to highlight the exact keywords that tipped the prediction.
- [ ] **Multilingual Support:** Implement XLM-RoBERTa for fake news detection across Hindi, Spanish, French, and regional languages.
- [ ] **Source Credibility & Graph Analysis:** Combine article text with publisher domain reputation scores and social network propagation graphs.
- [ ] **Browser Extension:** Build a lightweight Chrome Extension to scan news articles directly in web browsers.

---

## Author

**Kota Adbuth Kumar**
- GitHub: [@kotaAdbuthkumar](https://github.com/kotaAdbuthkumar)
- Repository: [Fake-News-Detection](https://github.com/kotaAdbuthkumar/Fake-News-Detection)

---

## License

This project is licensed under the MIT License by **Kota Adbuth Kumar** - see the [LICENSE](LICENSE) file for details.
