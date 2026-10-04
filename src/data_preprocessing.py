"""
Data Preprocessing Module for Fake News Detection System.

This module handles:
1. Loading and combining datasets (e.g., Fake.csv and True.csv).
2. Schema standardization and label mapping (Fake = 0, Real = 1).
3. Data hygiene (handling missing values, empty strings, duplicates).
4. Text preprocessing (lowercase, URL/HTML stripping, punctuation removal,
   negation-aware stopword filtering, and lemmatization).
5. Dataset export for model training and EDA.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure essential NLTK resources are available
def _init_nltk():
    """Download required NLTK corpora if not already present."""
    for resource in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource in ['stopwords', 'wordnet'] else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

_init_nltk()

# Initialize stopwords while preserving crucial negation tokens
# Crucial for sentiment and truth assertion (e.g., "did not commit" vs "did commit")
_NEGATION_WORDS = {'not', 'no', 'never', 'nor', 'neither', 'barely', 'scarcely', 'hardly'}
_BASE_STOPWORDS = set(stopwords.words('english'))
ACTIVE_STOPWORDS = _BASE_STOPWORDS - _NEGATION_WORDS

# Publisher datelines often present in real news (e.g., ISOT Reuters dataset)
# Stripping these ensures the model learns linguistic style rather than publisher shortcuts
SOURCE_PATTERNS = [
    r'^[a-z\s]+reuters\s*[-–—]?\s*',
    r'^\(reuters\)\s*[-–—]?\s*',
    r'^[a-z]+,\s*[a-z]+\s*\(reuters\)\s*[-–—]?\s*'
]

_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str, preserve_negations: bool = True, remove_source_tags: bool = True) -> str:
    """
    Cleans and standardizes raw text for NLP models.

    Steps:
    1. Check for valid string input.
    2. Convert to lowercase.
    3. Remove URLs and hyperlinks.
    4. Remove HTML tags and bracketed citations.
    5. Optionally strip common news agency datelines (e.g., 'Reuters').
    6. Remove non-alphabetic characters and punctuation.
    7. Remove extra whitespace.
    8. Tokenize and remove stopwords (preserving negations if requested).
    9. Lemmatize tokens to base form.

    Parameters:
    ----------
    text : str
        The raw article text or headline.
    preserve_negations : bool, default=True
        Whether to keep words like 'not', 'no', 'never' in the text.
    remove_source_tags : bool, default=True
        Whether to strip leading publisher tags like 'reuters - '.

    Returns:
    -------
    str
        The cleaned, normalized text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Remove URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', ' ', cleaned)

    # 3. Remove HTML tags
    cleaned = re.sub(r'<.*?>', ' ', cleaned)

    # 4. Remove brackets and contents like [12:00 PM] or [Video]
    cleaned = re.sub(r'\[.*?\]', ' ', cleaned)

    # 5. Optionally remove publisher dateline tags
    if remove_source_tags:
        for pattern in SOURCE_PATTERNS:
            cleaned = re.sub(pattern, ' ', cleaned)

    # 6. Remove punctuation and non-alphabetic characters
    cleaned = re.sub(r'[^a-z\s]', ' ', cleaned)

    # 7. Tokenize by whitespace
    tokens = cleaned.split()

    # 8. Select stopword filter
    stop_set = ACTIVE_STOPWORDS if preserve_negations else _BASE_STOPWORDS

    # 9. Filter and lemmatize tokens (length > 2 removes residual single letters)
    processed_tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in stop_set and len(token) > 2
    ]

    return ' '.join(processed_tokens)


def load_and_combine_data(
    fake_path: str = 'data/Fake.csv',
    true_path: str = 'data/True.csv',
    text_col: str = 'text',
    title_col: str = 'title',
    combine_title_and_text: bool = True,
    sample_size: int = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads separate Fake and True CSV files, assigns standard binary labels,
    combines them into a single DataFrame, and shuffles the records.

    Label encoding:
    - Fake News = 0
    - Real News = 1

    Parameters:
    ----------
    fake_path : str
        Path to Fake.csv.
    true_path : str
        Path to True.csv.
    text_col : str
        Name of the column containing the article body.
    title_col : str
        Name of the column containing the article headline.
    combine_title_and_text : bool
        If True, concatenates title and text into a single 'text' feature.
    sample_size : int, optional
        If set, samples up to sample_size balanced rows for rapid training.
    random_state : int
        Seed for reproducibility.

    Returns:
    -------
    pd.DataFrame
        Combined DataFrame with 'text' and 'label' columns.
    """
    if not os.path.exists(fake_path):
        raise FileNotFoundError(f"Fake news dataset not found at '{fake_path}'. Please check the path.")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"True news dataset not found at '{true_path}'. Please check the path.")

    print(f"Loading Fake news from: {fake_path}")
    fake_df = pd.read_csv(fake_path)
    print(f"Loading Real news from: {true_path}")
    real_df = pd.read_csv(true_path)

    # Assign binary labels: Fake = 0, Real = 1
    fake_df['label'] = 0
    real_df['label'] = 1

    # Combine title and text if present
    for df in [fake_df, real_df]:
        if combine_title_and_text and title_col in df.columns and text_col in df.columns:
            df['text'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
        elif text_col in df.columns:
            df['text'] = df[text_col].fillna('')
        elif title_col in df.columns:
            df['text'] = df[title_col].fillna('')
        else:
            raise KeyError(f"Neither '{text_col}' nor '{title_col}' column found in dataset.")

    # Keep only relevant columns
    fake_subset = fake_df[['text', 'label']].copy()
    real_subset = real_df[['text', 'label']].copy()

    # Optional balanced subsampling
    if sample_size and sample_size < (len(fake_subset) + len(real_subset)):
        half_sample = sample_size // 2
        fake_subset = fake_subset.sample(n=min(half_sample, len(fake_subset)), random_state=random_state)
        real_subset = real_subset.sample(n=min(half_sample, len(real_subset)), random_state=random_state)
        print(f"Sampled {len(fake_subset)} fake and {len(real_subset)} real articles for balanced training.")

    # Combine and shuffle
    combined_df = pd.concat([fake_subset, real_subset], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return combined_df


def load_custom_dataset(
    filepath: str,
    text_col: str = 'text',
    label_col: str = 'label',
    fake_label_value = 0,
    real_label_value = 1
) -> pd.DataFrame:
    """
    Loads any custom CSV with configurable column names and maps labels to 0 and 1.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'.")

    df = pd.read_csv(filepath)
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found in dataset. Columns: {df.columns.tolist()}")
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataset. Columns: {df.columns.tolist()}")

    # Normalize labels to 0 and 1
    mapping = {fake_label_value: 0, real_label_value: 1}
    standardized = pd.DataFrame()
    standardized['text'] = df[text_col].astype(str)
    standardized['label'] = df[label_col].map(mapping).fillna(df[label_col]).astype(int)

    return standardized


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Performs data cleaning:
    - Drops null values
    - Strips whitespace
    - Drops empty text strings
    - Optionally removes duplicate records
    """
    initial_count = len(df)
    
    # Drop NaNs
    df = df.dropna(subset=['text', 'label']).copy()

    # Strip and filter empty strings
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 10].copy()

    # Drop duplicates
    if drop_duplicates:
        df = df.drop_duplicates(subset=['text']).copy()

    dropped = initial_count - len(df)
    print(f"Cleaned dataset: {len(df)} records remaining ({dropped} invalid/duplicate records removed).")
    return df.reset_index(drop=True)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generates summary statistics of the dataset.
    """
    fake_count = (df['label'] == 0).sum()
    real_count = (df['label'] == 1).sum()
    
    summary = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_records': int(df.duplicated(subset=['text']).sum()),
        'class_distribution': {
            'Fake (0)': int(fake_count),
            'Real (1)': int(real_count)
        },
        'fake_percentage': round(fake_count / len(df) * 100, 2) if len(df) else 0,
        'real_percentage': round(real_count / len(df) * 100, 2) if len(df) else 0
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Preprocess Fake News Dataset")
    parser.add_argument('--fake_path', type=str, default='data/Fake.csv', help='Path to Fake.csv')
    parser.add_argument('--true_path', type=str, default='data/True.csv', help='Path to True.csv')
    parser.add_argument('--output_path', type=str, default='data/processed_data.csv', help='Output path')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of rows to sample (optional)')
    parser.add_argument('--no_preprocess_text', action='store_true', help='Skip regex text cleaning')
    args = parser.parse_args()

    # Adjust paths relative to project root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    fake_p = os.path.join(base_dir, args.fake_path) if not os.path.isabs(args.fake_path) else args.fake_path
    true_p = os.path.join(base_dir, args.true_path) if not os.path.isabs(args.true_path) else args.true_path
    out_p = os.path.join(base_dir, args.output_path) if not os.path.isabs(args.output_path) else args.output_path

    print("=== Step 1: Loading Dataset ===")
    df = load_and_combine_data(fake_p, true_p, sample_size=args.sample_size)
    summary = get_dataset_summary(df)
    print(f"Total records loaded: {summary['total_rows']}")
    print(f"Class distribution: Fake={summary['class_distribution']['Fake (0)']}, Real={summary['class_distribution']['Real (1)']}")

    print("\n=== Step 2: Data Cleaning ===")
    df = clean_dataset(df)

    if not args.no_preprocess_text:
        print("\n=== Applying NLP Text Preprocessing ===")
        print("Transforming raw text (lowercase, URL/HTML removal, punctuation, stopwords, lemmatization)...")
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        # Drop rows where cleaning produced empty string
        df = df[df['cleaned_text'].str.strip().str.len() > 5].copy()

    # Save processed dataset
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"\n[Success] Processed dataset saved to: {out_p}")
    print(f"Final shape: {df.shape}")


if __name__ == '__main__':
    main()
