"""
Model Evaluation and Error Analysis Module for Fake News Detection System.

This module:
1. Loads serialized training metrics and models.
2. Formats and prints classification reports.
3. Generates high-resolution confusion matrix heatmaps (Actual vs. Predicted).
4. Plots multi-metric comparative bar charts (Accuracy, Precision, Recall, F1).
5. Explains TP, TN, FP, FN, and analyzes the societal and systemic impacts of error types.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(cm: list, model_name: str, save_path: str):
    """
    Renders and saves a confusion matrix heatmap with clear labels.
    cm format:
    [[TN, FP],
     [FN, TP]]
    Where:
    - Row 0: Actual Fake (0) -> TN (Pred Fake), FP (Pred Real)
    - Row 1: Actual Real (1) -> FN (Pred Fake), TP (Pred Real)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Custom annotations with counts and percentages
    total = np.sum(cm_arr)
    labels = np.array([
        [f"True Negative\n(Fake correctly flagged)\n{cm_arr[0,0]:,}\n({cm_arr[0,0]/total:.1%})",
         f"False Positive\n(Fake misclassified as Real)\n{cm_arr[0,1]:,}\n({cm_arr[0,1]/total:.1%})"],
        [f"False Negative\n(Real misclassified as Fake)\n{cm_arr[1,0]:,}\n({cm_arr[1,0]/total:.1%})",
         f"True Positive\n(Real correctly verified)\n{cm_arr[1,1]:,}\n({cm_arr[1,1]/total:.1%})"]
    ])

    sns.heatmap(
        cm_arr,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        ax=ax,
        xticklabels=['Predicted FAKE (0)', 'Predicted REAL (1)'],
        yticklabels=['Actual FAKE (0)', 'Actual REAL (1)'],
        linewidths=1.5,
        linecolor='white'
    )

    plt.title(f"Confusion Matrix: {model_name}", fontsize=14, pad=15, fontweight='bold')
    plt.ylabel("Actual Ground Truth", fontsize=12)
    plt.xlabel("Model Prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization Saved] Confusion matrix saved to: {save_path}")


def plot_model_comparison(models_data: dict, save_path: str):
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    df_plot = pd.DataFrame(models_data).T[metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.plot(kind='bar', ax=ax, colormap='viridis', width=0.75, edgecolor='black', alpha=0.9)

    ax.set_ylim(0.70, 1.02)
    ax.set_title("Machine Learning Models Performance Comparison", fontsize=15, pad=15, fontweight='bold')
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=12)
    ax.set_xlabel("Machine Learning Classifier", fontsize=12)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    plt.xticks(rotation=0, fontsize=11, fontweight='semibold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Annotate value on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization Saved] Model comparison chart saved to: {save_path}")


def print_theoretical_error_analysis(cm: list, champion_name: str):
    """
    Provides an in-depth data science explanation of the Confusion Matrix and error tradeoffs.
    """
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    print("\n" + "=" * 70)
    print(f"DEEP DIVE: CONFUSION MATRIX & ERROR TRADEOFF ANALYSIS ({champion_name})")
    print("=" * 70)
    print(f"""
                      PREDICTED
                 Fake (0)      Real (1)
ACTUAL Fake (0)    {tn:<10}    {fp:<10}  (Total Fake: {tn+fp})
ACTUAL Real (1)    {fn:<10}    {tp:<10}  (Total Real: {fn+tp})

1. Breakdown of Matrix Components:
   ---------------------------------
   • True Negative (TN = {tn:,}):
     Fabricated / Fake articles correctly identified as FAKE.
     Protects users from consuming misinformation.

   • True Positive (TP = {tp:,}):
     Legitimate / Fact-based articles correctly identified as REAL.
     Preserves reader access to authentic journalism.

   • False Positive (FP = {fp:,}) [Type I Error]:
     Fake news erroneously classified as REAL.
     The system failed to detect deception, allowing falsehoods to circulate with perceived credibility.

   • False Negative (FN = {fn:,}) [Type II Error]:
     Authentic real news erroneously tagged as FAKE.
     Unfairly censors or discredits legitimate news publishers and causes public distrust in reliable facts.

2. Which Error Type is More Critical in Fake News Detection?
   ----------------------------------------------------------
   In a production misinformation prevention system:
   
   - High False Positives (FP) lead to Viral Misinformation:
     If a malicious headline (e.g. medical misinformation or financial scam) slips through as 'Real',
     public health, safety, or democratic integrity could be compromised.

   - High False Negatives (FN) lead to Censorship & Loss of Institutional Trust:
     If credible news agencies (e.g. AP, Reuters, BBC) are frequently flagged as 'Fake News',
     the platform faces intense backlash, censorship accusations, and loss of user faith.

   - Optimal Engineering Strategy:
     Rather than a hard binary cutoff, production systems use a threshold or triage zone:
     • High Confidence (>85%): Automated decision.
     • Ambiguous Confidence (40% - 60%): Route to human fact-checkers and display advisory tags.
""")


def evaluate_existing_models(
    metrics_path: str = 'models/model_metrics.json',
    reports_dir: str = 'reports'
):
    """
    Reads the stored metrics JSON, prints formatted tables, and generates visualization figures.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    full_metrics_path = os.path.join(base_dir, metrics_path)
    reports_dir_path = os.path.join(base_dir, reports_dir)

    if not os.path.exists(full_metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found at '{full_metrics_path}'. Please run 'train_model.py' first."
        )

    with open(full_metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    champion = data['champion_model']
    models_dict = data['models']

    print("=" * 70)
    print("EVALUATION REPORT: FAKE NEWS DETECTION ML PIPELINE")
    print("=" * 70)
    print(f"Total Dataset Samples: {data.get('sample_size', 'N/A'):,}")
    print(f"Training Set Samples:  {data.get('train_size', 'N/A'):,}")
    print(f"Test Set Samples:      {data.get('test_size', 'N/A'):,}")
    print(f"TF-IDF Vocabulary:     {data.get('vocabulary_size', 'N/A'):,} features")
    print(f"Champion Model:        {champion}")

    print("\n" + "-" * 70)
    print("BENCHMARK COMPARISON TABLE")
    print("-" * 70)
    summary_df = pd.DataFrame(models_dict).T[['Accuracy', 'Precision', 'Recall', 'F1 Score']]
    print(summary_df.to_string())

    # Generate visual charts
    cm_path = os.path.join(reports_dir_path, 'confusion_matrix.png')
    comp_path = os.path.join(reports_dir_path, 'model_comparison.png')

    champion_cm = models_dict[champion]['Confusion Matrix']
    plot_confusion_matrix(champion_cm, champion, cm_path)
    plot_model_comparison(models_dict, comp_path)

    # Print error analysis
    print_theoretical_error_analysis(champion_cm, champion)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models and Generate Reports")
    parser.add_argument('--metrics_path', type=str, default='models/model_metrics.json')
    parser.add_argument('--reports_dir', type=str, default='reports')
    args = parser.parse_args()

    evaluate_existing_models(metrics_path=args.metrics_path, reports_dir=args.reports_dir)


if __name__ == '__main__':
    main()
