"""
EDA and Visualization Generator for Fake News Detection.
Generates publication-quality charts into the reports/ folder:
1. Class Distribution (Fake vs Real)
2. Article Length & Word Count Distributions
3. Top Most Frequent Words in Fake vs Real Articles
4. WordClouds for Fake vs Real News
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

# Set styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
reports_dir = os.path.join(base_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)

data_path = os.path.join(base_dir, 'data', 'processed_data.csv')
if not os.path.exists(data_path):
    print("processed_data.csv not found, loading raw files...")
    from src.data_preprocessing import load_and_combine_data
    df = load_and_combine_data(os.path.join(base_dir, 'data', 'Fake.csv'),
                               os.path.join(base_dir, 'data', 'True.csv'),
                               sample_size=10000)
    df['cleaned_text'] = df['text']
else:
    df = pd.read_csv(data_path)

# Compute basic text stats
df['char_length'] = df['text'].astype(str).str.len()
df['word_count'] = df['text'].astype(str).str.split().str.len()

# 1. Class Distribution Plot
plt.figure(figsize=(7, 5))
palette = ['#EF4444', '#10B981'] # Red for Fake, Green for Real
ax = sns.countplot(x='label', data=df, hue='label', palette=palette, legend=False)
plt.title("Fake vs Real News Class Distribution", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("News Category", fontsize=12)
plt.ylabel("Number of Articles", fontsize=12)
plt.xticks([0, 1], ['Fake News (0)', 'Real News (1)'], fontsize=11)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')
plt.tight_layout()
dist_path = os.path.join(reports_dir, 'class_distribution.png')
plt.savefig(dist_path, dpi=300)
plt.close()
print(f"Saved: {dist_path}")

# 2. Word Count Distribution Plot
plt.figure(figsize=(10, 5))
# Clip extreme outliers for clean plotting
clipped_words = df[df['word_count'] < 1500]
sns.histplot(data=clipped_words, x='word_count', hue='label', kde=True,
             palette=palette, bins=40, alpha=0.6, element='step')
plt.title("Distribution of Article Word Counts (Fake vs Real)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Word Count per Article", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(['Real News (1)', 'Fake News (0)'], loc='upper right')
plt.tight_layout()
wc_dist_path = os.path.join(reports_dir, 'word_count_distribution.png')
plt.savefig(wc_dist_path, dpi=300)
plt.close()
print(f"Saved: {wc_dist_path}")

# 3. Top Words Comparison
fake_words = " ".join(df[df['label'] == 0]['cleaned_text'].dropna().astype(str)).split()
real_words = " ".join(df[df['label'] == 1]['cleaned_text'].dropna().astype(str)).split()

fake_common = Counter(fake_words).most_common(12)
real_common = Counter(real_words).most_common(12)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df_fake_words = pd.DataFrame(fake_common, columns=['word', 'count'])
sns.barplot(data=df_fake_words, x='count', y='word', ax=axes[0], color='#EF4444')
axes[0].set_title("Top 12 Frequent Words in Fake News", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Term Frequency", fontsize=11)

df_real_words = pd.DataFrame(real_common, columns=['word', 'count'])
sns.barplot(data=df_real_words, x='count', y='word', ax=axes[1], color='#10B981')
axes[1].set_title("Top 12 Frequent Words in Real News", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Term Frequency", fontsize=11)

plt.tight_layout()
top_words_path = os.path.join(reports_dir, 'top_words_comparison.png')
plt.savefig(top_words_path, dpi=300)
plt.close()
print(f"Saved: {top_words_path}")

# 4. Word Clouds
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

wc_fake = WordCloud(width=800, height=450, background_color='white',
                    colormap='Reds', max_words=100, contour_color='#B91C1C').generate(" ".join(fake_words[:50000]))
axes[0].imshow(wc_fake, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title("WordCloud: Fake News Vocabulary", fontsize=16, fontweight='bold', pad=12)

wc_real = WordCloud(width=800, height=450, background_color='white',
                    colormap='Greens', max_words=100, contour_color='#047857').generate(" ".join(real_words[:50000]))
axes[1].imshow(wc_real, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title("WordCloud: Real News Vocabulary", fontsize=16, fontweight='bold', pad=12)

plt.tight_layout()
wc_path = os.path.join(reports_dir, 'wordcloud_fake_vs_real.png')
plt.savefig(wc_path, dpi=300)
plt.close()
print(f"Saved: {wc_path}")
print("All EDA visualizations successfully created in reports/ directory.")
