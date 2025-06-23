import os
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

# === Paths ===
base_dir = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing"
spam_dir = os.path.join(base_dir, "spam_versions")
original_file = os.path.join(base_dir, "Shortened_Ratings.csv")

# === Load original dataset ===
original_df = pd.read_csv(original_file)
original_df.columns = ['user_id', 'item_id', 'normalized_rating']
original_df['normalized_rating'] = original_df['normalized_rating'] / 10.0

# Compute original aggregated average rankings
original_rankings = original_df.groupby("item_id")["normalized_rating"].mean().sort_values(ascending=False)

# === Evaluate robustness under spam ===
ratios = [10, 30, 50, 70]
kendall_results = []

for percent in ratios:
    spam_file = f"ratings_with_{percent}percent_spam.csv"
    spam_path = os.path.join(spam_dir, spam_file)

    df_spam = pd.read_csv(spam_path)
    df_spam.columns = ['user_id', 'item_id', 'normalized_rating']
    df_spam['normalized_rating'] = df_spam['normalized_rating'] / 10.0

    spam_rankings = df_spam.groupby("item_id")["normalized_rating"].mean().sort_values(ascending=False)

    # Compare using Kendall’s Tau
    common_items = original_rankings.index.intersection(spam_rankings.index)
    tau, _ = kendalltau(original_rankings[common_items], spam_rankings[common_items])
    kendall_results.append((percent, tau))
    print(f"[{percent}% Spam] Kendall’s τ: {tau:.4f}")

# === Plot: Kendall’s Tau vs. Spam Percentage ===
spam_levels, taus = zip(*kendall_results)
plt.figure(figsize=(8, 5))
plt.plot(spam_levels, taus, marker='o', linestyle='-', color='seagreen')
plt.xlabel('% Spam Injected', fontsize=12)
plt.ylabel("Kendall's τ", fontsize=12)
plt.title("Robustness of Aggregated Average (BookCrossing)", fontsize=14)
plt.grid(True)
plt.ylim(0, 1)
plt.xticks(spam_levels)
plt.tight_layout()
plt.show()
