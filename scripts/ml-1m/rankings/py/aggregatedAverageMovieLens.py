import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Suppress scientific notation
pd.set_option('display.float_format', lambda x: '%.6f' % x)


# === Paths ===
base_path = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m"
spam_dir = os.path.join(base_path, "spam_versions")
original_file = os.path.join(base_path, "normalized_ratings.dat")
bribery_path = os.path.join(base_path, "bribery_attack_sets")

# === Load original dataset ===
df_original = pd.read_csv(original_file, sep="::", engine="python",
                          names=["UserID", "MovieID", "Rating", "Timestamp", "NormalizedRating"])
original_rankings = df_original.groupby("MovieID")["NormalizedRating"].mean().sort_values(ascending=False)
original_avg_dict = original_rankings.to_dict()

# === ROBUSTNESS ===
ratios = [10, 30, 50, 70]
kendall_results = []

for percent in ratios:
    spam_file = f"ratings_with_{percent}percent_spam.csv"
    spam_path = os.path.join(spam_dir, spam_file)

    df_spam = pd.read_csv(spam_path)
    df_spam.columns = ["UserID", "MovieID", "Rating", "Timestamp", "NormalizedRating"]

    spam_rankings = df_spam.groupby("MovieID")["NormalizedRating"].mean().sort_values(ascending=False)

    common_items = original_rankings.index.intersection(spam_rankings.index)
    tau, _ = kendalltau(original_rankings[common_items], spam_rankings[common_items])
    kendall_results.append((percent, tau))
    print(f"[{percent}% Spam] Kendall’s τ: {tau:.4f}")

# === BRIBERY RESISTANCE ===
def compute_wealth(df, item_id, rankings):
    n_ratings = df[df["MovieID"] == item_id].shape[0]
    avg_rating = rankings.get(item_id, 0)
    return n_ratings * avg_rating

def parse_strategy_cost(txt_path, scale_by_user=False):
    total_cost = 0.0
    with open(txt_path, "r") as f:
        for line in f:
            if "UserID" in line and "Old" in line and "New" in line:
                try:
                    parts = line.strip().split(", ")
                    old_val = parts[1].split(": ")[1]
                    new_val = parts[2].split(": ")[1]
                    old = float(old_val) if old_val != "None" else 0.0
                    new = float(new_val) if new_val != "None" else 0.0
                    total_cost += abs(new - old)
                except Exception as e:
                    print(f"⚠️ Error parsing line: {line.strip()} → {e}")
    if scale_by_user:
        total_cost *= 1
    return total_cost

target_items = [1749, 2261, 2858]
original_wealth = {item: compute_wealth(df_original, item, original_avg_dict) for item in target_items}
print("\n📊 Original Wealth (Aggregated Average):")
print(original_wealth)

bribery_results = []

for item_id in target_items:
    item_folder = os.path.join(bribery_path, f"item_{item_id}")
    for fname in os.listdir(item_folder):
        if fname.endswith(".csv"):
            scenario_name = fname.replace(".csv", "")
            csv_path = os.path.join(item_folder, fname)
            df_attack = pd.read_csv(csv_path)

            attacked_rankings = df_attack.groupby("MovieID")["NormalizedRating"].mean().to_dict()
            attacked_wealth = compute_wealth(df_attack, item_id, attacked_rankings)

            txt_path = os.path.join(item_folder, scenario_name + "_changes.txt")
            strategy_cost = 0.0
            if os.path.exists(txt_path):
                strategy_cost = parse_strategy_cost(txt_path, scale_by_user="new" in scenario_name.lower())

            delta = attacked_wealth - original_wealth[item_id] - strategy_cost

            bribery_results.append({
                "ItemID": item_id,
                "Scenario": scenario_name,
                "OriginalWealth": original_wealth[item_id],
                "AttackedWealth": attacked_wealth,
                "StrategyCost": strategy_cost,
                "Delta": delta
            })

bribery_df = pd.DataFrame(bribery_results)
bribery_df["AbsDelta"] = bribery_df["Delta"].abs()

sum_abs_deltas = bribery_df.groupby("ItemID")["AbsDelta"].sum().reset_index()
sum_abs_deltas.rename(columns={"AbsDelta": "TotalAbsDelta"}, inplace=True)

# # === PLOTS ===

# ## Plot 1: Distribution of Original Ratings
# ratings = list(original_avg_dict.values())
# bins = np.arange(0.1, 1.1, 0.1)
# hist, bin_edges = np.histogram(ratings, bins=bins)

# plt.figure(figsize=(10, 6))
# plt.bar(bin_edges[:-1], hist, width=0.08, align='edge', edgecolor='black')
# for i in range(len(hist)):
#     plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', fontsize=12)
# plt.xlabel('Aggregated Average Rating', fontsize=12)
# plt.ylabel('Number of Movies', fontsize=12)
# plt.title('Distribution of Aggregated Average Movie Ratings (Original)', fontsize=14)
# plt.xticks(bins)
# plt.tight_layout()
# plt.show()

# ## Plot 2: Kendall’s Tau vs. Spam Percentage
# spam_levels, taus = zip(*kendall_results)
# plt.figure(figsize=(8, 5))
# plt.plot(spam_levels, taus, marker='o', linestyle='-', color='steelblue')
# plt.xlabel('% Spam Injected', fontsize=12)
# plt.ylabel("Kendall's τ", fontsize=12)
# plt.title("Robustness of Aggregated Average Ranking to Spam", fontsize=14)
# plt.grid(True)
# plt.ylim(0, 1)
# plt.xticks(spam_levels)
# plt.tight_layout()
# plt.show()

# ## Plot 3: Bribery Resistance — Total |Delta| per Item
# plt.figure(figsize=(8, 5))
# plt.bar(sum_abs_deltas["ItemID"].astype(str), sum_abs_deltas["TotalAbsDelta"],
#         color='indianred', edgecolor='black')
# for idx, row in sum_abs_deltas.iterrows():
#     plt.text(row["ItemID"], row["TotalAbsDelta"] + 0.5,
#              f'{row["TotalAbsDelta"]:.2f}', ha='center', fontsize=12)
# plt.title("Bribery Resistance: Total |Delta| per Item", fontsize=14)
# plt.xlabel("Item ID", fontsize=12)
# plt.ylabel("Total |Delta| (Wealth Change - Cost)", fontsize=12)
# plt.tight_layout()
# plt.show()

# === Optional: Print detailed bribery results ===
print("\n📋 All Bribery Attack Scenarios:")
print(bribery_df.to_string(index=False))
