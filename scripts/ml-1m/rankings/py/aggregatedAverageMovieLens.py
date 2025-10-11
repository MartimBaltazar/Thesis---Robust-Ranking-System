import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# === Custom display format ===
def custom_format(x):
    if isinstance(x, (int, np.integer)):
        return str(x)  # integers as-is
    try:
        if abs(x) >= 1e4:
            return f"{x:.4e}"  # scientific notation
        else:
            return f"{x:.6f}"  # normal float
    except Exception:
        return str(x)

pd.set_option('display.float_format', custom_format)

# === Paths ===
spam_dir = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/spam_versions"
bribery_dir = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/bribery_attack_sets"
original_path = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/normalized_ratings.dat"

ratios = [10, 30, 50, 70]
target_items = [1749, 2261, 2858]


# === Load original dataset ===
df = pd.read_csv(original_path, sep="::", engine="python", 
                 names=["UserID", "MovieID", "Rating", "Timestamp", "NormalizedRating"])

# === Original aggregated average rankings ===
original_avg_rankings = df.groupby("MovieID")["NormalizedRating"].mean().sort_values(ascending=False)
original_avg_dict = original_avg_rankings.to_dict()

def compute_kendall_tau(dict1, dict2):
    common_keys = set(dict1.keys()).intersection(dict2.keys())
    v1 = [dict1[k] for k in common_keys]
    v2 = [dict2[k] for k in common_keys]
    tau, _ = kendalltau(v1, v2)
    return tau

# === Robustness (Spam Resistance) ===
print("=== Aggregated Average Robustness (Spam) ===")
for percent in ratios:
    start_time = time.time()

    file_name = f"ratings_with_{percent}percent_spam.csv"
    file_path = os.path.join(spam_dir, file_name)

    df_attack = pd.read_csv(file_path)
    df_attack.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp', 'NormalizedRating']

    attack_avg_rankings = df_attack.groupby("MovieID")["NormalizedRating"].mean().sort_values(ascending=False)
    attack_avg_dict = attack_avg_rankings.to_dict()

    tau_value = compute_kendall_tau(original_avg_dict, attack_avg_dict)

    elapsed_time = time.time() - start_time
    print(f"[{percent}% Spam] Kendall’s τ: {tau_value:.4f} — Time: {elapsed_time:.2f} seconds")


# === Plot Original Distribution ===
ratings = list(original_avg_dict.values())
bins = np.arange(0, 1.1, 0.1)
hist, bin_edges = np.histogram(ratings, bins=bins)

plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=0.1, align='edge', edgecolor='black', color='skyblue')
for i in range(len(hist)):
    plt.text(bin_edges[i] + 0.05, hist[i] + 0.5, str(hist[i]), ha='center', fontsize=12)
plt.xlabel('Aggregated Average Rating', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.title('Distribution of Aggregated Average Movie Ratings (Original)', fontsize=14)
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# === Bribery Resistance ===
def compute_wealth_avg(df, item_id, rankings):
    n_ratings = df[df["MovieID"] == item_id].shape[0]
    avg_rating = rankings.get(item_id, 0)
    print(f"Item {item_id}: n_ratings={n_ratings}, avg_rating={avg_rating}")
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

# Compute original wealth
original_wealth_avg = {item: compute_wealth_avg(df, item, original_avg_dict) for item in target_items}
print("\n📊 Original Wealth (Aggregated Average):")
print(original_wealth_avg)

results_avg = []
for item_id in target_items:
    item_folder = os.path.join(bribery_dir, f"item_{item_id}")
    for fname in os.listdir(item_folder):
        if fname.endswith(".csv"):
            scenario_name = fname.replace(".csv", "")
            full_path_csv = os.path.join(item_folder, fname)
            altered_df = pd.read_csv(full_path_csv)

            txt_path = os.path.join(item_folder, scenario_name + "_changes.txt")
            strategy_cost = 0.0
            scale_by_user = "new" in scenario_name.lower()
            if os.path.exists(txt_path):
                strategy_cost = parse_strategy_cost(txt_path, scale_by_user=scale_by_user)

            attacked_rankings_avg = altered_df.groupby("MovieID")["NormalizedRating"].mean().to_dict()
            attacked_wealth = compute_wealth_avg(altered_df, item_id, attacked_rankings_avg)

            delta = attacked_wealth - original_wealth_avg[item_id] - strategy_cost

            results_avg.append({
                "ItemID": item_id,
                "Scenario": scenario_name,
                "OriginalWealth": original_wealth_avg[item_id],
                "AttackedWealth": attacked_wealth,
                "StrategyCost": strategy_cost,
                "Delta": delta
            })

results_df_avg = pd.DataFrame(results_avg)
print("\n📋 Aggregated Average Bribery Resistance Results:")
print(results_df_avg.to_string(index=False))

results_df_avg["AbsDelta"] = results_df_avg["Delta"].abs()
sum_abs_deltas_avg = results_df_avg.groupby("ItemID")["AbsDelta"].sum().reset_index()
sum_abs_deltas_avg.rename(columns={"AbsDelta": "TotalAbsDelta"}, inplace=True)

print("\n📊 Total Sum of |Delta| per Item (Aggregated Average):")
print(sum_abs_deltas_avg.to_string(index=False))
