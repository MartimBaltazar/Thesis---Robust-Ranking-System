import os
import pandas as pd
import numpy as np
import time


# === Aggregated Average Ranking Algorithm ===
def aggregated_ranking_algorithm(df):
    return df.groupby("item_id")["normalized_rating"].mean().to_dict()

# === Robustness Evaluation (Spam Injection) ===
spam_dir = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/spam_versions"
ratios = [10, 30, 50, 70]

# Load original dataset
file_path = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Shortened_Ratings.csv"
original_df = pd.read_csv(file_path) 
print("LOADEDDDD")
original_df.columns = ['user_id', 'item_id', 'normalized_rating']
# Select the rows for the specific item
item_df = original_df[original_df["item_id"] == "0971880107"]

# Maximum normalized rating
max_rating = item_df["normalized_rating"].max()

# Average normalized rating
avg_rating = item_df["normalized_rating"].mean()

print(f"Item 0971880107: max rating = {max_rating}, average rating = {avg_rating:.3f}")
original_df['normalized_rating'] = original_df['normalized_rating'] / 10

# Compute original aggregated rankings
rankings = aggregated_ranking_algorithm(original_df)

# print("=== Robustness Analysis (Aggregated Average) ===")
# for percent in ratios:
#     start_time = time.time()

#     file_name = f"ratings_with_{percent}percent_spam.csv"
#     file_path = os.path.join(spam_dir, file_name)

#     df_attack = pd.read_csv(file_path)
#     df_attack.columns = ['user_id', 'item_id', 'normalized_rating']
#     df_attack['normalized_rating'] = df_attack['normalized_rating'] / 10

#     rankingsSpam = aggregated_ranking_algorithm(df_attack)

#     # Compute Kendall’s τ
#     tau_value = compute_kendall_tau(rankings, rankingsSpam)

#     elapsed = time.time() - start_time
#     print(f"[{percent}% Spam] Kendall’s τ: {tau_value:.4f} | Time taken: {elapsed:.2f} seconds")


# === Bribery Resistance Evaluation ===
def compute_wealth(df, item_id, rankings):
    n_ratings = df[df["item_id"] == item_id].shape[0]
    avg_rating = rankings.get(item_id, 0)
    print(f"Item {item_id}: n_ratings={n_ratings}, avg_rating={avg_rating}")
    return n_ratings * avg_rating

def parse_strategy_cost(txt_path, scale_by_user=False):
    total_cost = 0.0
    user_ids = set()
    with open(txt_path, "r") as f:
        for line in f:
            if "UserID" in line and "Old" in line and "New" in line:
                try:
                    parts = line.strip().split(", ")
                    user_id = int(parts[0].split(": ")[1])
                    old_val = parts[1].split(": ")[1]
                    new_val = parts[2].split(": ")[1]
                    old = float(old_val) if old_val != "None" else 0.0
                    new = float(new_val) if new_val != "None" else 0.0
                    user_ids.add(user_id)
                    total_cost += abs(new - old)
                except Exception as e:
                    print(f"⚠️ Error parsing line: {line.strip()} → {e}")
    if scale_by_user:
        total_cost *= 1
    return total_cost

# Target items
target_items = ["0971880107"]
original_wealth = {item: compute_wealth(original_df, item, rankings) for item in target_items}

print("\n📊 Original Wealth (Aggregated Average):")
print(original_wealth)

# Bribery attack sets path
base_path = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/bribery_attack_sets"
results = []

for item_id in target_items:
    item_folder = os.path.join(base_path, f"item_{item_id}")
    for fname in os.listdir(item_folder):
        if fname.endswith(".csv"):
            scenario_name = fname.replace(".csv", "")
            full_path = os.path.join(item_folder, fname)
            altered_df = pd.read_csv(full_path, names=['user_id', 'item_id', 'normalized_rating'], skiprows=1)
            txt_path = os.path.join(item_folder, scenario_name + "_changes.txt")
            strategy_cost = 0.0
            scale_by_user = "new" in scenario_name.lower()
            if os.path.exists(txt_path):
                strategy_cost = parse_strategy_cost(txt_path, scale_by_user=scale_by_user)

            rankings_altered = aggregated_ranking_algorithm(altered_df)
            attacked_wealth = compute_wealth(altered_df, item_id, rankings_altered)

            delta = attacked_wealth - original_wealth[item_id] - strategy_cost

            results.append({
                "ItemID": item_id,
                "Scenario": scenario_name,
                "OriginalWealth": original_wealth[item_id],
                "AttackedWealth": attacked_wealth,
                "StrategyCost": strategy_cost,
                "Delta": delta
            })
            print(f"✅ Attack simulated for item: {item_id} in scenario: {scenario_name}")

results_df = pd.DataFrame(results)

# Format numbers: plain unless >= 1e4
pd.set_option("display.float_format", lambda x: f"{x:.2f}" if abs(x) < 1e4 else f"{x:.2e}")

print("\n📋 All Attack Scenarios with Strategy Cost (Aggregated Average):")
print(results_df.to_string(index=False))

results_df["AbsDelta"] = results_df["Delta"].abs()
sum_abs_deltas = results_df.groupby("ItemID")["AbsDelta"].sum().reset_index()
sum_abs_deltas.rename(columns={"AbsDelta": "TotalAbsDelta"}, inplace=True)

print("\n📊 Total Sum of |Delta| per Item (Net of Strategy Cost):")
print(sum_abs_deltas.to_string(index=False))
