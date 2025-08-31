import os
import pandas as pd
import numpy as np
import time
import json
from scipy.stats import kendalltau

# === Aggregated Average Ranking Algorithm ===
def aggregated_ranking_algorithm(df):
    return df.groupby("book_id")["normalizedOverall"].mean().to_dict()

# === Kendall’s τ computation ===
def compute_kendall_tau(rankings1, rankings2):
    common_items = list(set(rankings1.keys()) & set(rankings2.keys()))
    if len(common_items) < 2:
        return 0
    r1 = [rankings1[i] for i in common_items]
    r2 = [rankings2[i] for i in common_items]
    tau, _ = kendalltau(r1, r2)
    return tau

# === Load Goodreads dataset ===
file_path = "/home/martimsbaltazar/Desktop/tese/datasets/goodreads/goodreads_reviews_spoiler.json"

data = []
with open(file_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        rating = (entry["rating"] - 1) / 4  # normalize to [0,1]
        data.append({
            "user_id": entry["user_id"],
            "book_id": entry["book_id"],
            "normalizedOverall": rating
        })

df = pd.DataFrame(data)

# Compute original rankings
rankings = aggregated_ranking_algorithm(df)

# === Robustness (spam injection) ===
# spam_dir = "/home/martimsbaltazar/Desktop/tese/datasets/goodreads/spam_versions_goodreads"
# ratios = [10, 30, 50, 70]

# print("=== Robustness Analysis (Aggregated Average, Goodreads) ===")
# for percent in ratios:
#     start_time = time.time()

#     file_name = f"reviews_with_{percent}percent_spam.json"
#     file_path = os.path.join(spam_dir, file_name)

    
#     # Load the dataset
#     with open(file_path, 'r') as f:
#         data = [json.loads(line) for line in f]
#     df_attack = pd.DataFrame(data)

#     # Get unique users and items
#     users = df_attack["user_id"].unique()
#     items = df_attack["book_id"].unique()

#     # Normalize the 'rating' column to range [0, 1]
#     min_rating = 1
#     max_rating = 5
#     df_attack["normalizedOverall"] = (df_attack["rating"] - min_rating) / (max_rating - min_rating)

#     # Compute rankings using your bipartite ranking algorithm
#     rankingsSpam = aggregated_ranking_algorithm(df_attack)

#     # Compute Kendall's tau
#     tau_value = compute_kendall_tau(rankings, rankingsSpam)

#     # Measure elapsed time
#     elapsed_time = time.time() - start_time

#     # Print the result
#     print(f"[{percent}% Spam] Kendall’s τ: {tau_value:.4f} | Time: {elapsed_time:.2f} seconds")



# === Bribery Resistance ===
import pandas as pd
import os
import sys


# === Compute original rankings and seller wealth ===
# original_rankings, _ = bipartite_ranking_algorithm(df)

def compute_wealth(df, item_id, rankings):
    item_id = str(item_id)  # Ensure string type match
    item_ratings = df[df["book_id"] == item_id]
    n_ratings = len(item_ratings)
    avg_score = rankings[item_id]
    print(f"Book {item_id} — #Ratings: {n_ratings}, Ranking Score: {avg_score}")
    return n_ratings * avg_score



def parse_strategy_cost(txt_path, scale_by_user=False):
    total_cost = 0.0
    user_ids = set()

    with open(txt_path, "r") as f:
        for line in f:
            if "UserID" in line and "Old" in line and "New" in line:
                try:
                    parts = line.strip().split(", ")
                    user_id = parts[0].split(": ")[1]
                    old_val = parts[1].split(": ")[1]
                    new_val = parts[2].split(": ")[1]
                    old = float(old_val) if old_val != "None" else 0.0
                    new = float(new_val) if new_val != "None" else 0.0
                    user_ids.add(user_id)
                    total_cost += abs(new - old)
                except Exception as e:
                    print(f"⚠️ Error parsing line: {line.strip()} → {e}")
    #alter to the multiplication factor
    if scale_by_user:
        total_cost *= 1
    return total_cost

# === Define target items ===
target_items = [11870085, 15897235, 23398702]
original_wealth = {item: compute_wealth(df, item, rankings) for item in target_items}

print("\n📊 Original Wealth:")
# print(original_wealth)

# === Iterate through each altered dataset ===
base_path = "/home/martimsbaltazar/Desktop/tese/datasets/goodreads/bribery_attack_sets"
results = []

for item_id in target_items:
    item_folder = os.path.join(base_path, f"item_{item_id}")
    for fname in os.listdir(item_folder):
        if fname.endswith(".json"):
            scenario_name = fname.replace(".json", "")
            full_path_csv = os.path.join(item_folder, fname)
            with open(full_path_csv, 'r') as f:
                data = [json.loads(line) for line in f]
            df_attack = pd.DataFrame(data)

            # Get unique users and items
            users = df_attack["user_id"].unique()
            items = df_attack["book_id"].unique()

            # Normalize the 'overall' column to range [0, 1]
            min_rating = 1
            max_rating = 5


            df_attack["normalizedOverall"] = (df_attack["rating"] - min_rating) / (max_rating - min_rating)

            # Strategy cost (from optional .txt file)
            txt_path = os.path.join(item_folder, scenario_name + "_changes.txt")
            strategy_cost = 0.0
            scale_by_user = "new" in scenario_name.lower()
            if os.path.exists(txt_path):
                strategy_cost = parse_strategy_cost(txt_path, scale_by_user=scale_by_user)
            print(f"Scenario: {scenario_name}, Strategy Cost: {strategy_cost:.2f}")
            # Run ranking algorithm on attacked dataset
            altered_rankings = aggregated_ranking_algorithm(df_attack)
            attacked_wealth = compute_wealth(df_attack, item_id, altered_rankings)

            delta = attacked_wealth - original_wealth[item_id] - strategy_cost

            results.append({
                "ItemID": item_id,
                "Scenario": scenario_name,
                "OriginalWealth": original_wealth[item_id],
                "AttackedWealth": attacked_wealth,
                "StrategyCost": strategy_cost,
                "Delta": delta
            })

# === Convert to DataFrame and display all results ===
results_df = pd.DataFrame(results)
print("\n📋 All Attack Scenarios with Strategy Cost:")
print(results_df.to_string(index=False))

# === Compute total sum of absolute delta per item ===
results_df["AbsDelta"] = results_df["Delta"].abs()
sum_abs_deltas = results_df.groupby("ItemID")["AbsDelta"].sum().reset_index()
sum_abs_deltas.rename(columns={"AbsDelta": "TotalAbsDelta"}, inplace=True)

print("\n📊 Total Sum of |Delta| per Item (Net of Strategy Cost):")
print(sum_abs_deltas.to_string(index=False))
