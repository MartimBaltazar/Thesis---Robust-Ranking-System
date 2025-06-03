import pandas as pd
import numpy as np
import os
import random

from reputationBipartite_ml_1m import bipartite_ranking_algorithm  # Adjust import as needed

# === Load MovieLens 1M dataset with NormalizedRating ===
file_path = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/normalized_ratings.dat"
df = pd.read_csv(file_path, sep="::", engine="python", 
                 names=["UserID", "MovieID", "Rating", "Timestamp", "NormalizedRating"])

# === Select 3 items: high, mid, low popularity (min 10 ratings) ===
item_counts = df["MovieID"].value_counts()
eligible_items = item_counts[item_counts >= 10]
sorted_items = eligible_items.sort_values(ascending=False)

high = sorted_items.index[0]
mid = sorted_items.index[len(sorted_items) // 2]
low = sorted_items.index[-1]
target_items = [high, mid, low]

print(f"Selected items: High - {high}, Mid - {mid}, Low - {low}")

# === Output folder setup ===
output_root = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/bribery_attack_sets"
os.makedirs(output_root, exist_ok=True)

# === Compute reputations ===
print("Computing reputations on the clean dataset...")
_, userReputation = bipartite_ranking_algorithm(df)

# === Attack simulation function ===
def simulate_attack(df, target_item, mode="push", method="same", strategy="random", reputation_dict=None, percentage=0.1, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    modified_df = df.copy()
    changes = []

    item_ratings = df[df["MovieID"] == target_item]
    original_users = item_ratings["UserID"].unique()
    n_target = max(1, int(len(original_users) * percentage))

    if method == "same":
        if strategy == "random":
            selected_users = list(item_ratings.sample(n=n_target)["UserID"].values)
        elif strategy == "targeted":
            if reputation_dict is None:
                raise ValueError("Reputation dictionary required for targeted attack.")

            # Sort users who rated the item by their reputation (descending)
            sorted_users = sorted(
                original_users,
                key=lambda u: reputation_dict.get(u, 0),
                reverse=True
            )
            selected_users = sorted_users[:n_target]

        for user_id in selected_users:
            mask = (modified_df["UserID"] == user_id) & (modified_df["MovieID"] == target_item)
            old_rating = modified_df.loc[mask, "Rating"].values[0]
            new_rating = 5 if mode == "push" else 1
            new_normalized = new_rating / 5 

            modified_df.loc[mask, "Rating"] = new_rating
            modified_df.loc[mask, "NormalizedRating"] = new_normalized
            changes.append((user_id, old_rating, new_rating))

    else:  # method == "new"
        max_user_id = df["UserID"].max()
        new_user_id = max_user_id + 1
        timestamp = int(df["Timestamp"].mean())
        new_rating = 5 if mode == "push" else 1
        new_normalized = new_rating / 5 

        for _ in range(n_target):
            new_row = pd.DataFrame(
                [[new_user_id, target_item, new_rating, timestamp, new_normalized]],
                columns=["UserID", "MovieID", "Rating", "Timestamp", "NormalizedRating"]
            )
            modified_df = pd.concat([modified_df, new_row], ignore_index=True)
            changes.append((new_user_id, None, new_rating))
            new_user_id += 1

    return modified_df, changes

# === Define scenarios including new targeted strategy ===
scenarios = [
    ("nuke", "same", "random"),
    ("nuke", "new", "random"),
    ("push", "same", "random"),
    ("push", "new", "random"),
    ("nuke", "same", "targeted"),
    ("push", "same", "targeted"),
]

# === Generate folders and files for each item and scenario ===
for item_id in target_items:
    folder_name = os.path.join(output_root, f"item_{item_id}")
    os.makedirs(folder_name, exist_ok=True)

    for mode, method, strategy in scenarios:
        print(f"Simulating: item {item_id}, mode={mode}, method={method}, strategy={strategy}")
        attacked_df, change_log = simulate_attack(
            df, item_id,
            mode=mode,
            method=method,
            strategy=strategy,
            reputation_dict=userReputation if strategy == "targeted" else None,
            percentage=0.1
        )

        file_prefix = f"{mode}_{method}_{strategy}"
        csv_path = os.path.join(folder_name, f"{file_prefix}.csv")
        txt_path = os.path.join(folder_name, f"{file_prefix}_changes.txt")

        attacked_df.to_csv(csv_path, index=False)
        
        with open(txt_path, "w") as f:
            for user_id, old, new in change_log:
                old_norm = old / 5.0 if old is not None else "None"
                new_norm = new / 5.0 if new is not None else "None"
                f.write(f"UserID: {user_id}, Old: {old_norm}, New: {new_norm}\n")


print("✔️ All bribery attack datasets and logs generated.")
