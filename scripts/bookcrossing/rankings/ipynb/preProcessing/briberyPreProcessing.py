import pandas as pd
import numpy as np
import os
import random
from collections import defaultdict

from reputationBipartite_ml_1m import bipartite_ranking_algorithm  

# === Load BookCrossing dataset ===
file_path = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Shortened_Ratings.csv"
df = pd.read_csv(file_path, sep=',')

# Rename columns for consistency with your code
df.columns = ['user_id', 'item_id', 'normalized_rating']

# Normalize ratings if not already normalized (assuming ratings are 0-10 scale)
df['normalized_rating'] = df['normalized_rating'] / 10

# === Compute reputations using bipartite ranking algorithm ===
rankings, userReputation = bipartite_ranking_algorithm(df.rename(columns={
    'user_id': 'UserID',
    'item_id': 'MovieID',
    'normalized_rating': 'NormalizedRating'
}))

print("User reputations computed:")
print(userReputation)

# === Select 3 items with at least 10 ratings: high, mid, low popularity ===
item_counts = df['item_id'].value_counts()
eligible_items = item_counts[item_counts >= 10]
sorted_items = eligible_items.sort_values(ascending=False)

high = sorted_items.index[0]
mid = sorted_items.index[len(sorted_items) // 2]
low = sorted_items.index[-1]
target_items = [high, mid, low]

print(f"Selected items: High - {high}, Mid - {mid}, Low - {low}")

# === Prepare output folder ===
output_root = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/bribery_attack_sets"
os.makedirs(output_root, exist_ok=True)

# === Attack simulation function adapted for BookCrossing ===
def simulate_attack(df, target_item, mode="push", method="same", strategy="random", reputation_dict=None, percentage=0.2, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    modified_df = df.copy()
    changes = []

    # Find ratings for the target item
    item_ratings = df[df["item_id"] == target_item]
    original_users = item_ratings["user_id"].unique()
    n_target = max(1, int(len(original_users) * percentage))

    if method == "same":
        if strategy == "random":
            selected_users = list(item_ratings.sample(n=n_target)["user_id"].values)
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
            mask = (modified_df["user_id"] == user_id) & (modified_df["item_id"] == target_item)
            old_rating = modified_df.loc[mask, "normalized_rating"].values[0]
            # For BookCrossing normalized rating [0..1], push = +0.25, nuke = -0.25, capped between 0 and 1
            new_rating = 1 if mode == "push" else 0.1
            modified_df.loc[mask, "normalized_rating"] = new_rating
            changes.append((user_id, old_rating, new_rating))

    else:  # method == "new"
        max_user_id = df["user_id"].max()
        new_user_id = max_user_id + 1
        timestamp = 0  # No timestamp in this dataset
        new_rating = 1.0 if mode == "push" else 0.2

        for _ in range(n_target):
            new_row = pd.DataFrame(
                [[new_user_id, target_item, new_rating]],
                columns=["user_id", "item_id", "normalized_rating"]
            )
            modified_df = pd.concat([modified_df, new_row], ignore_index=True)
            changes.append((new_user_id, None, new_rating))
            new_user_id += 1

    return modified_df, changes


# === Define attack scenarios ===
scenarios = [
    ("nuke", "same", "random"),
    ("nuke", "new", "random"),
    ("push", "same", "random"),
    ("push", "new", "random"),
    ("nuke", "same", "targeted"),
    ("push", "same", "targeted"),
]

# === Run simulations for each target item and scenario ===
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
            percentage=0.2
        )

        file_prefix = f"{mode}_{method}_{strategy}"
        csv_path = os.path.join(folder_name, f"{file_prefix}.csv")
        txt_path = os.path.join(folder_name, f"{file_prefix}_changes.txt")

        attacked_df.to_csv(csv_path, index=False)

        with open(txt_path, "w") as f:
            for user_id, old, new in change_log:
                f.write(f"UserID: {user_id}, Old: {old}, New: {new}\n")

print("✔️ All bribery attack datasets and logs generated.")
