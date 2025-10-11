import json
import os
import random
import numpy as np
from collections import defaultdict
import pandas as pd

from reputationBipartite_goodreads import bipartite_ranking_algorithm  

# === Load your custom dataset ===
file_path = "/home/martimsbaltazar/Desktop/tese/datasets/goodreads/goodreads_reviews_spoiler.json"  # Replace with your dataset path

# Load dataset
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)

# Get unique users and items
users = df["user_id"].unique()
items = df["book_id"].unique()

# Normalize the 'overall' column to range [0, 1]
min_rating = 0
max_rating = 5
print(df["rating"].min(), df["rating"].max())

df["normalizedOverall"] = (df["rating"] - min_rating + 1) / (max_rating - min_rating + 1)

# === Get item popularity and select high/mid/low popularity items ===
item_counts = df['book_id'].value_counts()
eligible_items = item_counts[item_counts >= 10]
sorted_items = eligible_items.sort_values(ascending=False)

high = sorted_items.index[0]
mid = sorted_items.index[len(sorted_items) // 2]
low = sorted_items.index[-1]
target_items = [high, mid, low] # add medium and low

print(f"Selected items: High - {high}, Mid - {mid}, Low - {low}")

# === Placeholder for reputation computation ===
# Replace with your own reputation algorithm

# === Compute reputations ===
print("Computing reputations on the clean dataset...")
_, userReputation = bipartite_ranking_algorithm(df)

# === Output folder setup ===
output_root = "/home/martimsbaltazar/Desktop/tese/datasets/goodreads/bribery_attack_sets3"
os.makedirs(output_root, exist_ok=True)

def simulate_attack_dict(data, target_item, mode="push", method="same", strategy="random",
                         reputation_dict=None, percentage=0.1, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    modified_data = data.copy()
    changes = []

    item_reviews = [r for r in data if r['book_id'] == target_item]
    original_users = list({r['user_id'] for r in item_reviews})
    n_target = max(1, int(len(original_users) * percentage))

    def normalize(rating): return (rating + 1) / 6

    if method == "same":
        if strategy == "random":
            selected_users = random.sample(original_users, n_target)
        elif strategy == "targeted":
            if reputation_dict is None:
                raise ValueError("Reputation dictionary required for targeted attack.")
            sorted_users = sorted(original_users, key=lambda u: reputation_dict.get(u, 0), reverse=True)
            selected_users = sorted_users[:n_target]

        for review in modified_data:
            if review['book_id'] == target_item and review['user_id'] in selected_users:
                old_rating = review['rating']
                new_rating = 5 if mode == "push" else 0
                review['rating'] = new_rating
                review['normalizedOverall'] = normalize(new_rating)
                changes.append((review['user_id'], normalize(old_rating), normalize(new_rating)))


    else:  # method == "new"
        new_reviews = []
        max_uid_base = hash(max([r['user_id'] for r in data], key=hash)) % (10**6)
        new_uid_counter = 0
        new_rating = 5 if mode == "push" else 0
        new_normalized = normalize(new_rating)
        timestamp = max(r['timestamp'] for r in data)

        for _ in range(n_target):
            new_user_id = f"synthetic_user_{max_uid_base + new_uid_counter}"
            new_review = {
                "user_id": new_user_id,
                "timestamp": timestamp,
                "review_sentences": [],
                "rating": new_rating,
                "normalizedOverall": new_normalized,
                "has_spoiler": False,
                "book_id": target_item,
                "review_id": f"synthetic_review_{random.getrandbits(64):x}"
            }
            new_reviews.append(new_review)
            changes.append((new_user_id, None, new_normalized))
            new_uid_counter += 1

        modified_data.extend(new_reviews)

    return modified_data, changes



# === Define scenarios ===
scenarios = [
    ("nuke", "same", "random"),
    ("nuke", "new", "random"),
    ("push", "same", "random"),
    ("push", "new", "random"),
    ("nuke", "same", "targeted"),
    ("push", "same", "targeted"),
]

# === Run simulation and save outputs ===
for item_id in target_items:
    folder_name = os.path.join(output_root, f"item_{item_id}")
    os.makedirs(folder_name, exist_ok=True)

    for mode, method, strategy in scenarios:
        attacked_data, change_log = simulate_attack_dict(
            data, item_id,
            mode=mode,
            method=method,
            strategy=strategy,
            reputation_dict=userReputation if strategy == "targeted" else None,
            percentage=0.1
        )
        
        df_attack = pd.DataFrame(attacked_data)
        
        #         # --- debug snippet (run in your notebook) ---
        # import numpy as np
        # from pprint import pprint

        # item = str(11870085)  # ensure string consistent

        # # 1) Basic counts & means (raw rating + normalized) before vs after
        # orig_mask = (df["book_id"] == item)
        # attack_mask = (df_attack["book_id"] == item)

        # print("Counts: original, altered:", orig_mask.sum(), attack_mask.sum())

        # orig_raw_mean = df.loc[orig_mask, "rating"].astype(float).mean()
        # attack_raw_mean = df_attack.loc[attack_mask, "rating"].astype(float).mean()
        # orig_norm_mean = df.loc[orig_mask, "normalizedOverall"].astype(float).mean()
        # attack_norm_mean = df_attack.loc[attack_mask, "normalizedOverall"].astype(float).mean()

        # print("Raw rating mean: original {:.6f}  altered {:.6f}".format(orig_raw_mean, attack_raw_mean))
        # print("Normalized mean: original {:.6f}  altered {:.6f}".format(orig_norm_mean, attack_norm_mean))

        # # 2) Confirm which users changed and how many changes the change_log reports
        # # If you have change_log from simulate_attack_dict (list of (user, old_norm, new_norm))
        # changed_users = [u for u, old, new in change_log]
        # print("Number of changed users in change_log:", len(changed_users))
        # print("Sample changed users:", changed_users[:30])

        # # 3) For every changed user, print original rating/normalised and altered values in df and df_attack
        # rows = []
        # for u in changed_users:
        #     before = df[(df["user_id"] == u) & (df["book_id"] == item)]
        #     after  = df_attack[(df_attack["user_id"] == u) & (df_attack["book_id"] == item)]
        #     b_raw = before["rating"].iloc[0] if not before.empty else None
        #     b_norm = float(before["normalizedOverall"].iloc[0]) if not before.empty else None
        #     a_raw = after["rating"].iloc[0] if not after.empty else None
        #     a_norm = float(after["normalizedOverall"].iloc[0]) if not after.empty else None
        #     rows.append((u, b_raw, b_norm, a_raw, a_norm))

        # print("\nChanged users (user, raw_before, norm_before, raw_after, norm_after):")
        # pprint(rows[:50])

        # # 4) Check if ANY changed user had a rating decreased (should be none for push)
        # decreased = [r for r in rows if r[3] is not None and r[1] is not None and float(r[3]) < float(r[1])]
        # if decreased:
        #     print("\nUsers whose raw rating decreased (unexpected):")
        #     pprint(decreased[:20])
        # else:
        #     print("\nNo changed users decreased their raw rating (as expected).")

        # # 5) Compare set of users who rated this item in original vs altered
        # orig_raters = set(df.loc[orig_mask, "user_id"].astype(str).unique())
        # alter_raters = set(df_attack.loc[attack_mask, "user_id"].astype(str).unique())
        # print("\nRaters counts: orig={}, altered={}, difference={}".format(len(orig_raters), len(alter_raters), len(alter_raters - orig_raters)))
        # if orig_raters != alter_raters:
        #     print("Raters present in altered but not original (sample):", list(alter_raters - orig_raters)[:10])
        #     print("Raters present in original but not altered (sample):", list(orig_raters - alter_raters)[:10])
        # else:
        #     print("Raters sets are identical.")


        # # 7) Quick sanity: recompute simple mean from change_log differences
        # # compute sum of old_norms and sum of new_norms from the change_log for these users
        # sum_old = sum(old for _, old, new in change_log if old is not None)
        # sum_new = sum(new for _, old, new in change_log if new is not None)
        # print("\nChange_log sums: sum_old_norms {:.6f}, sum_new_norms {:.6f}, delta {:.6f}".format(sum_old, sum_new, sum_new - sum_old))


        file_prefix = f"{mode}_{method}_{strategy}"
        json_path = os.path.join(folder_name, f"{file_prefix}.json")
        txt_path = os.path.join(folder_name, f"{file_prefix}_changes.txt")

        
        with open(json_path, "w") as f:
            for record in attacked_data:
                f.write(json.dumps(record) + "\n")


        with open(txt_path, "w") as f:
            for user_id, old, new in change_log:
                old_str = str(old) if old is not None else "None"
                new_str = str(new) if new is not None else "None"
                f.write(f"UserID: {user_id}, Old: {old_str}, New: {new_str}\n")

print("✔️ All bribery attack datasets and logs generated.")