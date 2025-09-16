import json
import os
import random
import numpy as np
from collections import defaultdict

# === Load your custom dataset ===
with open("/home/martimsbaltazar/Desktop/tese/datasets/goodreads/goodreads_reviews_spoiler.json", "r") as f:
       data = [json.loads(line) for line in f]
       
# === Convert to DataFrame for easier handling ===
import pandas as pd
df = pd.DataFrame(data)

# === Get item popularity and select high/mid/low popularity items ===
item_counts = df['book_id'].value_counts()
eligible_items = item_counts[item_counts >= 10]
sorted_items = eligible_items.sort_values(ascending=False)

high = sorted_items.index[0]
mid = sorted_items.index[len(sorted_items) // 2]
low = sorted_items.index[-1]
target_items = [high, mid, low]

print(f"Selected items: High - {high}, Mid - {mid}, Low - {low}")

# === Placeholder for reputation computation ===
# Replace with your own reputation algorithm
userReputation = defaultdict(lambda: 0.5)  # Dummy: everyone has 0.5 rep

# === Output folder setup ===
output_root = "/home/martimsbaltazar/Desktop/tese/datasets/goodreads/bribery_attack_sets2"
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
            sorted_users = sorted(original_users, key=lambda u: reputation_dict.get(u, 0), reverse=(mode == "push"))
            selected_users = sorted_users[:n_target]

        for review in modified_data:
            if review['book_id'] == target_item and review['user_id'] in selected_users:
                old_rating = review['rating']
                new_rating = 1 if mode == "push" else 1/6
                review['rating'] = new_rating
                review['normalizedOverall'] = new_rating
                changes.append((review['user_id'], normalize(old_rating), new_rating))


    else:  # method == "new"
        new_reviews = []
        max_uid_base = hash(max([r['user_id'] for r in data], key=hash)) % (10**6)
        new_uid_counter = 0
        new_rating = 1 if mode == "push" else 1/6
        new_normalized = new_rating
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
            changes.append((new_user_id, None, new_rating))
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