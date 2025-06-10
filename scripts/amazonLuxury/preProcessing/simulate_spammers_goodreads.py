import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

# === Load dataset ===
def load_jsonl_dataset(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    df["normalizedOverall"] = (df["rating"] - 1) / 4  # normalize to [0, 1]
    return df, data

# === Compute item popularity ===
def compute_item_popularity(df):
    return df["book_id"].value_counts(normalize=True)

# === Simulate random spammers (JSON-compatible) ===
def simulate_random_spammers_json(num_spammers, item_popularity_dist, user_id_start, lambda_poisson=5):
    spam_data = []
    item_ids = item_popularity_dist.index.tolist()
    item_probs = item_popularity_dist.values
    max_rating = 5

    for i in range(num_spammers):
        user_id = f"synthetic_user_{user_id_start + i}"
        num_ratings = np.random.poisson(lam=lambda_poisson) + 1
        sampled_items = np.random.choice(item_ids, size=min(num_ratings, len(item_ids)), replace=False, p=item_probs)

        for book_id in sampled_items:
            rating = np.random.randint(1, max_rating + 1)
            norm_rating = (rating - 1) / 4
            review = {
                "user_id": user_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d"),
                "review_sentences": [],
                "rating": rating,
                "normalizedOverall": norm_rating,
                "has_spoiler": False,
                "book_id": book_id,
                "review_id": f"synthetic_review_{np.random.randint(1e9):09d}"
            }
            spam_data.append(review)

    return spam_data

# === Inject spammers ===
def add_spammers_to_json_data(data, df, spammer_ratio=0.1, lambda_poisson=5):
    total_users = df["user_id"].nunique()
    num_spammers = int(np.ceil(spammer_ratio * total_users))
    item_popularity = compute_item_popularity(df)
    user_id_start = len(set(df["user_id"]))
    spam_reviews = simulate_random_spammers_json(num_spammers, item_popularity, user_id_start, lambda_poisson)
    return data + spam_reviews, spam_reviews

# === Generate spammy datasets ===
def generate_spam_versions(json_data, df, ratios, lambda_poisson=5, output_dir="spam_versions_goodreads"):
    os.makedirs(output_dir, exist_ok=True)

    for ratio in ratios:
        combined_data, spam_only = add_spammers_to_json_data(json_data, df, spammer_ratio=ratio, lambda_poisson=lambda_poisson)
        percent = int(ratio * 100)

        combined_path = os.path.join(output_dir, f"reviews_with_{percent}percent_spam.json")
        spam_only_path = os.path.join(output_dir, f"spam_only_{percent}percent.json")

        with open(combined_path, "w") as f:
            for entry in combined_data:
                f.write(json.dumps(entry) + "\n")

        with open(spam_only_path, "w") as f:
            for entry in spam_only:
                f.write(json.dumps(entry) + "\n")

        print(f"✔ Generated {percent}% spam version with {len(spam_only)} fake reviews.")

# === Run ===
if __name__ == "__main__":
    input_path = "/home/martimsbaltazar/Desktop/tese/datasets/goodreads/goodreads_reviews_spoiler.json"
    df, json_data = load_jsonl_dataset(input_path)

    spam_ratios = [0.10, 0.30, 0.50, 0.70]
    generate_spam_versions(json_data, df, spam_ratios, lambda_poisson=20)
