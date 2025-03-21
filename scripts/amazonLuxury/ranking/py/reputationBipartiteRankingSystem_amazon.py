import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

def bipartite_ranking_algorithm(file_path, lambda_factor=0.3, tol=1e-6, max_iter=2):
    # Load dataset
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)

    # Get unique users and items
    users = df["reviewerID"].unique()
    items = df["asin"].unique()

    # Initialize user reputations (equal for all users)
    user_reputation = {user: 1.0 for user in users}

    for iteration in range(max_iter):
        prev_item_rankings = {} if iteration == 0 else item_rankings.copy()

        # Update item rankings
        item_rankings = {}
        for item in items:
            item_ratings = df[df["asin"] == item]
            users_who_rated = item_ratings["reviewerID"].values

            if len(users_who_rated) == 0:
                continue

            weighted_sum = sum(user_reputation[u] * r for u, r in zip(users_who_rated, item_ratings["normalizedOverall"].values))
            total_weight = len(users_who_rated)

            item_rankings[item] = weighted_sum / total_weight if total_weight > 0 else 0

        # Update user reputations
        for user in users:
            user_ratings = df[df["reviewerID"] == user]
            items_rated = user_ratings["asin"].values

            if len(items_rated) == 0:
                continue

            rating_errors = [abs(r - item_rankings[i]) for i, r in zip(items_rated, user_ratings["normalizedOverall"].values)]
            avg_error = sum(rating_errors) / len(rating_errors)

            user_reputation[user] = 1 - lambda_factor * avg_error
            user_reputation[user] = max(user_reputation[user], 0)  # Ensure non-negative reputation

        # Check for convergence
        if iteration > 0:
            ranking_diff = sum(abs(prev_item_rankings[i] - item_rankings[i]) for i in items if i in prev_item_rankings)
            if ranking_diff < tol:
                break

    return item_rankings

# Example usage
file_path = "/home/martim/Desktop/tese/datasets/amazon_beauty/Luxury_Beauty_5_normalized.json"  # Replace with your dataset path
rankings = bipartite_ranking_algorithm(file_path)

# Extract ratings from the rankings
ratings = list(rankings.values())

# Define bins for the ratings (from 0.1 to 1.0 with steps of 0.1)
bins = np.arange(0.1, 1.1, 0.1)

# Count how many items fall into each rating bin
hist, bin_edges = np.histogram(ratings, bins=bins)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=0.08, align='edge', edgecolor='black')

# Add counts on top of each bin
for i in range(len(hist)):
    plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', fontsize=12)

plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Items', fontsize=12)
plt.title('Distribution of reputation-based Amazon Item Ratings', fontsize=14)
plt.xticks(bins)  # Ensure the x-ticks correspond to the rating bins
plt.tight_layout()
plt.show()
