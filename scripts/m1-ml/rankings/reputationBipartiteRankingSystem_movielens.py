import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bipartite_ranking_algorithm(file_path, lambda_factor=0.3, tol=1e-6, max_iter=2):
    # Step 1: Load dataset (MovieLens format with "::" separator)
    df = pd.read_csv(file_path, sep="::", engine="python", names=["user_id", "item_id", "rating", "timestamp", "normalized_rating"])
    
    # Step 2: Get unique users and items
    users = df["user_id"].unique()
    items = df["item_id"].unique()
    
    # Step 3: Initialize user reputations (equal for all users)
    user_reputation = {user: 1.0 for user in users}
    
    for iteration in range(max_iter):
        prev_item_rankings = {} if iteration == 0 else item_rankings.copy()

        # Step 4: Update item rankings
        item_rankings = {}
        for item in items:
            item_ratings = df[df["item_id"] == item]  # Ratings given to a specific movie
            users_who_rated = item_ratings["user_id"].values  # Users who rated this movie
            
            if len(users_who_rated) == 0:
                continue
            
            # Step 5: Compute weighted sum of ratings using user reputation
            weighted_sum = sum(float(user_reputation[u]) * float(r) for u, r in zip(users_who_rated, item_ratings["normalized_rating"].values))
            total_weight = len(users_who_rated)
            
            # Step 6: Assign ranking to item
            item_rankings[item] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Step 7: Update user reputations
        for user in users:
            user_ratings = df[df["user_id"] == user]
            items_rated = user_ratings["item_id"].values
            
            if len(items_rated) == 0:
                continue
            
            # Step 8: Compute average rating error
            rating_errors = [abs(r - item_rankings[i]) for i, r in zip(items_rated, user_ratings["normalized_rating"].values)]
            avg_error = sum(rating_errors) / len(rating_errors)
            
            # Step 9: Update user reputation
            user_reputation[user] = 1 - lambda_factor * avg_error
            user_reputation[user] = max(user_reputation[user], 0)  # Ensure non-negative reputation
        
        # Step 10: Check for convergence
        if iteration > 0:
            ranking_diff = sum(abs(prev_item_rankings[i] - item_rankings[i]) for i in items if i in prev_item_rankings)
            if ranking_diff < tol:
                break
    
    return item_rankings

# Example usage
file_path = "/home/martim/Desktop/tese/datasets/ml-1m/normalized_ratings.dat"  # Adjust to your actual dataset path
rankings = bipartite_ranking_algorithm(file_path)

# Extract the ratings from the rankings
ratings = list(rankings.values())

# Step 11: Define bins for the histogram (from 0.1 to 1.0 with steps of 0.1)
bins = np.arange(0.1, 1.1, 0.1)

# Step 12: Count how many movies fall into each rating bin
hist, bin_edges = np.histogram(ratings, bins=bins)

# Step 13: Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=0.08, align='edge', edgecolor='black')

# Step 14: Add counts on top of each bin
for i in range(len(hist)):
    plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', fontsize=10)

plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.title('Distribution of Reputation Bipartite Movie Rankings', fontsize=14)
plt.xticks(bins)  # Ensure the x-ticks correspond to the rating bins
plt.tight_layout()
plt.show()
