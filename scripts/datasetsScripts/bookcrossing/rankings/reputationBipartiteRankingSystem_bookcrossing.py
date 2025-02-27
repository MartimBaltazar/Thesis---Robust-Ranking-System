import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bipartite_ranking_algorithm(file_path, lambda_factor=0.3, tol=1e-6, max_iter=2):
    # Load dataset
    df = pd.read_csv(file_path, sep='\t', header=0)  # Properly use the first row as headers
    
    # Get unique users and items used for iteration
    users = df["user_id"].unique()
    items = df["item_id"].unique()
    
    # Initialize user reputations (equal for all users)
    user_reputation = {user: 1.0 for user in users}
    
    for iteration in range(max_iter):
        prev_item_rankings = {} if iteration == 0 else item_rankings.copy()

        # Update item rankings
        item_rankings = {}
        for item in items:
            item_ratings = df[df["item_id"] == item] # Returns a subset of df, which contains all the ratings given to that specific item by different users.
            users_who_rated = item_ratings["user_id"].values # This gives the list of users who rated the specific item.
            
            if len(users_who_rated) == 0: # alterar isto, so vais faezr coisas se len » 0
                continue
            
            weighted_sum = sum(float(user_reputation[u]) * float(r) for u, r in zip(users_who_rated, item_ratings["normalized_rating"].values))
            total_weight = len(users_who_rated)
            
            item_rankings[item] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Update user reputations
        for user in users:
            user_ratings = df[df["user_id"] == user]
            items_rated = user_ratings["item_id"].values
            
            if len(items_rated) == 0:
                continue
            
            rating_errors = [abs(r - item_rankings[i]) for i, r in zip(items_rated, user_ratings["normalized_rating"].values)]
            avg_error = sum(rating_errors) / len(rating_errors)
            
            user_reputation[user] = 1 - lambda_factor * avg_error
            user_reputation[user] = max(user_reputation[user], 0)  # Ensure non-negative reputation
        
        # Check for convergence
        if iteration > 0:
            ranking_diff = sum(abs(prev_item_rankings[i] - item_rankings[i]) for i in items)
            if ranking_diff < tol:
                break
    
    return item_rankings

# Example usage
rankings = bipartite_ranking_algorithm("/home/martim/Desktop/tese/datasets/book_crossing/book_ratings_normalized.dat")

# Extract the ratings from the rankings
ratings = list(rankings.values())

# Define bins for the ratings (from 0.1 to 1.0 with steps of 0.1)
bins = np.arange(0.1, 1.1, 0.1)

# Count how many items fall into each rating bin
hist, bin_edges = np.histogram(ratings, bins=bins)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=0.08, align='edge', edgecolor='black')

# Adding numbers on top of bars
for i in range(len(hist)):
    plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', fontsize=10)

plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Items', fontsize=12)
plt.title('Distribution of Item Ratings', fontsize=14)
plt.xticks(bins)  # Ensure the x-ticks correspond to the rating bins
plt.tight_layout()
plt.show()
