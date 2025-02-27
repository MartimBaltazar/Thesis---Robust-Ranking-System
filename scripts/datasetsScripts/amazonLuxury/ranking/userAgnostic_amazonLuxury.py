import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

def user_agnostic_ranking(file_path, tol=1e-6):
    # Step 1: Load dataset from JSON lines
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)

    # Step 2: Define output dictionary for item rankings
    item_rankings = {}

    # Step 3: Iterate over all unique items (products)
    for item, ratings in df.groupby("asin")["normalizedOverall"]:
        ratings = list(ratings)  # Convert pandas Series to list
        converged = False

        # Step 5: Loop until a fixed point is reached
        while not converged:
            # Step 6: Compute the mean (μᵢ) of ratings for product i
            μ_i = np.mean(ratings)

            # Step 7: Compute the standard deviation (σᵢ)
            σ_i = np.std(ratings) if len(ratings) > 1 else 0

            # Step 8: Filter out ratings that deviate significantly from the mean
            new_ratings = [r for r in ratings if (r - μ_i) ** 2 <= σ_i]

            # Step 9: Check for convergence (if no ratings were removed)
            converged = (len(new_ratings) == len(ratings))

            # Step 10: Update the ratings
            ratings = new_ratings
        
        # Step 12: Compute the final ranking as the mean of the filtered ratings
        item_rankings[item] = np.mean(ratings)

    return item_rankings

# Example usage
file_path = "/home/martim/Desktop/tese/datasets/amazon_beauty/Luxury_Beauty_5_normalized.json"  # Adjust to your actual dataset path
rankings = user_agnostic_ranking(file_path)

# Extract the ratings from the rankings
ratings = list(rankings.values())

# Step 13: Define bins for the histogram (from 0.1 to 1.0 with steps of 0.1)
bins = np.arange(0.1, 1.1, 0.1)

# Step 14: Count how many products fall into each rating bin
hist, bin_edges = np.histogram(ratings, bins=bins)

# Step 15: Plot histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=0.08, align='edge', edgecolor='black')

# Step 16: Add counts on top of each bin
for i in range(len(hist)):
    plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', fontsize=12)

plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Products', fontsize=12)
plt.title('User-Agnostic Amazon Product Rankings Distribution', fontsize=14)
plt.xticks(bins)  # Ensure the x-ticks correspond to the rating bins
plt.tight_layout()
plt.show()
