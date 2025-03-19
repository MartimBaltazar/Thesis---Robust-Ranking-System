import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def user_agnostic_ranking(file_path, tol=1e-6):
    # Step 1: Load dataset (skip header line explicitly)
    df = pd.read_csv(file_path, sep='\t', header=0, names=["user_id", "item_id", "rating", "normalized_rating"])
    
    # Step 2: Define output dictionary for item rankings
    item_rankings = {}

    # Step 3: Iterate over all unique items
    for item, ratings in df.groupby("item_id")["normalized_rating"]:
        ratings = list(ratings)  # Convert to list
        converged = False

        # Step 5: Loop until a fixed point is reached
        while not converged:
            # Step 6: Compute the mean (μᵢ) of ratings for item i
            μ_i = np.mean(ratings)
            
            # Step 7: Compute the standard deviation (σᵢ)
            σ_i = np.std(ratings) if len(ratings) > 1 else 0

            # Step 8: Filter out ratings that deviate significantly from the mean
            new_ratings = [r for r in ratings if (r - μ_i) ** 2 <= σ_i]

            # Step 9: Check for convergence (if no ratings were removed)
            converged = (len(new_ratings) == len(ratings))

            #step 10: Update the ratings
            ratings = new_ratings
        
        # Step 12: Compute the final ranking as the mean of the filtered ratings
        item_rankings[item] = np.mean(ratings)

    return item_rankings

# Example usage
file_path = "/home/martim/Desktop/tese/datasets/book_crossing/book_ratings_normalized.dat"  # Adjust to your actual dataset path
rankings = user_agnostic_ranking(file_path)

# Extract the ratings from the rankings
ratings = list(rankings.values())

bins = np.arange(0.1, 1.1, 0.1)

hist, bin_edges = np.histogram(ratings, bins=bins)

plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=0.08, align='edge', edgecolor='black')

for i in range(len(hist)):
    plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', fontsize=12)

plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Books', fontsize=12)
plt.title('User-Agnostic Book Rankings Distribution', fontsize=14)
plt.xticks(bins)  # Ensure the x-ticks correspond to the rating bins
plt.tight_layout()
plt.show()
