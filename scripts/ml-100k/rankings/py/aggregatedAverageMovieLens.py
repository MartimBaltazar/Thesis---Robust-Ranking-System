import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def aggregated_ranking_algorithm(file_path):
    # Load dataset (assuming "::" is the separator with no headers)
    df = pd.read_csv(file_path, sep="::", engine="python", names=["user_id", "item_id", "rating", "timestamp", "normalized_rating"])
    
    # Compute item rankings as a simple average of normalized ratings
    item_rankings = df.groupby("item_id")["normalized_rating"].mean().to_dict()
    
    return item_rankings

# Example usage
rankings = aggregated_ranking_algorithm("/home/martimsbaltazar/Desktop/tese/datasets/ml-100k/normalized_rating_movielens_100k.dat")

# Extract the ratings from the rankings
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
plt.ylabel('Number of Movies', fontsize=12)
plt.title('Distribution of Aggregated Average Movie Ratings', fontsize=14)
plt.xticks(bins)  # Ensure the x-ticks correspond to the rating bins
plt.tight_layout()
plt.show()
