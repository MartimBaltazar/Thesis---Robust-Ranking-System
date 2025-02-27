import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

def aggregated_ranking_algorithm(file_path):
    # Load dataset
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)

    # Compute item rankings as the average of normalized ratings
    item_rankings = df.groupby("asin")["normalizedOverall"].mean().to_dict()

    return item_rankings

# Example usage
file_path = "/home/martim/Desktop/tese/datasets/amazon_beauty/Luxury_Beauty_5_normalized.json"  # Replace with your dataset path
rankings = aggregated_ranking_algorithm(file_path)

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
plt.title('Distribution of Aggregated Amazon Item Ratings', fontsize=14)
plt.xticks(bins)  # Ensure the x-ticks correspond to the rating bins
plt.tight_layout()
plt.show()
