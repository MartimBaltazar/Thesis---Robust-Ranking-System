import pandas as pd
import numpy as np
import networkx as nx
import zlib
import pickle
import os
import matplotlib.pyplot as plt
from itertools import combinations
import sys

# File to store precomputed compressed sizes
CACHE_FILE = "compressed_sizes.pkl"

# Load dataset
import pandas as pd

def load_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0, usecols=[0, 1, 2, 3])  # Ignore extra columns
    return df

# Compression Similarity with precomputed sizes
def compression_similarity(ratings_u, ratings_v, compressed_sizes):
    u_string = "".join(f"{k}:{v}" for k, v in sorted(ratings_u.items()))
    v_string = "".join(f"{k}:{v}" for k, v in sorted(ratings_v.items()))
    c_uv = len(zlib.compress((u_string + v_string).encode()))
    c_u = compressed_sizes[u_string]
    c_v = compressed_sizes[v_string]
    return 1 - (c_uv - min(c_u, c_v)) / max(c_u, c_v)

# Kolmogorov Similarity with precomputed sizes
def kolmogorov_similarity(ratings_u, ratings_v, compressed_sizes):
    u_string = "".join(f"{k}:{v}" for k, v in sorted(ratings_u.items()))
    v_string = "".join(f"{k}:{v}" for k, v in sorted(ratings_v.items()))
    c_u = compressed_sizes[u_string]
    c_v = compressed_sizes[v_string]
    return 1 / (1 + abs(c_u - c_v))

# Compute user similarity matrix and construct graph (with precomputed sizes)
def compute_similarity_matrix(user_ratings, similarity_measure, compressed_sizes):
    similarity_graph = nx.Graph()
    for (u, v) in combinations(user_ratings.keys(), 2):
        sim = similarity_measure(user_ratings[u], user_ratings[v], compressed_sizes)
        if sim > 0.9:
            print("newEdge")
            similarity_graph.add_edge(u, v, weight=sim)
    
    return similarity_graph

# Detect user clusters from similarity graph
def detect_groups(similarity_graph):
    return list(nx.connected_components(similarity_graph))

# Compute reputation-based ranking for each group
def reputation_based_ranking(df, user_groups, lambda_factor=0.3, tol=1e-6, max_iter=10):
    item_rankings = {item: 0.5 for item in df["item_id"].unique()}  # Default starting ranking
    user_reputation = {user: 1.0 for user in df["user_id"].unique()}

    for iteration in range(max_iter):
        print("Iteration", iteration)
        prev_rankings = item_rankings.copy()

        # Compute rankings per cluster
        cluster_rankings = {}
        for group in user_groups:
            group_users = list(group)
            group_df = df[df["user_id"].isin(group_users)]

            for item, group_item in group_df.groupby("item_id"):
                users = group_item["user_id"].values
                weighted_sum = sum(user_reputation[u] * r for u, r in zip(users, group_item["normalized_rating"].values))
                total_weight = sum(user_reputation[u] for u in users)
                cluster_rankings.setdefault(item, {})[frozenset(group)] = weighted_sum / total_weight if total_weight > 0 else None

        # Assign rankings: if a product is unrated in a cluster, inherit from nearest cluster
        for item in item_rankings.keys():
            assigned = False
            for group in user_groups:
                if frozenset(group) in cluster_rankings.get(item, {}):
                    item_rankings[item] = cluster_rankings[item][frozenset(group)]
                    assigned = True
                    break
            if not assigned:  # Inherit from the closest cluster with a ranking
                for other_item in cluster_rankings:
                    for cluster in cluster_rankings[other_item]:
                        if cluster_rankings[other_item][cluster] is not None:
                            item_rankings[item] = cluster_rankings[other_item][cluster]
                            break
        # Update user reputations within each cluster
        for user in df["user_id"].unique():
            user_ratings = df[df["user_id"] == user]
            items_rated = user_ratings["item_id"].values

            if len(items_rated) == 0:
                continue

            rating_errors = [abs(r - item_rankings[i]) for i, r in zip(items_rated, user_ratings["normalized_rating"].values)]
            avg_error = sum(rating_errors) / len(rating_errors)
            
            user_reputation[user] = max(1 - lambda_factor * avg_error, 0)

        print("Iteration", iteration, "done")
        
        # Check for convergence
        ranking_diff = sum(abs(prev_rankings[i] - item_rankings[i]) for i in item_rankings if prev_rankings[i] is not None)
        if iteration > 0 and ranking_diff < tol:
            break

    return item_rankings

# Main execution
file_path = "/home/martim/Desktop/tese/datasets/book_crossing/book_ratings_normalized.dat"
df = load_dataset(file_path)

# Prepare user ratings
user_ratings = {user: dict(zip(group["item_id"], group["normalized_rating"])) for user, group in df.groupby("user_id")}

# Compute or load compressed sizes
compressed_sizes = compute_compressed_sizes(user_ratings)
print("comprimido")
# Choose similarity measure: compression_similarity or kolmogorov_similarity
similarity_graph = compute_similarity_matrix(user_ratings, kolmogorov_similarity, compressed_sizes)

# Detect user clusters from similarity graph
user_groups = list(nx.connected_components(similarity_graph))

# Print the number of detected clusters
print(f"User clusters: {len(user_groups)}")

# Compute rankings
rankings = reputation_based_ranking(df, user_groups)

# Extract rankings
ratings = list(rankings.values())

# Plot distribution
bins = np.arange(0.1, 1.1, 0.1)
hist, bin_edges = np.histogram(ratings, bins=bins)

plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist, width=0.08, align='edge', edgecolor='black')

# Add counts on top of bars
for i in range(len(hist)):
    plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', fontsize=12)

plt.xlabel('Rating', fontsize=12)
plt.ylabel('Number of Books', fontsize=12)
plt.title('Multipartite Reputation-Based Book Rankings', fontsize=14)
plt.xticks(bins)
plt.tight_layout()
plt.show()
