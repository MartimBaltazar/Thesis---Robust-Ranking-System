#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import json
from collections import defaultdict
import os
import sys


def bipartite_ranking_algorithm(df, lambda_factor=0.3, tol=1e-6, max_iter=50):
    users = df["UserID"].unique()
    items = df["MovieID"].unique()

    # Preprocess: Group ratings by item and user
    item_to_ratings = dict(tuple(df.groupby("MovieID")))
    user_to_ratings = dict(tuple(df.groupby("UserID")))

    # Step 1: Initialize user reputations (equal for all users)
    user_reputation = {user: 1.0 for user in users}
    prev_total_ranking = None

    for iteration in range(max_iter):
        # Step 2: Update item rankings
        item_rankings = {}
        for item in items:
            if item not in item_to_ratings:
                continue

            item_ratings = item_to_ratings[item]
            users_who_rated = item_ratings["UserID"].values
            ratings = item_ratings["NormalizedRating"].values

            if len(users_who_rated) == 0:
                continue

            weighted_sum = sum(
                float(user_reputation[u]) * float(r)
                for u, r in zip(users_who_rated, ratings)
            )
            total_weight = len(users_who_rated)

            item_rankings[item] = weighted_sum / total_weight if total_weight > 0 else 0

        # Step 5: Update user reputations
        for user in users:
            if user not in user_to_ratings:
                continue

            user_ratings = user_to_ratings[user]
            items_rated = user_ratings["MovieID"].values
            ratings = user_ratings["NormalizedRating"].values

            if len(items_rated) == 0:
                continue

            rating_errors = [
                abs(r - item_rankings[i]) for i, r in zip(items_rated, ratings)
                if i in item_rankings
            ]
            if not rating_errors:
                continue

            avg_error = sum(rating_errors) / len(rating_errors)
            user_reputation[user] = max(1 - lambda_factor * avg_error, 0)

        # Step 8: Check for convergence based on total ranking score
        total_ranking = sum(item_rankings.values())
        if prev_total_ranking is not None:
            if abs(total_ranking - prev_total_ranking) < tol:
                break
        prev_total_ranking = total_ranking

    return item_rankings, user_reputation


def compute_aggregated_average_ranking(df):
    return df.groupby("MovieID")["NormalizedRating"].mean().to_dict()


def compute_kendall_tau(ranking_1, ranking_2):
    common_items = set(ranking_1.keys()) & set(ranking_2.keys())  # Ensure only common items are compared
    
    if len(common_items) < 2:  # Need at least two rankings to compute Kendall's tau
        return 0
    
    sorted_items = sorted(common_items)  # Sort items to ensure consistent order
    
    list_1 = [ranking_1[item] for item in sorted_items]
    list_2 = [ranking_2[item] for item in sorted_items]
    
    return kendalltau(list_1, list_2).correlation


def demographic_reputation_gap(group_users, user_reputations, all_users):
    target_users = group_users & all_users
    complementary_users = all_users - target_users

    target_reps = [user_reputations[u] for u in target_users]
    comp_reps = [user_reputations[u] for u in complementary_users]

    target_avg = sum(target_reps) / len(target_reps) if target_reps else 0
    comp_avg = sum(comp_reps) / len(comp_reps) if comp_reps else 0

    return target_avg - comp_avg


def compute_wealth_dat(df, item_id, item_rankings):
    # Filter ratings for the item
    item_ratings = df[df["MovieID"] == item_id]

    # Count how many users rated it
    n_ratings = len(item_ratings)

    # Retrieve item's ranking score (average weighted normalized rating)
    ranking_score = item_rankings.get(item_id, 0)

    # Debug output
    print(f"Item {item_id} → Ratings: {n_ratings}, Ranking Score: {ranking_score:.4f}")

    # Compute wealth
    return n_ratings * ranking_score


def compute_wealth(df, item_id, rankings):
    n_ratings = df[df["MovieID"] == item_id].shape[0]
    avg_rating = rankings.get(item_id, 0)
    print(f"Item {item_id} - Ratings: {n_ratings}, Avg Rating: {avg_rating}")
    return n_ratings * avg_rating


if __name__ == "__main__":
    # === 2. Load Dataset ===
    file_path = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/normalized_ratings.dat"
    df = pd.read_csv(file_path, sep="::", engine="python", names=["UserID", "MovieID", "Rating", "Timestamp", "NormalizedRating"])

    all_users = set(df["UserID"].unique())
    items = df["MovieID"].unique()

    # === 4. Compute Item Rankings ===
    rankings, userReputation = bipartite_ranking_algorithm(df)

    print("Rankings:", rankings)
    print("User Reputations:", userReputation)

    # === 5. Visualizing the Distribution of Item Rankings ===
    ratings = list(rankings.values())
    bins = np.arange(0.1, 1.1, 0.1)
    hist, bin_edges = np.histogram(ratings, bins=bins)

    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=0.08, align='edge', edgecolor='black')

    for i in range(len(hist)):
        plt.text(bin_edges[i], hist[i] + 1, str(hist[i]), ha='center', fontsize=10)

    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Number of Items', fontsize=12)
    plt.title('Distribution of Item Ratings', fontsize=14)
    plt.xticks(bins)
    plt.tight_layout()
    plt.show()

    # === 6. Demographics ===
    age_ranges = {
        "< 18": lambda age: age < 18,
        "18-24": lambda age: 18 <= age <= 24,
        "25-34": lambda age: 25 <= age <= 34,
        "35-44": lambda age: 35 <= age <= 44,
        "45-54": lambda age: 45 <= age <= 54,
        ">= 55": lambda age: age >= 55,
    }

    gender_groups = defaultdict(set)
    age_groups = defaultdict(set)

    user_file_path = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/users.dat"

    with open(user_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split("::")
            if len(parts) != 5:
                continue

            user_id, gender, age_str, _, _ = parts
            user_id = int(user_id)
            age = int(age_str)

            for label, condition in age_ranges.items():
                if condition(age):
                    age_groups[label].add(user_id)
                    break

            gender_groups[gender].add(user_id)

    print(f"Gender Groups: {gender_groups}")
    print(f"\nAge Groups: {age_groups}")

    # === 7.1 Effectiveness ===
    aa_rankings = compute_aggregated_average_ranking(df)
    tau_value = compute_kendall_tau(rankings, aa_rankings)
    print(f"Kendall’s τ: {tau_value:.4f}")

    # === 7.2 Bias ===
    print("=== Gender-Based Reputation Gaps ===")
    gap = demographic_reputation_gap(gender_groups["F"], userReputation, all_users)
    print(f"Reputation gap (Female vs rest): {gap:.4f}")
    gap = demographic_reputation_gap(gender_groups["M"], userReputation, all_users)
    print(f"Reputation gap (Male vs rest): {gap:.4f}")

    print("\n=== Age-Based Reputation Gaps ===")
    for label in ["< 18", "18-24", "25-34", "35-44", "45-54", ">= 55"]:
        gap = demographic_reputation_gap(age_groups[label], userReputation, all_users)
        print(f"Reputation gap ({label} vs rest): {gap:.4f}")

    # === 7.3 Robustness: Spammers dataset ===
    spam_file_path = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/spam_versions/ratings_with_10percent_spam.csv"
    df_attack = pd.read_csv(spam_file_path)
    df_attack.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp', 'NormalizedRating']

    rankingsSpam, userReputationSpam = bipartite_ranking_algorithm(df_attack)
    print(rankingsSpam)

    # === 7.3.2 Robustness Kendall's Tau ===
    spam_dir = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/spam_versions"
    ratios = [10, 30, 50, 70]

    for percent in ratios:
        file_name = f"ratings_with_{percent}percent_spam.csv"
        file_path = os.path.join(spam_dir, file_name)

        df_attack = pd.read_csv(file_path)
        df_attack.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp', 'NormalizedRating']

        rankingsSpam, _ = bipartite_ranking_algorithm(df_attack)
        tau_value = compute_kendall_tau(rankings, rankingsSpam)

        print(f"Kendall’s τ with {percent}% spam: {tau_value:.4f}")

    # === 8. Compute Wealth ===
    selected_items = [2858, 1196, 593]  # Examples for computing wealth

    for item_id in selected_items:
        wealth = compute_wealth_dat(df, item_id, rankings)
        print(f"Wealth for item {item_id}: {wealth:.4f}")
