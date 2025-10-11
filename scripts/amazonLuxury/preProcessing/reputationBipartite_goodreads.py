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

def bipartite_ranking_algorithm(df, lambda_factor=0.5, max_iter=7):
    """
    Bipartite ranking algorithm with fixed 7 iterations and lambda_factor=0.5.
    Ignores convergence check.
    """
    users = df["user_id"].unique()
    items = df["book_id"].unique()

    # Preprocess: Group ratings by item and user
    item_to_ratings = dict(tuple(df.groupby("book_id")))
    user_to_ratings = dict(tuple(df.groupby("user_id")))

    # Initialize user reputations
    user_reputation = {user: 1.0 for user in users}

    for _ in range(max_iter):  # fixed 7 iterations
        # Step 1: Update item rankings
        item_rankings = {}
        for item in items:
            if item not in item_to_ratings:
                continue

            item_ratings = item_to_ratings[item]
            users_who_rated = item_ratings["user_id"].values
            ratings = item_ratings["normalizedOverall"].values

            if len(users_who_rated) == 0:
                continue

            weighted_sum = sum(
                float(user_reputation[u]) * float(r)
                for u, r in zip(users_who_rated, ratings)
            )
            total_weight = len(users_who_rated)

            item_rankings[item] = weighted_sum / total_weight if total_weight > 0 else 0

        # Step 2: Update user reputations
        for user in users:
            if user not in user_to_ratings:
                continue

            user_ratings = user_to_ratings[user]
            items_rated = user_ratings["book_id"].values
            ratings = user_ratings["normalizedOverall"].values

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

    return item_rankings, user_reputation
