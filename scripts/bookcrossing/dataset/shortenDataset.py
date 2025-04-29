import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load datasets
ratings = pd.read_csv('/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Ratings.csv', sep=';', skipinitialspace=True)
users = pd.read_csv('/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Users.csv', sep=',', skipinitialspace=True)

# Step 1: Filter Users.csv to only keep users with a valid integer Age
users['Age'] = pd.to_numeric(users['Age'], errors='coerce')
users = users.dropna(subset=['Age'])
users['Age'] = users['Age'].astype(int)

# Step 2: Filter Ratings.csv to only include users present in the cleaned Users.csv
valid_user_ids = set(users['User-ID'])
ratings = ratings[ratings['User-ID'].isin(valid_user_ids)]

# Step 3: Apply 5-core filtering (users with at least 5 ratings)
def filter_5_core(ratings_df):
    while True:
        user_counts = ratings_df['User-ID'].value_counts()
        to_keep = user_counts[user_counts >= 5].index
        new_ratings_df = ratings_df[ratings_df['User-ID'].isin(to_keep)]
        if len(new_ratings_df) == len(ratings_df):
            break
        ratings_df = new_ratings_df
    return ratings_df

ratings = filter_5_core(ratings)

# Step 4: Update Users.csv again to only include users that survived after 5-core filtering
valid_user_ids = set(ratings['User-ID'])
users = users[users['User-ID'].isin(valid_user_ids)]

# Step 5: If more than 6000 users, randomly sample 6000 users
if len(users) > 6000:
    sampled_user_ids = np.random.choice(list(users['User-ID']), size=6000, replace=False)
    users = users[users['User-ID'].isin(sampled_user_ids)]
    ratings = ratings[ratings['User-ID'].isin(sampled_user_ids)]

# Final output
users.to_csv('/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Shortened_Users.csv', index=False)
ratings.to_csv('/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Shortened_Ratings.csv', index=False)

# Print final stats
print(f"Number of users: {len(users)}")
print(f"Number of ratings: {len(ratings)}")
