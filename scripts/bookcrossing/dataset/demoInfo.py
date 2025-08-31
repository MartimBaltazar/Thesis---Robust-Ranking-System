import pandas as pd
from collections import defaultdict

# --- Function to describe ratings datasets ---
def describe_dataset(path, name, sep=','):
    df = pd.read_csv(path, sep=sep)
    df.columns = ['user_id', 'item_id', 'normalized_rating']
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    num_ratings = len(df)
    print(f"\n=== {name} ===")
    print(f"Unique users:   {num_users}")
    print(f"Unique items:   {num_items}")
    print(f"Total ratings:  {num_ratings}")
    return df

# --- Function to compute age group counts ---
def count_age_groups(path, name, sep=","):
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip()
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df.dropna(subset=['Age'])
    df['Age'] = df['Age'].astype(int)
    df['User-ID'] = df['User-ID'].astype(int)

    age_ranges = {
        "<18":    lambda age: age < 18,
        "18–24":  lambda age: 18 <= age <= 24,
        "25–34":  lambda age: 25 <= age <= 34,
        "35–44":  lambda age: 35 <= age <= 44,
        "45–54":  lambda age: 45 <= age <= 54,
        "≥55":    lambda age: age >= 55,
    }

    age_groups = defaultdict(set)
    for _, row in df.iterrows():
        user_id = row['User-ID']

        age = row['Age']
        for label, condition in age_ranges.items():
            if condition(age):
                age_groups[label].add(user_id)
                break

    counts = {label: len(age_groups.get(label, [])) for label in age_ranges}
    return name, counts

# --- Paths ---
short_ratings_path = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Shortened_Ratings.csv"
full_ratings_path  = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Ratings.csv"
short_users_path   = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Shortened_Users.csv"
full_users_path    = "/home/martimsbaltazar/Desktop/tese/datasets/BookCrossing/Users.csv"

# --- Ratings datasets ---
describe_dataset(full_ratings_path, "Full Ratings Dataset", sep=';')
describe_dataset(short_ratings_path, "Shortened Ratings Dataset", sep=',')

# --- Age group distributions ---
datasets = [
    count_age_groups(full_users_path, "Full Users", sep=','),
    count_age_groups(short_users_path, "Shortened Users", sep=',')
]

print(f"\n=== Age Group Comparison ===")
print(f"{'Age Group':<8} | {'Full Users':>12} | {'Shortened Users':>16}")
print("-" * 42)
age_labels = ["<18", "18–24", "25–34", "35–44", "45–54", "≥55"]
full_counts = datasets[0][1]
short_counts = datasets[1][1]

for label in age_labels:
    full_count = full_counts.get(label, 0)
    short_count = short_counts.get(label, 0)
    print(f"{label:<8} | {full_count:12} | {short_count:16}")
