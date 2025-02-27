import pandas as pd

# Define min and max ratings
r_min = 1
r_max = 10

# Load the dataset
file_path = "/home/martim/Desktop/tese/datasets/book_crossing/book_ratings.dat"
df = pd.read_csv(file_path, sep="\t", header=None, names=["user_id", "item_id", "rating"])

# Convert ratings to numeric type
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Apply normalization formula
df["normalized_rating"] = (df["rating"] - r_min + 1) / (r_max - r_min + 1)

# Save the normalized dataset
output_path = "/home/martim/Desktop/tese/datasets/book_crossing/book_ratings_normalized.dat"
df.to_csv(output_path, sep="\t", index=False)

print(f"Normalization complete. Saved to {output_path}")
