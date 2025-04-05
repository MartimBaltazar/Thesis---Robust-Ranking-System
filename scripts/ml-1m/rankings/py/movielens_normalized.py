import pandas as pd

# File paths (modify these as needed)
input_file = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/ratings.dat"
output_file = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/normalized_ratings.dat"

# Load data (assuming "::" separator and no header)
df = pd.read_csv(input_file, sep="::", names=["UserID", "MovieID", "Rating", "Timestamp"], engine="python")

# Normalize ratings: r' = r / 5
df["NormalizedRating"] = df["Rating"] / 5

# Save with single-character separator first (to avoid pandas error)
temp_file = "/home/martimsbaltazar/Desktop/tese/datasets/ml-1m/temp_ratings.dat"
df.to_csv(temp_file, sep="|", index=False, header=False)

# Replace "|" with "::" manually
with open(temp_file, "r") as f:
    data = f.read().replace("|", "::")

with open(output_file, "w") as f:
    f.write(data)

print(f"Normalization complete. Saved as '{output_file}'.")
