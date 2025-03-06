# Define input and output file names
input_file = "/home/martim/Desktop/tese/datasets/ml-100k/u.data"
output_file = "/home/martim/Desktop/tese/datasets/ml-100k/normalized_rating_movielns_100k.dat"

# Define min and max rating
r_min = 1
r_max = 5

# Open input file, process data, and write to output file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # Split each line into components (tab-separated)
        user_id, movie_id, rating, timestamp = line.strip().split("\t")
        
        # Convert rating to float and normalize it
        rating = int(rating)
        normalized_rating = (rating - r_min + 1) / (r_max - r_min + 1)
        
        # Write the formatted output to the new file
        outfile.write(f"{user_id}::{movie_id}::{rating}::{timestamp}::{normalized_rating:.1f}\n")

print(f"Normalized ratings saved to {output_file}")
