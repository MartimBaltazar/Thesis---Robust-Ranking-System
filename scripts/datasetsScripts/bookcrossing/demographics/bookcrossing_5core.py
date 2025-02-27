from collections import defaultdict

# File paths

input_file = "/home/martim/Desktop/tese/datasets/book_crossing/book_ratings.dat"
output_file = "/home/martim/Desktop/tese/datasets/book_crossing/book_ratings_5Core.dat"

# Step 1: Read data and count user ratings
user_ratings_count = defaultdict(int)
data_lines = []

with open(input_file, 'r') as file:
    for line in file:
        user_id, item_id, rating = line.strip().split('\t')
        user_ratings_count[user_id] += 1
        data_lines.append((user_id, item_id, rating))

# Step 2: Filter users with at least 5 ratings
users_to_exclude = {user for user, count in user_ratings_count.items() if count < 5}

# Print users with less than 5 ratings
print("Users with less than 5 ratings:", users_to_exclude)

# Step 3: Write filtered data to output file
with open(output_file, 'w') as file:
    for user_id, item_id, rating in data_lines:
        if user_id not in users_to_exclude:
            file.write(f"{user_id}\t{item_id}\t{rating}\n")

print(f"Filtered data saved to {output_file}")
