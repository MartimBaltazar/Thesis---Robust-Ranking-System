import json

def normalize_rating(rating, r_min, r_max):
    return (rating - r_min + 1) / (r_max - r_min + 1)

def normalize_json_ratings(input_file, output_file):
    # Reading the JSON data from the file
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Extracting the ratings
    ratings = [entry['overall'] for entry in data]
    
    # Finding the minimum and maximum ratings
    r_min = min(ratings)
    r_max = max(ratings)

    # Normalizing the ratings
    for entry in data:
        entry['normalizedOverall'] = normalize_rating(entry['overall'], r_min, r_max)

    # Writing the normalized data back to a new JSON file
    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

# Example usage
input_file = '/home/martim/Desktop/tese/datasets/amazon_beauty/Luxury_Beauty_5.json'  # Replace with your input JSON file path
output_file = '/home/martim/Desktop/tese/datasets/amazon_beauty/Luxury_Beauty_5_normalized.json'  # Replace with your desired output file path

normalize_json_ratings(input_file, output_file)
