import json

# Function to load JSON data from a file
def load_reviews(filename):
    with open(filename, 'r') as file:
        return [json.loads(line.strip()) for line in file]

# Function to analyze the reviews
def analyze_reviews(reviews):
    # Total number of reviews
    total_reviews = len(reviews)
    
    # Unique users based on reviewerID
    unique_users = len(set(review["reviewerID"] for review in reviews))
    
    # Calculate average rating
    avg_rating = sum(review["overall"] for review in reviews) / total_reviews
    
    # Find the most frequent reviewer
    reviewer_counts = {}
    for review in reviews:
        reviewer_id = review["reviewerID"]
        reviewer_counts[reviewer_id] = reviewer_counts.get(reviewer_id, 0) + 1
    most_frequent_reviewer = max(reviewer_counts, key=reviewer_counts.get)
    
    # Output the analysis
    print(f"Total number of reviews: {total_reviews}")
    print(f"Number of unique users: {unique_users}")
    print(f"Average rating: {avg_rating:.2f}")
    print(f"Most frequent reviewer: {most_frequent_reviewer} with {reviewer_counts[most_frequent_reviewer]} reviews")

# Example usage
if __name__ == "__main__":
    # Specify the path to your JSON file
    filename = "/home/martim/Desktop/tese/Luxury_Beauty_5.json"  # Change this to your file path
    
    # Load and analyze the reviews
    reviews = load_reviews(filename)
    analyze_reviews(reviews)
