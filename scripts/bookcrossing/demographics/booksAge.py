
from collections import defaultdict

# Load and inspect the contents of the uploaded file to understand its structure
file_path = '/home/martim/Desktop/tese/datasets/book_crossing/users_info.dat'

# Define the age ranges
age_ranges = {
    "< 18": lambda age: age < 18,
    "18-24": lambda age: 18 <= age <= 24,
    "25-34": lambda age: 25 <= age <= 34,
    "35-44": lambda age: 35 <= age <= 44,
    "45-49": lambda age: 45 <= age <= 49,
    "50-55": lambda age: 50 <= age <= 55,
    "> 55": lambda age: age > 55,
}

# Initialize a dictionary to store counts for each range
age_counts = defaultdict(int)

# Parse the file and count ages within each range
with open(file_path, 'r') as file:
    # Skip the header
    next(file)
    
    for line in file:
        fields = line.strip().split('\t')
        # Extract and process the age field
        try:
            age = int(fields[2])  # Convert age to integer
            for range_label, condition in age_ranges.items():
                if condition(age):
                    age_counts[range_label] += 1
                    break
        except (ValueError, IndexError):
            # Skip lines with missing or invalid age data
            continue

# Convert defaultdict to a standard dict for easier readability
age_counts = dict(age_counts)
print(age_counts)
