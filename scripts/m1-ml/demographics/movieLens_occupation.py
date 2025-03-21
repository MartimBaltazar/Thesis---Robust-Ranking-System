import csv

# Define the occupation mapping based on the given dataset structure
occupation_mapping = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
}

def count_occupations(file_path):
    occupation_counts = {name: 0 for name in occupation_mapping.values()}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue  # Skip invalid lines
            
            occupation_id = int(parts[3])
            occupation_name = occupation_mapping.get(occupation_id, "other")
            occupation_counts[occupation_name] += 1

    return occupation_counts

# File path
file_path = "/home/martim/Desktop/tese/datasets/ml-1m/users.dat"

# Get occupation counts
occupation_counts = count_occupations(file_path)

# Print result
print(occupation_counts)
