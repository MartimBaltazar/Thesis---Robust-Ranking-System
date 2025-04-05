from collections import defaultdict

# Initialize counters for gender, age ranges, and occupations
gender_counts = {"M": 0, "F": 0}
age_ranges = {
    "Under 18": 0,
    "18-24": 0,
    "25-34": 0,
    "35-44": 0,
    "45-49": 0,
    "50-55": 0,
    "56+": 0,
}
occupation_counts = defaultdict(int)

# Occupation mapping
occupations = {
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
    20: "writer",
}

# File path
file_path = "/home/martim/Desktop/tese/datasets/ml-1m/users.dat"

# Parse the file and update counts
with open(file_path, "r") as file:
    for line in file:
        fields = line.strip().split("::")
        if len(fields) < 5:
            continue  # Skip invalid lines
        
        # Extract fields
        gender = fields[1]
        age = int(fields[2])
        occupation = int(fields[3])

        # Update gender count
        if gender in gender_counts:
            gender_counts[gender] += 1

        # Update age range count
        if age == 1:
            age_ranges["Under 18"] += 1
        elif age == 18:
            age_ranges["18-24"] += 1
        elif age == 25:
            age_ranges["25-34"] += 1
        elif age == 35:
            age_ranges["35-44"] += 1
        elif age == 45:
            age_ranges["45-49"] += 1
        elif age == 50:
            age_ranges["50-55"] += 1
        elif age == 56:
            age_ranges["56+"] += 1

        # Update occupation count
        if occupation in occupations:
            occupation_counts[occupations[occupation]] += 1

# Display results
print("Gender Counts:", gender_counts)
print("Age Distribution:", age_ranges)
print("Occupation Distribution:")
for occupation, count in occupation_counts.items():
    print(f"  {occupation}: {count}")
