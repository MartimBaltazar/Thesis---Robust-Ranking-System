import csv
from collections import defaultdict

# Define the mapping of extracted countries to continents
country_to_continent = {
    # Americas
    "usa": "Americas", "united states": "Americas", "united state": "Americas",
    "canada": "Americas", "brazil": "Americas", "dominican republic": "Americas",
    "argentina": "Americas", "mexico": "Americas", "chile": "Americas",
    "venezuela": "Americas", "colombia": "Americas", "peru": "Americas",
    "puerto rico": "Americas", "uruguay": "Americas",

    # Africa
    "egypt": "Africa", "south africa": "Africa", "morocco": "Africa",
    "nigeria": "Africa", "tunisia": "Africa",

    # Asia + Oceania
    "china": "Asia/Oceania", "japan": "Asia/Oceania", "australia": "Asia/Oceania",
    "new zealand": "Asia/Oceania", "iran": "Asia/Oceania", "kuwait": "Asia/Oceania",
    "malaysia": "Asia/Oceania", "philippines": "Asia/Oceania", "phillipines": "Asia/Oceania",
    "hong kong": "Asia/Oceania", "singapore": "Asia/Oceania", "taiwan": "Asia/Oceania",
    "india": "Asia/Oceania", "indonesia": "Asia/Oceania", "south korea": "Asia/Oceania",
    "pakistan": "Asia/Oceania", "thailand": "Asia/Oceania", "vietnam": "Asia/Oceania",

    # Europe
    "france": "Europe", "germany": "Europe", "united kingdom": "Europe",
    "italy": "Europe", "netherlands": "Europe", "sweden": "Europe",
    "switzerland": "Europe", "spain": "Europe", "norway": "Europe",
    "finland": "Europe", "denmark": "Europe", "portugal": "Europe",
    "romania": "Europe", "russia": "Europe", "poland": "Europe",
    "belgium": "Europe", "austria": "Europe", "iceland": "Europe",
    "czech republic": "Europe", "bulgaria": "Europe", "cyprus": "Europe",
    "greece": "Europe", "ireland": "Europe", "slovakia": "Europe",
    "croatia": "Europe", "hungary": "Europe", "serbia": "Europe",

    # Unknown or unspecified locations
    "far away...": "Unknown", "universe": "Unknown", "burma": "Unknown", "qatar": "Unknown",
    "unknown": "Unknown"
}

# Function to calculate continent distribution
def calculate_distribution(file_path):
    continent_counts = defaultdict(int)
    total_users = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # Skip header

        for row in reader:
            if len(row) < 2:
                continue  # Skip invalid rows

            location_parts = row[1].split(',')
            country = location_parts[-1].strip().lower()  # Extract country

            continent = country_to_continent.get(country, "Unknown")
            continent_counts[continent] += 1
            total_users += 1

    # Calculate percentages
    distribution = {
        continent: (count, (count / total_users) * 100) for continent, count in continent_counts.items()
    }
    return distribution

# File path
file_path = "/home/martim/Desktop/tese/datasets/book_crossing/users_info.dat"  # Change this to the correct path if necessary

# Generate and display the results
distribution = calculate_distribution(file_path)

print("User Distribution by Continent:")
for continent, (count, percentage) in distribution.items():
    print(f"{continent}: {count} users ({percentage:.2f}%)")
