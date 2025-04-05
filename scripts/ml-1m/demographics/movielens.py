import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

# Set professional style using seaborn for improved visuals
sns.set(style="whitegrid", palette="muted")

# Define color scheme for consistency
colors = {
    'gender': ['#4C72B0', '#FF6F61'],  # Blue and Light Pink
    'age': '#4C72B0',  # Soft Blue for age bars
    'occupation': '#2C3E50'  # Darker Blue for occupation bars
}

# Function to load and parse the users.dat file
def load_data(filename):
    users = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split('::')
            if len(parts) == 5:
                users.append(parts)
    return users

# Function to get the gender distribution
def gender_distribution(users):
    genders = [user[1] for user in users]
    gender_count = Counter(genders)
    return gender_count

# Function to get the age distribution
def age_distribution(users):
    age_groups = [user[2] for user in users]
    age_count = Counter(age_groups)
    return age_count

# Function to get the occupation distribution
def occupation_distribution(users):
    occupations = [user[3] for user in users]
    occupation_count = Counter(occupations)
    return occupation_count

# Function to generate and display the plots (gender as pie chart, others as bar charts)
def plot_demographics(gender_count, age_count, occupation_count):
    # Gender Distribution Pie Chart
    gender_labels = ['Male', 'Female']
    gender_values = [gender_count.get('M', 0), gender_count.get('F', 0)]
    
    plt.figure(figsize=(8, 8))
    plt.pie(gender_values, labels=gender_labels, autopct='%1.1f%%', startangle=90, colors=colors['gender'], 
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    plt.title('Sex Distribution', fontsize=18, fontweight='bold')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.savefig('gender_distribution.pdf')  # Save as PDF
    plt.close()  # Close the plot to avoid overlap with the next one

    # Age Distribution Bar Chart
    age_labels = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']
    age_values = [age_count.get(str(i), 0) for i in [1, 18, 25, 35, 45, 50, 56]]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(age_labels, age_values, color=colors['age'], edgecolor='black', linewidth=1.5)
    plt.title('Age Distribution', fontsize=18, fontweight='bold')
    plt.xlabel('Age Range', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 10, str(height), ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('age_distribution.pdf')  # Save as PDF
    plt.close()

    # Occupation Distribution Bar Chart
    occupation_labels = [
        "Other", "Academic", "Artist", "Admin", "Student", "Customer Service", "Doctor", 
        "Executive", "Farmer", "Homemaker", "K-12 Student", "Lawyer", "Programmer", 
        "Retired", "Sales", "Scientist", "Self-Employed", "Technician", "Tradesman", 
        "Unemployed", "Writer"
    ]
    
    occupation_values = [occupation_count.get(str(i), 0) for i in range(21)]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(occupation_labels, occupation_values, color=colors['occupation'], edgecolor='black', linewidth=1.5)
    plt.xticks(rotation=90)
    plt.title('Occupation Distribution', fontsize=18, fontweight='bold')
    plt.xlabel('Occupation', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 10, str(height), ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('occupation_distribution.pdf')  # Save as PDF
    plt.close()

# Main function
def main():
    # Load user data
    filename = '/home/martim/Desktop/tese/ml-1m/users.dat'  # Replace with the actual path to your users.dat file
    users = load_data(filename)
    
    # Get the demographic distributions
    gender_count = gender_distribution(users)
    age_count = age_distribution(users)
    occupation_count = occupation_distribution(users)
    
    # Plot and save the demographic distributions to PDF
    plot_demographics(gender_count, age_count, occupation_count)

if __name__ == "__main__":
    main()
