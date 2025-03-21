import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set professional style using seaborn for improved visuals
sns.set(style="whitegrid", palette="muted")

# Define color scheme for consistency
colors = {
    'age': '#4C72B0',  # Soft Blue for age bars
    'occupation': '#55A868',  # Green for occupation bars
}

# Main function
def main():
    # Load user data
    data = pd.read_csv("/home/martim/Desktop/tese/datasets/bookcrossing/Users.csv", delimiter=";")

    # Convert 'Age' column to numeric, coercing errors to NaN
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

    # Drop rows where 'Age' is NaN
    data = data.dropna(subset=['Age'])

    # Extract the 'Age' column
    ages = data['Age']

    # Define age ranges
    bins = [0, 18, 24, 34, 44, 49, 55, float('inf')]
    labels = ['Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']

    # Create a new column for age ranges
    age_ranges = pd.cut(ages, bins=bins, labels=labels, right=False)

    # Count the occurrences of each age range
    age_range_counts = age_ranges.value_counts().sort_index()

    # Create the bar plot with seaborn aesthetics
    plt.figure(figsize=(10, 6))
    bars = plt.bar(age_range_counts.index, age_range_counts.values, 
                   color=colors['age'], edgecolor='black', linewidth=1.5)

    # Add count numbers on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 10, 
                 str(height), ha='center', va='bottom', fontsize=12)

    # Set plot aesthetics
    plt.title('Age Distribution', fontsize=18, fontweight='bold')
    plt.xlabel('Age Range', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig('age_distribution_bookCrossing.pdf')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
