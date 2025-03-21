def sort_dat_file(input_file, output_file):
    """
    Sorts a .dat file based on the first attribute (user_id).

    Args:
        input_file (str): Path to the input .dat file.
        output_file (str): Path to the output .dat file.
    """
    try:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        # Parse lines into lists, convert user_id to int for sorting
        data = []
        for line in lines:
            parts = line.strip().split('::')
            data.append((int(parts[0]), parts))  # Store user_id as int

        # Sort the data based on user_id (the first element of each tuple)
        data.sort(key=lambda x: x[0])

        # Write the sorted data to the output file
        with open(output_file, 'w') as outfile:
            for user_id, parts in data:
                outfile.write('::'.join(parts) + '\n')

        print(f"Successfully sorted '{input_file}' and saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
input_file = '/home/martim/Desktop/tese/datasets/ml-100k/normalized_rating_movielens_100k.dat'  # Replace with your input file name
output_file = '/home/martim/Desktop/tese/datasets/ml-100k/sorted_normalized_rating_movielns_100k.dat' # Replace with your output file name.
sort_dat_file(input_file, output_file)