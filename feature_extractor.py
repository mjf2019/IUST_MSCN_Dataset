import os
import yaml
import pandas as pd

def read_yaml_features(yaml_file):
    """Reads feature list and directory path from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['fe']['level_num'], config['fe']['features'], config['fe']['input_dataset_path'], config['fe']['output_dataset_path']

def extract_features(input_csv, features, output_directory):
    """
    Reads a CSV file, removes specified features, and saves the resulting DataFrame to a new CSV file.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - features_to_remove (list): List of feature names (columns) to remove.
    - output_csv (str): Path to save the new CSV file.
    """
    # Extract the directory path and file name
    directory = os.path.dirname(output_directory)
    # Check if the directory exists
    if not os.path.exists(directory):
        print("Directory does not exist. Creating now.")
        os.makedirs(directory)
    # Load the data from the CSV file
    df = pd.read_csv(input_csv)
    for fe in features:
        # Initialize new columns with default values of 0
        df[f'{fe}_Greater'] = 0
        df[f'{fe}_Equal'] = 0
        df[f'{fe}_Smaller'] = 0

        # Compute the previous values of 'SynAck'
        previous_value = df[fe].shift(1)

        # Define conditions for each new feature
        df[f'{fe}_Greater'] = (df[fe] > previous_value).astype(int)
        df[f'{fe}_Equal'] = (df[fe] == previous_value).astype(int)
        df[f'{fe}_Smaller'] = (df[fe] < previous_value).astype(int)

        # Handle NaN values in the new columns (if needed)
        df.fillna(0, inplace=True)
        # Define the list of columns to move to the rightmost position
        # Remove the 'SynAck' column
        df = df.drop(columns=[fe])

    columns_to_move = ['class', 'label']

    # Get the remaining columns (excluding the ones to move)
    remaining_columns = [col for col in df.columns if col not in columns_to_move]

    # Reorder columns to have 'class' and 'label' at the end
    new_order = remaining_columns + columns_to_move

    # Reorder the DataFrame columns
    df = df[new_order]

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv(output_directory, index=False)
    print("\nnew Dataset created")

if __name__ == '__main__':

    yaml_file = 'project_conf.yaml'
    # Read features, input directory, and output directory from YAML file
    level_num, features, input_csv_path, output_csv_path = read_yaml_features(yaml_file)
    # Range is num - 1
    level_num += 1

    for i in range(1, level_num):
        input = f"{input_csv_path}level_{i}.csv"
        output = f"{output_csv_path}level_{i}.csv"
        extract_features(input, features, output)
