import os
import yaml
import pandas as pd

def read_yaml_features(yaml_file):
    """Reads feature list and directory path from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['fs']['level_num'], config['fs']['extract_features'], config['fs']['input_dataset_path'], config['fs']['output_dataset_path']

def select_features(input_csv, features, output_directory, label):
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

    
    # Remove the specified features
    df_select = df[features]
    
    # Rename the 'label' column to 'type'
    df_select.rename(columns={'label': 'class'}, inplace=True)
    df_select['label'] = label
    
    # Save the cleaned DataFrame to a new CSV file
    df_select.to_csv(output_directory, index=False)
    print("dataframe created . . .")

if __name__ == '__main__':

    yaml_file = 'project_conf.yaml'
    # Read features, input directory, and output directory from YAML file
    level_num, features, input_csv_path, output_csv_path = read_yaml_features(yaml_file)
    # Range is num - 1
    level_num += 1

    for i in range(1, level_num):
        input = f"{input_csv_path}level_{i}.csv"
        output = f"{output_csv_path}level_{i}.csv"
        label = i
        select_features(input, features, output, label)
