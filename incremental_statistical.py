import pandas as pd
import os
import numpy as np
import re
import yaml

def read_and_merge_csvs(directory_path):
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    df_list = []
    
    # Read and concatenate all CSV files into a single DataFrame
    for f in csv_files:
        # Use regular expression to extract the number
        match = re.search(r'_(\d+)', f)
        if match:
            number = match.group(1)
            print(f"Extracted number: {number}")
        else:
            print("No number found in the filename.")
        df = pd.read_csv(os.path.join(directory_path, f))
        df['label'] = number
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    
    return merged_df

def compute_incremental_statistics(df, feature_list, label_column):
    # Initialize a DataFrame to store the results
    results = pd.DataFrame(index=df.index)
    
    # Iterate over each feature
    for feature in feature_list:
        # Create columns for incremental statistics
        results[f'{feature}_min'] = df[feature].expanding().min()
        results[f'{feature}_max'] = df[feature].expanding().max()
        results[f'{feature}_median'] = df[feature].expanding().median()
        results[f'{feature}_mean'] = df[feature].expanding().mean()
        results[f'{feature}_std'] = df[feature].expanding().std()
    
    # Add the label column
    results[label_column] = df[label_column].values
    
    return results

def main(directory_path, feature_list, label_column):
    # Read and merge all CSV files
    merged_df = read_and_merge_csvs(directory_path)
    
    # Check if all features and label_column are present in the DataFrame
    missing_cols = [col for col in feature_list + [label_column] if col not in merged_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(missing_cols)}")
    
    # Compute incremental statistics
    result_df = compute_incremental_statistics(merged_df, feature_list, label_column)
    result_df = result_df.dropna()
    
    return result_df

def read_yaml_features(yaml_file):
    """Reads feature list and directory path from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['sf']['features'], config['sf']['input_directory'], config['sf']['output_directory'], config['sf']['out_filename'], config['sf']['label_column']


if __name__ == '__main__':
    yaml_file = 'project_conf.yaml'
    # Read features, input directory, and output directory from YAML file
    features, input_directory, output_directory, name_of_out_file, label = read_yaml_features(yaml_file)

    result = main(input_directory, features, label)

    # Create the directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created.")

    file_path = os.path.join(output_directory, name_of_out_file)
    # Save the merged DataFrame to a CSV file
    result.to_csv(file_path, index=False)
    print(f"Merged DataFrame saved to '{file_path}'")

    print(result)
