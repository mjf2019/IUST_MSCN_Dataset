import os
import re
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def read_csv_files(directory):
    """Reads all CSV files from a specified directory and returns a list of DataFrames with their filenames."""
    dfs = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            dfs.append(pd.read_csv(file_path))
            filenames.append(filename)
    return dfs, filenames

def read_yaml_features(yaml_file):
    """Reads feature list and directory path from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['fa']['features'], config['fa']['input_directory'], config['fa']['output_directory'], config['fa']['out_filename']

def analyze_features(dfs, filenames, features):
    """Analyzes the distribution and statistical measures of specified features in the DataFrames."""
    analysis_results = {}
    
    for feature in features:
        feature_results = {}
        all_data = []
        
        for i, (df, filename) in enumerate(zip(dfs, filenames)):
            if feature in df.columns:
                data = df[feature].dropna()
                feature_results[filename] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'median': data.median()
                }
                all_data.extend(data.tolist())
            else:
                feature_results[filename] = 'Feature not found'
        
        # Overall analysis
        all_data = np.array(all_data)
        feature_results['Overall'] = {
            'mean': np.mean(all_data),
            'std': np.std(all_data),
            'min': np.min(all_data),
            'max': np.max(all_data),
            'median': np.median(all_data)
        }
        
        analysis_results[feature] = feature_results
    
    return analysis_results

def plot_feature_values(dfs, filenames, features):
    """Plots feature values for each feature across different datasets."""
    for feature in features:
        plt.figure(figsize=(12, 6))
        for df, filename in zip(dfs, filenames):
            if feature in df.columns:
                data = df[feature].dropna()
                plt.plot(data.values, 'o', alpha=0.5, label=filename)
        
        plt.title(f'Feature Values of {feature}')
        plt.xlabel('Index')
        plt.ylabel(feature)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

def calculate_differences(dfs, filenames, features):
    """Calculates how far each feature is from its previous value and counts positive, zero, and negative differences."""
    differences_results = {}
    
    for feature in features:
        feature_results = {}
        
        for df, filename in zip(dfs, filenames):
            if feature in df.columns:
                data = df[feature].dropna()
                differences = data.diff().dropna()  # Calculate differences between consecutive values
                
                total_diffs = len(differences)
                if total_diffs > 0:
                    positive_diffs = (differences > 0).sum()
                    zero_diffs = (differences == 0).sum()
                    negative_diffs = (differences < 0).sum()
                    
                    # Calculate percentages
                    positive_percent = (positive_diffs / total_diffs) * 100
                    zero_percent = (zero_diffs / total_diffs) * 100
                    negative_percent = (negative_diffs / total_diffs) * 100
                    
                    feature_results[filename] = {
                        'positive_diffs': positive_diffs,
                        'positive_percent': positive_percent,
                        'zero_diffs': zero_diffs,
                        'zero_percent': zero_percent,
                        'negative_diffs': negative_diffs,
                        'negative_percent': negative_percent
                    }
                else:
                    feature_results[filename] = 'Not enough data to compute differences'
            else:
                feature_results[filename] = 'Feature not found'
        
        differences_results[feature] = feature_results
    
    return differences_results

def calculate_states(dfs, filenames, features, out_dir, name_of_out_file):
    """Calculates and counts the combined states of features."""
    state_counts = {}
    new_dfs = []
    file_path = os.path.join(out_dir, name_of_out_file)

    # Create the directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Directory '{out_dir}' created.")
    
    for i, (df, filename) in enumerate(zip(dfs, filenames)):
        state_vectors = []

        # Use regular expression to extract the number
        match = re.search(r'_(\d+)', filename)
        if match:
            number = match.group(1)
            print(f"Extracted number: {number}")
        else:
            print("No number found in the filename.")
       
        df['label'] = number
        label = df['label']
        
        for feature in features:
            if feature in df.columns:
                data = df[feature].dropna()
                differences = data.diff().dropna()
                
                # Determine state: 1 for positive, 0 for negative
                states = (differences > 0).astype(int)
                state_vectors.append(states)
            else:
                state_vectors.append(pd.Series([None] * len(df)))
        # Combine state vectors
        combined_states = pd.concat(state_vectors, axis=1)
        combined_states = pd.concat([combined_states, label], axis=1)
        combined_states = combined_states.dropna()  # Drop rows with missing values
        new_dfs.append(combined_states)
        print(combined_states)
        # Convert state vectors to tuples for counting
        state_tuples = [tuple(row) for row in combined_states.to_numpy()]
        state_counts[filename] = Counter(state_tuples)
    
    # Merge the DataFrames
    merged_df = pd.concat(new_dfs, ignore_index=True)

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv(file_path, index=False)
    print(f"Merged DataFrame saved to '{file_path}'")
    
    return state_counts

def plot_state_distribution(state_counts):
    """Plots the distribution of each state across different datasets."""
    states = set()
    for counts in state_counts.values():
        states.update(counts.keys())
    
    states = list(states)  # Convert set to list for consistent ordering
    
    for state in states:
        plt.figure(figsize=(14, 8))
        for dataset, counts in state_counts.items():
            state_count = counts.get(state, 0)
            plt.bar(dataset, state_count, label=f'{state}')
        
        plt.title(f'Distribution of State {state}')
        plt.xlabel('Dataset')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()


# Function to extract the numeric part from the filename
def extract_number(filename):
    match = re.search(r'level_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def main(yaml_file):
    # Read features, input directory, and output directory from YAML file
    features, input_directory, output_directory, name_of_out_file = read_yaml_features(yaml_file)
    
    # Read CSV files
    dfs, filenames = read_csv_files(input_directory)
    
    # Analyze features
    analysis_results = analyze_features(dfs, filenames, features)
    print("Feature Analysis Results:")
    for feature, results in analysis_results.items():
        print(f"\nFeature: {feature}")
        for dataset, metrics in results.items():
            print(f"  {dataset}: {metrics}")
    
    # Plot feature values
    #plot_feature_values(dfs, filenames, features)
    
    # Calculate and print differences
    differences_results = calculate_differences(dfs, filenames, features)
    print("--------------------------------------")
    print("\nFeature Differences Results:")
    for feature, results in differences_results.items():
        # Sort the dictionary keys based on the extracted numeric value
        results = dict(sorted(results.items(), key=lambda item: extract_number(item[0])))
        print(f"\nFeature: {feature}")
        for dataset, metrics in results.items():
            print(f"  {dataset}")
            print(f"  {dataset}: {metrics}")
    
    # Calculate and plot states
    state_counts = calculate_states(dfs, filenames, features, output_directory, name_of_out_file)
    print("\nCombined State Counts:")
    # Sort the dictionary keys based on the extracted numeric value
    state_counts = dict(sorted(state_counts.items(), key=lambda item: extract_number(item[0])))
    # Remove key-value pairs where the values are less than 100
    state_counts = {
        outer_key: {
            inner_key: value
            for inner_key, value in inner_dict.items()
            if value >= 1000
        }
        for outer_key, inner_dict in state_counts.items()
        if any(value >= 100 for value in inner_dict.values())
    }
    print("--------------------------------------")
    for dataset, counts in state_counts.items():
        print(f"\nDataset: {dataset}")
        #for state, count in counts.items():
        #    print(f"  State {state}: {count}")
    
    # Plot state distributions
    #plot_state_distribution(state_counts)
    
# Example usage
if __name__ == "__main__":
    main('project_conf.yaml')
