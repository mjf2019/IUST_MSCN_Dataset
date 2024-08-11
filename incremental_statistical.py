import pandas as pd
import os
import numpy as np
import re

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


if __name__ == '__main__':
    # Example usage
    directory_path = '4_kpi_not_scale_mr_std_dataset/scale_0.001/all/'
    feature_list = ['SynAck', 'TcpRtt', 'AckDat']  # Replace with your feature column names
    label_column = 'label'  # Replace with your label column name
    out_dir = '5_statictical_dataset/'
    file_name = 'not_scale_dataset.csv'

    result = main(directory_path, feature_list, label_column)

    # Create the directory if it does not exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Directory '{out_dir}' created.")

    file_path = os.path.join(out_dir, file_name)
    # Save the merged DataFrame to a CSV file
    result.to_csv(file_path, index=False)
    print(f"Merged DataFrame saved to '{file_path}'")

    print(result)
