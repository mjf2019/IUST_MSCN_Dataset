import os
import yaml
import glob
import pandas as pd
import numpy as np

def read_yaml_features(yaml_file):
    """Reads feature list and directory path from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['lfm']['input_dataset_path'], config['lfm']['output_dataset_path'], config['lfm']['window_size']

# Define a function to create sequences for LSTM
def create_sequences(data, labels, window_size):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        sequence_labels.append(labels[i + window_size - 1])  # Use the last element in the sequence as the label
    return np.array(sequences), np.array(sequence_labels)

if __name__ == '__main__':
    yaml_file = 'project_conf.yaml'
    # Read features, input directory, and output directory from YAML file
    input_csv_path, output_csv_path, window_size = read_yaml_features(yaml_file)

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(os.path.join(input_csv_path, '*.csv'))

    # Initialize a list to hold DataFrames
    dfs = []

    # Loop through the list of files and read each one into a DataFrame
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    print(f'Number of rows in merged DataFrame: {len(merged_df)}')

    # Initialize lists to hold sequences and labels
    all_sequences = []
    all_labels = []
    all_classes = []

    # Group by the 'class' column
    grouped_by_class = merged_df.groupby('class')

    # Get all column names
    all_columns = merged_df.columns.tolist()

    # Define the columns to exclude
    exclude_columns = ['class', 'label']

    # Create a list of feature names excluding the specified columns
    feature_columns = [col for col in all_columns if col not in exclude_columns]

    # Iterate over each class group
    for class_label, class_group in grouped_by_class:
        # Group by 'label' within each class group
        grouped_by_label = class_group.groupby('label')
        
        # Iterate over each label group within the class group
        for label_value, label_group in grouped_by_label:
            # Ensure data is sorted by time or index
            label_group = label_group.sort_index()
            
            # Extract relevant features and labels
            features = label_group[feature_columns].values
            labels = label_group['label'].values
            
            # Create sequences for LSTM
            sequences, sequence_labels = create_sequences(features, labels, window_size)
            
            # Append to lists
            all_sequences.extend(sequences)
            all_labels.extend(sequence_labels)
            all_classes.extend([class_label] * len(sequences))

    # Convert sequences to list of strings (if necessary, or use another format)
    all_sequences = [seq.tolist() for seq in all_sequences]

    # Create a DataFrame
    df_lstm = pd.DataFrame({
        'class': all_classes,
        'lstm_sequence': all_sequences,
        'label': all_labels
    })

    # Check if the directory exists
    if not os.path.exists(output_csv_path):
        print("Directory does not exist. Creating now.")
        os.makedirs(output_csv_path)

    # Save the DataFrame to a CSV file
    df_lstm.to_csv(os.path.join(output_csv_path, 'lstm_sequences.csv'), index=False)

    print("LSTM DataFrame saved to CSV file successfully.")
