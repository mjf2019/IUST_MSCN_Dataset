import os
import yaml
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer

def read_yaml_features(yaml_file):
    """Reads feature list and directory path from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config['lfm']['input_dataset_path'], config['lfm']['output_dataset_path'], config['lfm']['window_size']

def create_sequences(data, labels, window_size):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        sequence_labels.append(labels[i + window_size - 1])  # Use the last element in the sequence as the label
    return np.array(sequences), np.array(sequence_labels)

if __name__ == '__main__':
    yaml_file = 'project_conf.yaml'
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

    # Remove 'class' column
    if 'class' in merged_df.columns:
        merged_df = merged_df.drop(columns=['class'])

    # Initialize lists to hold sequences and labels
    all_sequences = []
    all_labels = []

    # Get all column names
    all_columns = merged_df.columns.tolist()

    # Define the columns to exclude
    exclude_columns = ['label']

    # Create a list of feature names excluding the specified columns
    feature_columns = [col for col in all_columns if col not in exclude_columns]

    # Extract relevant features and labels
    features = merged_df[feature_columns].values
    labels = merged_df['label'].values

    # Create sequences for LSTM
    sequences, sequence_labels = create_sequences(features, labels, window_size)

    # Convert sequences and labels to numpy arrays
    X = np.array(sequences)
    y = np.array(sequence_labels)
    print(X)
    print(y[0])

    # One-hot encode the labels
    encoder = OneHotEncoder(categories='auto')
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    # Define the model
    model = Sequential()
    model.add(InputLayer(input_shape=(window_size, X.shape[2])))  # Explicitly define the input shape
    model.add(LSTM(400, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Assuming 14 classes

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Fit the model
    model.fit(X, y_one_hot, epochs=100, batch_size=313, validation_split=0.2, verbose=1)

    # Print the shape of the prepared data
    print(f'Sequences shape: {X.shape}')
    print(f'Labels shape: {y_one_hot.shape}')

    # Optionally, save the model
    if not os.path.exists(output_csv_path):
        print("Directory does not exist. Creating now.")
        os.makedirs(output_csv_path)
    
    # Save the model
    model.save(os.path.join(output_csv_path, 'lstm_model.h5'))

    print("LSTM model trained and saved successfully.")
