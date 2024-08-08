import os
import yaml
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

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

def train_lstm_model(X_train, y_train, window_size, num_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=(window_size, X_train.shape[2])))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Number of classes
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, verbose=1)
    
    return model, history

def evaluate_model(model, X_test, y_test):
    # Predict the test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    
    return accuracy

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

    # Initialize a list to hold DataFrames for each class
    class_dfs = {}

    # Split merged_df into separate DataFrames based on the 'class' column
    for class_label, class_group in merged_df.groupby('class'):
        class_dfs[class_label] = class_group
    
    # Iterate over each class DataFrame
    for class_label, class_df in class_dfs.items():
        print(f'\nProcessing class: {class_label}')
        
        # Get all column names
        all_columns = class_df.columns.tolist()

        # Define the columns to exclude
        exclude_columns = ['class', 'label']

        # Create a list of feature names excluding the specified columns
        feature_columns = [col for col in all_columns if col not in exclude_columns]

        # Ensure data is sorted by time or index
        class_df = class_df.sort_index()

        # Extract relevant features and labels
        features = class_df[feature_columns].values
        labels = class_df['label'].values

        # Create sequences for LSTM
        sequences, sequence_labels = create_sequences(features, labels, window_size)

        # Convert sequences and labels to numpy arrays
        X = np.array(sequences)
        y = np.array(sequence_labels)

        # One-hot encode the labels
        encoder = OneHotEncoder(categories='auto')
        y_one_hot = encoder.fit_transform(y.reshape(-1, 1)).toarray()

        # Define the number of classes for this specific class model
        num_classes = y_one_hot.shape[1]

        # Split data into training and testing sets
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_one_hot[:split_idx], y_one_hot[split_idx:]

        # Train the LSTM model
        model, history = train_lstm_model(X_train, y_train, window_size, num_classes)

        # Evaluate the model
        accuracy = evaluate_model(model, X_test, y_test)
        print(f'Accuracy for class {class_label}: {accuracy:.4f}')

        # Save the model
        if not os.path.exists(output_csv_path):
            os.makedirs(output_csv_path)
        
        model.save(os.path.join(output_csv_path, f'lstm_model_class_{class_label}.h5'))

    print("LSTM models trained and saved successfully.")
