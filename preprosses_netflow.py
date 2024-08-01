import os
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import re


class PreprocessNetflow:
    
    def __init__(self, yaml_file_path):
        self.yaml_file_path = yaml_file_path
        self.pre_directory_path = None
        self.ori_directory_path = None
        if self.yaml_file_exists():
            self.load_config()
        else:
            print(f"Error: The YAML file {self.yaml_file_path} does not exist.")
    
    def yaml_file_exists(self):
        """Check if the YAML file exists."""
        return os.path.isfile(self.yaml_file_path)
    
    def load_config(self):
        """Load the YAML configuration file and extract the directory path."""
        try:
            with open(self.yaml_file_path, 'r') as file:
                config = yaml.safe_load(file)
                self.pre_directory_path = config['pre']['output_dataset_path']
                self.ori_directory_path = config['pre']['input_dataset_path']
        except yaml.YAMLError as exc:
            print(f"Error: Failed to parse YAML file. {exc}")
    
    def ensure_directory_exists(self):
        """Create the directory if it does not exist."""
        if self.pre_directory_path:
            if not os.path.isdir(self.pre_directory_path):
                try:
                    os.makedirs(self.pre_directory_path)
                    print(f"Directory created: {self.pre_directory_path}")
                except OSError as e:
                    print(f"Error: Failed to create directory {self.pre_directory_path}. {e}")
        if not os.path.isdir(self.ori_directory_path):
            try:
                os.makedirs(self.ori_directory_path)
                print(f"Directory created: {self.ori_directory_path}")
            except OSError as e:
                print(f"Error: Failed to create directory {self.ori_directory_path}. {e}")

        
    # Function to remove columns with very low diversity
    def remove_low_diversity_columns(self, df, threshold=0.05):
        variances = df.var()
        return df.loc[:, variances >= threshold]

    # Function to preprocess each dataset
    def preprocess_dataset(self, df):
        # Drop columns that contain only missing values
        df = df.dropna(axis=1, how='all')
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        
        # Check if DataFrame has either numeric or non-numeric columns
        if not numeric_cols.empty or not non_numeric_cols.empty:
            # Handle missing values for numeric data
            if not numeric_cols.empty:
                imputer_numeric = SimpleImputer(strategy='mean')
                df_numeric = pd.DataFrame(imputer_numeric.fit_transform(df[numeric_cols]), columns=numeric_cols)
            else:
                df_numeric = pd.DataFrame()  # Empty DataFrame for numeric data if no numeric columns exist
            
            # Handle missing values for non-numeric data
            if not non_numeric_cols.empty:
                imputer_non_numeric = SimpleImputer(strategy='most_frequent')
                df_non_numeric = pd.DataFrame(imputer_non_numeric.fit_transform(df[non_numeric_cols]), columns=non_numeric_cols)
                # Convert non-numeric features to one-hot encoding
                encoder = OneHotEncoder(drop='first')
                encoded = encoder.fit_transform(df_non_numeric)
                encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(non_numeric_cols))
            else:
                encoded_df = pd.DataFrame()  # Empty DataFrame for encoded non-numeric data if no non-numeric columns exist
            
            # Concatenate processed numeric and encoded non-numeric data
            df_preprocessed = pd.concat([df_numeric, encoded_df], axis=1)
            
            # Remove columns with very low diversity
            # Scale features
            #scaler = MinMaxScaler()
            #df_scaled = pd.DataFrame(scaler.fit_transform(df_preprocessed), columns=df_preprocessed.columns)
            
            return df_preprocessed
        else:
            return pd.DataFrame()

    def preprocess(self):

        # Load and combine all datasets
        combined_X = pd.DataFrame()
        combined_y = pd.DataFrame()

        for filename in os.listdir(self.ori_directory_path):
            # Datasets are stored in .flow format
            if filename.endswith(".flow"):
                # Extract number from filename as label value
                label_match = re.search(r'(\d+)', filename)
                if label_match:
                    label_value = int(label_match.group(1))
                    filepath = os.path.join(self.ori_directory_path, filename)
                    try:
                        df = pd.read_csv(filepath)
                        if df.empty:
                            print("The CSV file has headers but no data rows. Skipped it")
                            continue
                        else:
                            print("DataFrame loaded successfully")
                    except pd.errors.EmptyDataError:
                        print("The file is empty. Skipped it.")
                        continue
                    # Separate features (X) and labels (y)
                    y = pd.Series(label_value, index=range(len(df)))
                    combined_X = pd.concat([combined_X, df], ignore_index=True)
                    combined_y = pd.concat([combined_y, y], ignore_index=True)

        # Preprocess the features (X)
        preprocessed_X = self.preprocess_dataset(combined_X)

        # Add the label column to df_features
        preprocessed_X['label'] = combined_y
        # Save preprocessed features (X) and labels (y) to a CSV file
        output_X_filepath = os.path.join(self.pre_directory_path, 'preprocessed_dataset.csv')
        preprocessed_X.to_csv(output_X_filepath, index=False)

        print("Preprocessing complete.")

def main():
    # Path to your YAML file
    yaml_file_path = 'project_conf.yaml'
    # Create an instance
    preprocessor = PreprocessNetflow(yaml_file_path)
    
    # Ensure target directory
    preprocessor.ensure_directory_exists()
    # Preprocess netflow to csv
    preprocessor.preprocess()

if __name__ == '__main__':
    main()