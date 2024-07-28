import pandas as pd
import os
import yaml

class DatasetMerger:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.input_directory = self.config['standard_dataset_path']
        self.input_directory = os.path.join(self.input_directory, self.config['feature_type'])
        self.output_directory = self.config['merged_standard_dataset']
        self.output_directory = os.path.join(self.output_directory, self.config['feature_type'])
        self.label_data = {i: [] for i in range(1, 15)}  # Label numbers 1 to 14
        # Ensure the output directory exists
        os.makedirs(self.output_directory, exist_ok=True)
    
    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def process_files(self):
        # Process each CSV file in the input directory
        for filename in os.listdir(self.input_directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.input_directory, filename)
                df = pd.read_csv(filepath)
                
                # Extract the prefix from the filename (before the first underscore)
                prefix = filename.split('_')[0]
                
                # Add the prefix as a column named 'source'
                df['source'] = prefix
                
                # Append rows to the corresponding label list
                for label in range(1, 15):
                    if label in df['label'].values:
                        # Filter rows with the specific label number
                        label_df = df[df['label'] == label]
                        if not label_df.empty:
                            self.label_data[label].append(label_df)

    def merge_and_save(self):
        # Merge and save DataFrames for each label
        for label, dfs in self.label_data.items():
            if dfs:
                # Concatenate all DataFrames for this label
                combined_df = pd.concat(dfs, ignore_index=True)

                # Remove any existing 'label' column if it exists
                if 'label' in combined_df.columns:
                    combined_df = combined_df.drop(columns=['label'])
                
                # Rename 'source' column to 'label'
                combined_df = combined_df.rename(columns={'source': 'label'})
                
                # Create the output filename
                output_filename = os.path.join(self.output_directory, f'level_{label}.csv')
                
                # Save the combined DataFrame to a new CSV file
                combined_df.to_csv(output_filename, index=False)
                print(f'File for label {label} saved as {output_filename}')

if __name__ == "__main__":
    # Path to your YAML file
    yaml_file_path = 'project_conf.yaml'
    merger = DatasetMerger(yaml_file_path)
    merger.process_files()
    merger.merge_and_save()
