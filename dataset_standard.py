import pandas as pd
import os
import yaml

class DataFrameProcessor:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.input_directory = self.config['std']['input_dataset_path']
        self.input_directory = os.path.join(self.input_directory, self.config['std']['feature_type'])
        self.output_directory = self.config['std']['out_dataset_path']
        self.output_directory = os.path.join(self.output_directory, self.config['std']['feature_type'])
        self.label_column = self.config['std']['label_column']
        self.dataframes_by_prefix = {}

        # Ensure the output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def read_and_group_csv_files(self):
        for filename in os.listdir(self.input_directory):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.input_directory, filename)
                df = pd.read_csv(filepath)

                # Extract the prefix from the filename (before the first underscore)
                prefix = filename.split('_')[0]

                # Group DataFrames by their prefix
                if prefix not in self.dataframes_by_prefix:
                    self.dataframes_by_prefix[prefix] = []
                self.dataframes_by_prefix[prefix].append(df)

    def determine_common_columns(self):
        all_common_columns = None
        for prefix, dfs in self.dataframes_by_prefix.items():
            # Find the common columns for the current prefix
            common_columns = set(dfs[0].columns)
            for df in dfs[1:]:
                common_columns.intersection_update(df.columns)

            # Initialize the overall common columns set
            if all_common_columns is None:
                all_common_columns = set(common_columns)
            else:
                all_common_columns.intersection_update(common_columns)

        # Ensure the label column is part of the common columns
        if self.label_column in all_common_columns:
            all_common_columns.remove(self.label_column)
        all_common_columns = list(all_common_columns)
        all_common_columns.append(self.label_column)  # Add label column to the end

        return all_common_columns

    def process_and_save_dataframes(self, common_columns):
        for prefix, dfs in self.dataframes_by_prefix.items():
            # Standardize columns for all DataFrames by keeping only the common columns
            standardized_dataframes = [df[common_columns] for df in dfs]

            # Merge the standardized DataFrames
            merged_df = pd.concat(standardized_dataframes, ignore_index=True)

            # Save the merged DataFrame to a new CSV file
            output_filename = f'{self.output_directory}/{prefix}_standard.csv'
            merged_df.to_csv(output_filename, index=False)
            print(f'Standardized file saved as {output_filename}')

    def run(self):
        self.read_and_group_csv_files()
        common_columns = self.determine_common_columns()
        print(f"Number of standard columns: {len(common_columns)}")
        self.process_and_save_dataframes(common_columns)

if __name__ == "__main__":
    # Path to your YAML file
    yaml_file_path = 'project_conf.yaml'
    processor = DataFrameProcessor(yaml_file_path)
    processor.run()
