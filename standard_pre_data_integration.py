import pandas as pd
import os
import yaml

class DatasetIntegrator:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.preprocessed_dataset_path = self.config['preprocessed_dataset_path']
        self.output_dataset_path = self.config['all_classes_dataset_path']
        self.feature_type = self.config['feature_type']
        self.class_types = self.config['class_types']

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def process_datasets(self):
        for class_type in self.class_types:
            input_directory = os.path.join(self.preprocessed_dataset_path, class_type, self.feature_type)
            output_directory = os.path.join(self.output_dataset_path, self.feature_type)
            
            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)

            if os.path.exists(input_directory):
                for filename in os.listdir(input_directory):
                    if filename.endswith(".csv"):
                        filepath = os.path.join(input_directory, filename)
                        df = pd.read_csv(filepath)
                        
                        # Modify the filename
                        new_filename = f"{class_type}_{filename}"
                        output_filepath = os.path.join(output_directory, new_filename)
                        
                        # Save the modified DataFrame
                        df.to_csv(output_filepath, index=False)
                        print(f'Saved: {output_filepath}')
            else:
                print(f'Directory does not exist: {input_directory}')

if __name__ == "__main__":
    # Path to your YAML file
    yaml_file_path = 'project_conf.yaml'
    processor = DatasetIntegrator(yaml_file_path)
    processor.process_datasets()
