import pandas as pd
import os
import yaml

def update_labels(csv_file, level_1, level_2, level_3, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Ensure 'label' column exists in the DataFrame
    if 'label' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'label' column.")
    
    # Create sets for fast lookups
    set_level_1 = set(level_1)
    set_level_2 = set(level_2)
    set_level_3 = set(level_3)

    # Define a function to update the label
    def update_label(row):
        if row['label'] in set_level_1:
            return 1
        elif row['label'] in set_level_2:
            return 2
        elif row['label'] in set_level_3:
            return 3
        else:
            return row['label']  # or handle it differently if needed

    # Apply the function to the 'label' column
    df['label'] = df.apply(update_label, axis=1)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    
    print(f"Updated CSV saved to: {output_file}")

def main():
    # Load configuration
    with open('project_conf.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract configuration values
    input_file = config['kpi']['input_dataset_path']
    output_file = config['kpi']['output_dataset_path']
    level_1 = config['kpi']['level_1']
    level_2 = config['kpi']['level_2']
    level_3 = config['kpi']['level_3']
    
    # Call the update_labels function with config values
    update_labels(input_file, level_1, level_2, level_3, output_file)

if __name__ == "__main__":
    main()
