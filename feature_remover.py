import pandas as pd

def remove_features(input_csv, features_to_remove, output_csv):
    """
    Reads a CSV file, removes specified features, and saves the resulting DataFrame to a new CSV file.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - features_to_remove (list): List of feature names (columns) to remove.
    - output_csv (str): Path to save the new CSV file.
    """
    # Load the data from the CSV file
    df = pd.read_csv(input_csv)
    
    # Display the first few rows of the dataframe for verification
    print("Original DataFrame:")
    print(df.head())
    
    # Remove the specified features
    df_cleaned = df.drop(columns=features_to_remove, errors='ignore')
    
    # Display the first few rows of the cleaned dataframe for verification
    print("\nCleaned DataFrame:")
    print(df_cleaned.head())
    
    # Save the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(output_csv, index=False)
    print(f"\nCleaned DataFrame saved to '{output_csv}'")

# Example usage
input_csv_path = 'preprocessed_dataset\scale_0.001\VIDEO\\all\preprocessed_dataset.csv'  # Path to the input CSV file
features_to_remove = ['sTtl', 'SynAck', 'dTtl','DstWin', 'DstGap', 'DstRtt', 'AckDat', 'SrcWin']  # List of features to remove
output_csv_path = 'VIDEO_Feature_Removed.csv'  # Path to save the new CSV file

remove_features(input_csv_path, features_to_remove, output_csv_path)
