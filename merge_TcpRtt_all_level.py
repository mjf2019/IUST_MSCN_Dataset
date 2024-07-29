import os
import pandas as pd

def merge_csv_files(directory, feature_cols, label_col, output_file):
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    df_list = []
    for file in all_files:
        df = pd.read_csv(os.path.join(directory, file))
        if set(feature_cols + [label_col]).issubset(df.columns):
            df = df[feature_cols + [label_col]]
            df_list.append(df)
    print(df_list)
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV saved to {output_file}")

# Example usage
directory = 'D:\Dataset\IUST_MSCN_Dataset\standard_dataset\scale_0.001\\all'
feature_cols = ['SynAck','AckDat','TcpRtt']  # Replace with your actual feature column names
label_col = 'label'  # Replace with your actual label column name
output_file = 'merged_FI_data.csv'

merge_csv_files(directory, feature_cols, label_col, output_file)
