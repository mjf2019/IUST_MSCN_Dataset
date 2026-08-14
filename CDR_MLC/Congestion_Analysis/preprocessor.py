import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def keep_features(df, features_to_keep):
    """
    Drop all columns from the DataFrame except for the specified features.
    
    Parameters:
    - df: pd.DataFrame, the input DataFrame
    - features_to_keep: list, list of column names to retain
    
    Returns:
    - pd.DataFrame with only the specified columns
    """
    # Ensure that the features_to_keep are in the DataFrame
    features_to_keep = [feature for feature in features_to_keep if feature in df.columns]
    
    # Return a DataFrame with only the specified features
    return df[features_to_keep]


def preprocess_data(df, single_row=False):

    if single_row:
        # Ensure that df is a DataFrame with a single row
        if len(df) != 1:
            raise ValueError("DataFrame must contain exactly one row when `single_row=True`.")
        df = df.iloc[0:1]  # Extract the single row DataFrame
    
    # Drop columns that contain only missing values
    df = df.dropna(axis=1, how='all')
    
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Preprocess numeric columns
    if not numeric_cols.empty:
        imputer_numeric = SimpleImputer(strategy='mean')
        df_numeric = pd.DataFrame(imputer_numeric.fit_transform(df[numeric_cols]), columns=numeric_cols)
    else:
        df_numeric = pd.DataFrame()  # Empty DataFrame for numeric data if no numeric columns exist
    
    # Preprocess non-numeric columns
    if not non_numeric_cols.empty:
        imputer_non_numeric = SimpleImputer(strategy='most_frequent')
        df_non_numeric = pd.DataFrame(imputer_non_numeric.fit_transform(df[non_numeric_cols]), columns=non_numeric_cols)
        print(df_non_numeric)
        
        encoder = OneHotEncoder(drop='first', sparse_output=False) 
        encoded = encoder.fit_transform(df_non_numeric)
        
        # Check if the number of columns matches the expected length
        encoded_columns = encoder.get_feature_names_out(non_numeric_cols)
        if encoded.shape[1] != len(encoded_columns):
            raise ValueError(f"Shape mismatch: Encoded shape {encoded.shape[1]} does not match the number of columns {len(encoded_columns)}")
        
        encoded_df = pd.DataFrame(encoded, columns=encoded_columns)
    else:
        encoded_df = pd.DataFrame()  # Empty DataFrame for encoded non-numeric data if no non-numeric columns exist
    
    # Concatenate processed numeric and encoded non-numeric data
    df_preprocessed = pd.concat([df_numeric, encoded_df], axis=1)
    
    features = [
        'State__FA', 'State__A', 'pRetran', 'Max,sMeanPktSz', 'SrcRetra', 'PCRatio', 'State_PA_', 'State_PA_A',
        'SrcWin,SrcLoss', 'DstRate', 'SrcLoad', 'TcpOpt_MwsS  T', 'Load', 'DstLoad', 'TcpRtt', 'Flgs_ e g      ',
        'State_A_PA', 'Flgs_ e d      ', 'Sum', 'AckDat', 'dTtl', 'State__PA', 'Min', 'pLoss', 'DstLoss', 'State_S_',
        'Cause_Status', 'State_FA_', 'Loss', 'StdDev', 'Rate', 'SrcRate', 'IdleTime', 'Dur', 'SrcPkts', 'Flgs_ e s      ',
        'SrcGap', 'DstBytes', 'DstGap', 'sTtl', 'DstWin', 'TotPkts', 'State_S_SA', 'DstPkts', 'Flgs_ e *      ', 'Mean',
        'SrcBytes', 'State__SA', 'TotBytes', 'Cause_Start', 'dMeanPktSz', 'State_A_A', 'DstRetra', 'SynAck'
    ]

    # Drop all columns except the ones in features_to_keep
    df_preprocessed = keep_features(df_preprocessed, features)  

    return df_preprocessed
