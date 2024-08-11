import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def read_and_merge_csv(directory_path):
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(directory_path, f)) for f in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def get_feature_columns(df, target, class_column):
    return [col for col in df.columns if col not in [target, class_column]]

def fit_regression_model(group, feature_columns, target):
    X = group[feature_columns]
    y = group[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {
        'model': model,
        'mse': mse
    }

# Path to your directory containing CSV files
directory_path = '6_kpi_not_scale_FE_dataset/scale_0.001/'  # Replace with your actual directory path

# Read and merge all CSV files
df = read_and_merge_csv(directory_path)

# Define target and class columns
target = 'label'  # Replace with your actual target column
class_column = 'class'  # Replace with your actual class column

# Get feature columns and target column
feature_columns = get_feature_columns(df, target, class_column)

# Group by class and apply regression model fitting
grouped = df.groupby(class_column)
regression_results = grouped.apply(lambda x: fit_regression_model(x, feature_columns, target))

# View results
for group_name, result in regression_results.items():
    print(f"Class: {group_name}")
    print(f"Mean Squared Error: {result['mse']:.2f}")
    # Access the model if needed: result['model']
