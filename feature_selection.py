import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier  # Use RandomForestRegressor for regression tasks
from sklearn.linear_model import LogisticRegression  # Use an appropriate model for your task
from sklearn.model_selection import train_test_split

def read_and_merge_csv(directory_path):
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(directory_path, f)) for f in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def feature_selection(df, target_column, class_column=None):
    # Define features and target
    if class_column:
        features = df.drop([target_column, class_column], axis=1, errors='ignore')
    else:
        features = df.drop(target_column, axis=1, errors='ignore')
    target_series = df[target_column]
    
    # Handle classification or regression tasks
    score_func = f_classif if target_series.dtype == 'object' else f_regression
    model = LogisticRegression() if target_series.dtype == 'object' else RandomForestClassifier()
    
    # 1. Univariate Feature Selection
    selector = SelectKBest(score_func=score_func, k='all')
    selector.fit(features, target_series)
    univariate_scores = pd.DataFrame({'Feature': features.columns, 'Score': selector.scores_})
    
    # 2. Recursive Feature Elimination (RFE)
    rfe = RFE(estimator=model, n_features_to_select=5)
    rfe.fit(features, target_series)
    rfe_features = features.columns[rfe.support_]
    
    # 3. Feature Importance from Random Forest
    rf_model = RandomForestClassifier()  # Use RandomForestRegressor for regression tasks
    rf_model.fit(features, target_series)
    feature_importances = pd.DataFrame({'Feature': features.columns, 'Importance': rf_model.feature_importances_})
    
    # Combine Results
    univariate_scores.set_index('Feature', inplace=True)
    feature_importances.set_index('Feature', inplace=True)
    univariate_scores['Normalized_Score'] = (univariate_scores['Score'] - univariate_scores['Score'].min()) / (univariate_scores['Score'].max() - univariate_scores['Score'].min())
    feature_importances['Normalized_Importance'] = (feature_importances['Importance'] - feature_importances['Importance'].min()) / (feature_importances['Importance'].max() - feature_importances['Importance'].min())
    
    combined_scores = univariate_scores.join(feature_importances[['Normalized_Importance']], how='outer').fillna(0)
    combined_scores['Total_Score'] = combined_scores['Normalized_Score'] + combined_scores['Normalized_Importance']
    combined_scores = combined_scores.sort_values(by='Total_Score', ascending=False)
    
    return combined_scores, rfe_features

# Path to your directory containing CSV files
#directory_path = '6_kpi_not_scale_FE_dataset/scale_0.001/'  # Replace with your actual directory path
directory_path = '3_kpi_not_scale_std_dataset/scale_0.001/all'
# Read and merge all CSV files
df = read_and_merge_csv(directory_path)

# Define target and class columns
target_column = 'label'  # Replace with your actual target column
class_column = 'class'  # Replace with your actual class column if applicable

# Perform feature selection
combined_scores, rfe_features = feature_selection(df, target_column, class_column)

# Output results
print("Combined Feature Selection Scores:")
print(combined_scores)

print(f"\nSelected Features by RFE: {rfe_features.tolist()}")
