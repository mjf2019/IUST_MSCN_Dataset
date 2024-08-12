import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scaler_processor import ScalerHandler
import yaml

class ModelTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()
        self.rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_importances = None

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.mode = config['mode']
        self.train_csv = config['train_csv']
        self.test_csv = config['test_csv']
        self.random_state = config['random_state']
        self.dir_mode = config['dir_mode']
        
        if self.mode == 'single_file':
            self.use_separate_files = False
        elif self.mode == 'separate_files':
            self.use_separate_files = True
        else:
            raise ValueError("Invalid mode specified in the config file. Use 'single_file' or 'separate_files'.")

    def load_data(self):
        # Load training data
        if self.dir_mode:
            directory, filename = os.path.split(self.train_csv)
            # Get a list of all CSV files in the directory
            csv_files = glob.glob(os.path.join(directory, '*.csv'))
            print(csv_files)

            # Initialize a list to hold DataFrames
            dfs = []

            # Loop through the list of files and read each one into a DataFrame
            for file in csv_files:
                df = pd.read_csv(file)
                dfs.append(df)

            # Concatenate all DataFrames into a single DataFrame
            merged_df = pd.concat(dfs, ignore_index=True)
            df_train = merged_df
            self.X_train = df_train.drop(columns=['label','class'])

        else:
            df_train = pd.read_csv(self.train_csv)
            self.X_train = df_train.drop(columns=['label'])

        sh = ScalerHandler()
        scaler = sh.load_scaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.y_train = df_train['label']

        
        if self.use_separate_files and self.test_csv:
            # Load test data if separate files are used
            df_test = pd.read_csv(self.test_csv)
            self.X_test = df_test.drop(columns=['label'])
            sh = ScalerHandler()
            scaler = sh.load_scaler()
            self.X_test = pd.DataFrame(scaler.fit_transform(self.X_test), columns=self.X_test.columns)
            self.y_test = df_test['label']
        else:
            # Split the data into train and test sets (for evaluation)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=0.3, random_state=self.random_state)

    def train_and_evaluate(self):
        # Train the Random Forest classifier
        self.rf.fit(self.X_train, self.y_train)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.rf, self.X_test, self.y_test, cv=5, scoring='accuracy')
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Make predictions
        y_pred = self.rf.predict(self.X_test)
        
        # Evaluate the classifier
        print("\nClassification Report for Test Data:")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        
        # Save predictions to a new CSV file
        predictions_df = pd.DataFrame({
            'True_Label': self.y_test,
            'Predicted_Label': y_pred
        })
        
        # Compute and plot confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print('Confusion Matrix:\n', cm)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Get and print feature importances
        self.feature_importances = self.rf.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.feature_importances
        })
        importance_df = importance_df[importance_df['importance'] > 0].sort_values(by='importance', ascending=False).reset_index(drop=True)
        print("\nFeature Importance:")
        print(importance_df)

if __name__ == "__main__":
    # Path to your YAML file
    yaml_file_path = 'project_conf.yaml'
    trainer = ModelTrainer(yaml_file_path)
    trainer.load_data()
    trainer.train_and_evaluate()
