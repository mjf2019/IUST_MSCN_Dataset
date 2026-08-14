import pandas as pd
import joblib
import os
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class ScalerHandler:
    def __init__(self, config_file='project_conf.yaml'):
        # Load configuration
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
        self.scaler_type = self.config['scaler']['scaler_type']
        # For scaler parameter saving
        self.scaler_directory = self.config['scaler']['output_directory']
        self.orig_input_csv_directory = self.config['scaler']['orig_input_csv_directory']
        self.orig_input_csv_name = self.config['scaler']['orig_input_csv_name']
        self.sta_input_csv_directory = self.config['scaler']['sta_input_csv_directory']
        self.sta_input_csv_name = self.config['scaler']['sta_input_csv_name']
        self.sta_mode = self.config['scaler']['scaler_sta_mode']
        self.orig_mode = self.config['scaler']['scaler_orig_mode']
        
        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure that the required directories exist"""
        os.makedirs(self.scaler_directory, exist_ok=True)

    def get_scaler(self):
        """Return the scaler object based on the configuration"""
        if self.scaler_type == 'zscore':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")

    def save_sta_scaler(self):
        """Fit and save the scaler based on the configuration"""
        # Load the data
        data = pd.read_csv(os.path.join(self.sta_input_csv_directory, self.sta_input_csv_name))
        # Check if 'label' column exists and drop it
        data = data[data['label'] == 1]
        if 'label' in data.columns:
            data = data.drop(columns=['label'])
        # Initialize and fit the scaler
        scaler = self.get_scaler()
        scaler.fit(data)
        
        # Save the scaler with type postfix in the filename
        scaler_file_name = f'{self.sta_mode}_{self.scaler_type}.joblib'
        scaler_file_path = os.path.join(self.scaler_directory, scaler_file_name)
        joblib.dump(scaler, scaler_file_path)
                
        print(f'Scaler saved to {scaler_file_path}')
        
    def save_orig_scaler(self):
        """Fit and save the scaler based on the configuration"""
        # Load the data
        data = pd.read_csv(os.path.join(self.orig_input_csv_directory, self.orig_input_csv_name))
        # Check if 'label' column exists and drop it
        if 'label' in data.columns:
            data = data.drop(columns=['label'])
        # Initialize and fit the scaler
        scaler = self.get_scaler()
        scaler.fit(data)
        
        # Save the scaler with type postfix in the filename
        scaler_file_name = f'{self.orig_mode}_{self.scaler_type}.joblib'
        scaler_file_path = os.path.join(self.scaler_directory, scaler_file_name)
        joblib.dump(scaler, scaler_file_path)
                
        print(f'Scaler saved to {scaler_file_path}')

    def load_orig_scaler(self):
        """Load the scaler and apply it to new data"""
        # Load the scaler with type postfix in the filename
        scaler_file_name = f'{self.orig_mode}_{self.scaler_type}.joblib'
        scaler_file_path = os.path.join(self.scaler_directory, scaler_file_name)
        
        if not os.path.exists(scaler_file_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_file_path}")
        
        scaler = joblib.load(scaler_file_path)
        
        return scaler
    
    def load_statistical_scaler(self):
        """Load the scaler and apply it to new data"""
        # Load the scaler with type postfix in the filename
        scaler_file_name = f'{self.sta_mode}_{self.scaler_type}.joblib'
        scaler_file_path = os.path.join(self.scaler_directory, scaler_file_name)
        
        if not os.path.exists(scaler_file_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_file_path}")
        
        scaler = joblib.load(scaler_file_path)
        
        return scaler
if __name__ == '__main__':

    handler = ScalerHandler()

    handler.save_orig_scaler()
    handler.save_sta_scaler()