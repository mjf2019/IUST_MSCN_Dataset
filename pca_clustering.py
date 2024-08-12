import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scaler_processor import ScalerHandler
import matplotlib.pyplot as plt
import yaml
import os

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def pca_clustering(config):
    # Load the dataset
    df = pd.read_csv(config['cluster']['cluster_dataset_path'])
    
    # Extract features and labels
    features = df.drop(columns=['label'], errors='ignore')  # Drop 'label' column if present
    labels = df['label'] if 'label' in df.columns else None  # Preserve labels if they exist

    # Standardize features
    sh = ScalerHandler()
    scaler = sh.load_scaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Apply PCA
    pca = PCA(n_components=config['cluster']['n_components'])
    features_pca = pca.fit_transform(features)
    
    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=config['cluster']['cluster_num'], random_state=0)
    clusters = kmeans.fit_predict(features_pca)

    # Add cluster labels to the DataFrame
    df['cluster'] = clusters

    # Define default output file name
    output_file = 'clustered_output.csv'
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)
    
    # Save the DataFrame with cluster labels to a new CSV file
    #df.to_csv(output_file, index=False)
    print(f"Data with cluster labels saved to: {output_file}")

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(features_pca[:, 0], features_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
    plt.title('PCA and K-Means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

# Example usage
if __name__ == "__main__":
    config_file = 'project_conf.yaml'  # Path to your YAML configuration file
    config = load_config(config_file)
    pca_clustering(config)
