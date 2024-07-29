import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import yaml

class ClusteringAnalysis:
    def __init__(self, config_path, random_state=0):
        self.config_path = config_path
        self.random_state = random_state
        self.data = None
        self.labels = None
        self.clusters = None
        self.results = None
        self.primary_clusters = None
        self.primary_cluster_info = None
        self.pca_data = None
        self.file_path = self.load_config()
        self.n_clusters = None  # Placeholder for the optimal number of clusters

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config['cluster_dataset_path']

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.labels = self.data.iloc[:, -1]  # Assuming the label is in the last column
        self.data = self.data.iloc[:, :-1]  # Exclude the label from the features
    
    def determine_optimal_clusters(self, max_clusters=10):
        """ Determine the optimal number of clusters using the Elbow Method. """
        inertia = []
        silhouette_scores = []
        
        for n in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=self.random_state)
            kmeans.fit(self.pca_data)  # Use PCA-transformed data
            inertia.append(kmeans.inertia_)
            if n > 1:
                silhouette_avg = silhouette_score(self.pca_data, kmeans.labels_)
                silhouette_scores.append(silhouette_avg)
        
        # Plot the Elbow Method
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), inertia, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')

        # Plot Silhouette Scores
        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores')
        
        plt.tight_layout()
        plt.show()
        
        # Choose the optimal number of clusters (can be improved by more sophisticated methods)
        self.n_clusters = int(input("Enter the optimal number of clusters based on the plots: "))

    def cluster_data(self):
        # Perform PCA before clustering
        pca = PCA(n_components=2)
        self.pca_data = pca.fit_transform(self.data)
        
        if self.n_clusters is None:
            self.determine_optimal_clusters()
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = kmeans.fit_predict(self.pca_data)
        self.results = pd.DataFrame({'Label': self.labels, 'Cluster': self.clusters})
    
    def analyze_clusters(self):
        label_cluster_counts = self.results.groupby(['Label', 'Cluster']).size().unstack(fill_value=0)
        print("Count of each label in each cluster:")
        print(label_cluster_counts)
        
        self.primary_clusters = label_cluster_counts.idxmax(axis=1)
        print("\nPrimary cluster of each label:")
        print(self.primary_clusters)
        
        total_samples = len(self.labels)
        primary_cluster_counts = label_cluster_counts.max(axis=1)
        primary_cluster_percentages = (primary_cluster_counts / label_cluster_counts.sum(axis=1)) * 100
        overall_percentages = (label_cluster_counts.sum(axis=1) / total_samples) * 100
        
        self.primary_cluster_info = pd.DataFrame({
            'Primary Cluster': self.primary_clusters,
            'Count in Primary Cluster': primary_cluster_counts,
            'Percentage in Primary Cluster': primary_cluster_percentages,
            'Overall Percentage': overall_percentages
        })
        
        print("\nPrimary cluster information with percentages:")
        print(self.primary_cluster_info)
    
    def visualize_clusters(self):
        plt.figure(figsize=(10, 6))
        self.primary_clusters.plot(kind='bar', color='skyblue')
        plt.xlabel('Label')
        plt.ylabel('Primary Cluster')
        plt.title('Primary Cluster of Each Label')
        plt.xticks(rotation=45)
        plt.show()
    
    def perform_pca(self):
        pca = PCA(n_components=2)
        self.pca_data = pca.fit_transform(self.data)
        plot_data = pd.DataFrame(self.pca_data, columns=['PCA1', 'PCA2'])
        plot_data['Cluster'] = self.clusters
        plot_data['Label'] = self.labels
        
        plt.figure(figsize=(14, 10))
        sns.scatterplot(
            x='PCA1', y='PCA2',
            hue='Cluster',
            palette=sns.color_palette('tab10', n_colors=14),
            data=plot_data,
            legend='full',
            alpha=0.6,
            edgecolor=None
        )
        
        for label, cluster in self.primary_clusters.items():
            label_data = plot_data[plot_data['Label'] == label]
            centroid = label_data[label_data['Cluster'] == cluster][['PCA1', 'PCA2']].mean()
            plt.text(
                centroid['PCA1'], centroid['PCA2'],
                f'{label} ({cluster})',
                horizontalalignment='center',
                verticalalignment='center',
                size=12, weight='bold',
                color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
            )
        
        plt.title('PCA Visualization of Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.show()

    def list_feature_importance(self):
        pca = PCA(n_components=2)
        pca.fit(self.data)
        feature_importance = pd.DataFrame({'Feature': self.data.columns, 'Importance': np.abs(pca.components_[0])})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        print("\nFeature importance based on PCA component 1:")
        print(feature_importance)

    def run_analysis(self):
        self.load_data()
        self.cluster_data()
        self.analyze_clusters()
        self.visualize_clusters()
        self.perform_pca()
        self.list_feature_importance()

if __name__ == "__main__":
    # Path to your YAML file
    yaml_file_path = 'project_conf.yaml'
    analysis = ClusteringAnalysis(yaml_file_path)
    analysis.run_analysis()
