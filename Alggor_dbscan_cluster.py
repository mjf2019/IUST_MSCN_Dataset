import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scaler_processor import ScalerHandler
from sklearn.model_selection import train_test_split

# Load data from CSV
file_path = '5_statictical_dataset/not_scale_dataset.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Assuming the CSV file has feature columns and no header for labels
# You might need to adjust this based on your data's structure
X = data.drop(columns=['label']).values
true_labels = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, true_labels, test_size=0.1, random_state=42)
X = X_test

#sh = ScalerHandler()
#scaler = sh.load_scaler()
#X = pd.DataFrame(scaler.fit_transform(X))

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_clustering.fit_predict(X)

# Evaluate DBSCAN
# Silhouette score is not well-defined for DBSCAN if it produces noise (-1 labels)
dbscan_score = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else 'N/A'
print(f'DBSCAN Silhouette Score: {dbscan_score}')

# Evaluate Agglomerative Clustering
agg_score = silhouette_score(X, agg_labels)
print(f'Agglomerative Clustering Silhouette Score: {agg_score:.2f}')

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot DBSCAN results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster Label')

# Plot Agglomerative Clustering results
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='viridis', marker='o')
plt.title('Agglomerative Clustering')
plt.colorbar(label='Cluster Label')

plt.show()
