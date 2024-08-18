import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
dbscan_score = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else 'N/A'
print(f'DBSCAN Silhouette Score: {dbscan_score}')

# Evaluate Agglomerative Clustering
agg_score = silhouette_score(X, agg_labels)
print(f'Agglomerative Clustering Silhouette Score: {agg_score:.2f}')

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['DBSCAN Cluster'] = dbscan_labels
df_pca['Agglomerative Cluster'] = agg_labels

# Plot DBSCAN results
fig_dbscan = go.Figure(data=go.Scattergl(
    x=df_pca['PC1'],
    y=df_pca['PC2'],
    mode='markers',
    marker=dict(color=df_pca['DBSCAN Cluster'], colorscale='viridis', size=50),
    text=df_pca['DBSCAN Cluster']
))
fig_dbscan.update_layout(
    title='DBSCAN Clustering',
    xaxis_title='Principal Component 1',
    yaxis_title='Principal Component 2',
    coloraxis_colorbar=dict(title='Cluster Label')
)

# Plot Agglomerative Clustering results
fig_agg = go.Figure(data=go.Scattergl(
    x=df_pca['PC1'],
    y=df_pca['PC2'],
    mode='markers',
    marker=dict(color=df_pca['Agglomerative Cluster'], colorscale='viridis', size=50),
    text=df_pca['Agglomerative Cluster']
))
fig_agg.update_layout(
    title='Agglomerative Clustering',
    xaxis_title='Principal Component 1',
    yaxis_title='Principal Component 2',
    coloraxis_colorbar=dict(title='Cluster Label')
)

# Show the plots
fig_dbscan.show()
fig_agg.show()
