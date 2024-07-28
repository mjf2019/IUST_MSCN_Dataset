import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.decomposition import PCA

# Load your dataset
# In windows
data = pd.read_csv('VIDEO\\preprocessed_dataset.csv')


# Assuming the label is in the first column and features are in the remaining columns
labels = data.iloc[:, -1]  # First column for labels
data = data.iloc[:, 1:]  # Remaining columns for features

# Cluster the dataset
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(data)

# Create a DataFrame with the labels and clusters
results = pd.DataFrame({'Label': labels, 'Cluster': clusters})

# Count of each label in each cluster
label_cluster_counts = results.groupby(['Label', 'Cluster']).size().unstack(fill_value=0)
print("Count of each label in each cluster:")
print(label_cluster_counts)

# Primary cluster of each label (the cluster with the highest count for each label)
primary_clusters = label_cluster_counts.idxmax(axis=1)
print("\nPrimary cluster of each label:")
print(primary_clusters)

# Visualization of the primary cluster for each label
plt.figure(figsize=(10, 6))
primary_clusters.plot(kind='bar', color='skyblue')
plt.xlabel('Label')
plt.ylabel('Primary Cluster')
plt.title('Primary Cluster of Each Label')
plt.xticks(rotation=45)
plt.show()



# Count of each label in each cluster
label_cluster_counts = results.groupby(['Label', 'Cluster']).size().unstack(fill_value=0)
print("Count of each label in each cluster:")
print(label_cluster_counts)

# Primary cluster of each label (the cluster with the highest count for each label)
primary_clusters = label_cluster_counts.idxmax(axis=1)
print("\nPrimary cluster of each label:")
print(primary_clusters)

# Percentage calculations
total_samples = len(labels)
primary_cluster_counts = label_cluster_counts.max(axis=1)
primary_cluster_percentages = (primary_cluster_counts / label_cluster_counts.sum(axis=1)) * 100
overall_percentages = (label_cluster_counts.sum(axis=1) / total_samples) * 100

# Print primary cluster with percentages
primary_cluster_info = pd.DataFrame({
    'Primary Cluster': primary_clusters,
    'Count in Primary Cluster': primary_cluster_counts,
    'Percentage in Primary Cluster': primary_cluster_percentages,
    'Overall Percentage': overall_percentages
})

print("\nPrimary cluster information with percentages:")
print(primary_cluster_info)



# Perform PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# Create a DataFrame for plotting
plot_data = pd.DataFrame(data_2d, columns=['PCA1', 'PCA2'])
plot_data['Cluster'] = clusters
plot_data['Label'] = labels

# Visualization using seaborn
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

# Annotate with primary clusters
for label, cluster in primary_clusters.items():
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
