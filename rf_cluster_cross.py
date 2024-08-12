from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load data from CSV
file_path = '5_statictical_dataset/not_scale_dataset.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Assuming the CSV file has feature columns and a 'label' column
X = data.drop(columns=['label']).values
true_labels = data['label'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, true_labels, test_size=0.1, random_state=42)
X = X_test
y = y_test
true_labels = y

# Optional: Scale the features if needed
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Cluster data
clustering = AgglomerativeClustering(n_clusters=3)
cluster_labels = clustering.fit_predict(X)

# Add cluster labels to features for Random Forest
X_with_clusters = np.hstack((X, cluster_labels.reshape(-1, 1)))

# Split data for Random Forest
X_train, X_test, y_train, y_test = train_test_split(X_with_clusters, y, test_size=0.3, random_state=0)

# Train Random Forest
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy with Cluster Features: {accuracy:.2f}')

# Cross-validation
cv_scores = cross_val_score(rf, X_with_clusters, y, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Average CV Score: {np.mean(cv_scores):.2f}')

# Compare cluster labels with true labels
ari = adjusted_rand_score(true_labels, cluster_labels)
nmi = normalized_mutual_info_score(true_labels, cluster_labels)
print(f'Adjusted Rand Index (ARI): {ari:.2f}')
print(f'Normalized Mutual Information (NMI): {nmi:.2f}')

# Visualize Clusters vs True Labels
def plot_clusters_vs_true_labels(X, true_labels, cluster_labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    df['True Label'] = true_labels
    df['Cluster Label'] = cluster_labels

    plt.figure(figsize=(14, 7))

    # Plot true labels
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='PCA1', y='PCA2', hue='True Label', palette='tab10', data=df, alpha=0.6)
    plt.title('True Labels')

    # Plot cluster labels
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster Label', palette='tab10', data=df, alpha=0.6)
    plt.title('Cluster Labels')

    plt.show()

plot_clusters_vs_true_labels(X, true_labels, cluster_labels)
