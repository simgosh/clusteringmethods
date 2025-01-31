import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

# Load the dataset
glass = pd.read_csv("~/datasets/glass.csv")

# Check initial dataset details
print(glass.info())
print(glass.head())
print(glass.describe())
print(glass.isnull().sum())
print(glass.Type.value_counts())

# Preprocessing function
def preprocess_data(glass):
    glass_numeric = glass.drop(columns=["Type"])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(glass_numeric)
    return scaled_data

def kmeans_clustering(scaled_data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    return kmeans

def calculate_validity_indices(scaled_data, kmeans):
    labels = kmeans.labels_
    sil_score = silhouette_score(scaled_data, labels)
    calinski_harabasz = calinski_harabasz_score(scaled_data, labels)
    db_index = davies_bouldin_score(scaled_data, labels)
    inertia = kmeans.inertia_
    return sil_score, calinski_harabasz, db_index, inertia

def visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10):
    num_plots = max_clusters - min_clusters + 1
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Ensure enough rows
    fig, axs = plt.subplots(rows, 3, figsize=(15, 12))
    axs = axs.flatten()  # Flatten to treat as 1D array

    cluster_numbers = range(min_clusters, max_clusters + 1)
    
    for i, n_clusters in enumerate(cluster_numbers):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        centers_2d = pca.transform(kmeans.cluster_centers_)

        # Scatter plot with the clusters
        sns.scatterplot(ax=axs[i], x=pca_data[:, 0], y=pca_data[:, 1], hue=kmeans.labels_,
                        palette="deep", s=60, alpha=0.7)
        axs[i].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=100, label="Cluster Centers")

        axs[i].set_title(f"KMeans Clustering with {n_clusters} Clusters", fontsize=12)
        axs[i].set_xlabel("PCA1", fontsize=10)
        axs[i].set_ylabel("PCA2", fontsize=10)
        axs[i].legend(title="Cluster", fontsize=8)

    for j in range(num_plots, len(axs)):  
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()

def elbow_method(scaled_data, min_clusters=2, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)
        sil_score = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores.append(sil_score)

    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', color='red')
    plt.title("Elbow Method For Optimal k (Silhouette Score)", fontsize=14)
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.xticks(range(min_clusters, max_clusters + 1))
    plt.grid(True)
    plt.show()

def evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10):
    results = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)
        sil_score, calinski, db_index, inertia = calculate_validity_indices(scaled_data, kmeans)    
        results.append({
            'Clusters': n_clusters,
            'Silhouette Score': sil_score,
            'Calinski Harabasz Score': calinski,
            'Davies Bouldin Score': db_index,
            'Inertia': inertia
        })

    results_df = pd.DataFrame(results)
    print("\nFinal Cluster Validity Index Results:")
    print(results_df)
    return results_df


#######################################

# Preprocess the data
scaled_data = preprocess_data(glass)

visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10)

elbow_method(scaled_data, min_clusters=2, max_clusters=10)
results_df = evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10)
