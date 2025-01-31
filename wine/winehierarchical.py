import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def load_and_preprocess_data(file_path):
    wine = pd.read_csv(file_path)
    wine = wine.drop(columns=["Id", "quality"])  
    scaler = RobustScaler()  
    wine_scaled = scaler.fit_transform(wine)
    return wine, wine_scaled

def reduce_dimensionality(wine_scaled):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(wine_scaled)
    return reduced_data

def hierarchical_clustering(wine_scaled, method='complete'):
    Z = linkage(wine_scaled, method=method)
    return Z

def plot_dendrograms_subplot(wine_scaled, methods):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, method in enumerate(methods):
        Z = hierarchical_clustering(wine_scaled, method)
        dendrogram(Z, leaf_font_size=10, color_threshold=5, leaf_rotation=90, truncate_mode='level', p=5, ax=axes[i])
        axes[i].set_title(f"{method.capitalize()} Linkage")
        axes[i].set_xlabel("Data points", fontsize=9)
        axes[i].set_ylabel("Distance", fontsize=9)

    plt.tight_layout()
    plt.show()

def elbow_and_silhouette_subplot(Z, max_clusters, wine_scaled):
    inertia = []
    silhouette_avg = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(wine_scaled)
        inertia.append(kmeans.inertia_)  # Inertia (Elbow Method)
        
        labels = kmeans.labels_
        silhouette_avg.append(silhouette_score(wine_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(range(2, max_clusters + 1), inertia, marker='o')
    axes[0].set_title("Elbow Method - Inertia")
    axes[0].set_xlabel("Cluster Count")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(range(2, max_clusters + 1), silhouette_avg, marker='o')
    axes[1].set_title("Silhouette Score for Clusters")
    axes[1].set_xlabel("Cluster Count")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.show()

def plot_clusters_subplot(X, Z, n_clusters_list, method):
    fig, axes = plt.subplots(1, len(n_clusters_list), figsize=(18, 6))
    for ax, n_clusters in zip(axes, n_clusters_list):
        n_clusters = int(n_clusters)
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')  
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=labels,
            palette=sns.color_palette("Set2", n_clusters),
            ax=ax,
            s=60,
            alpha=0.7,
            legend=None)
        ax.set_title(f"{method.capitalize()} Linkage\n{n_clusters} Clusters", fontsize=12)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def get_clusters(Z, n_clusters):
    labels = fcluster(Z, n_clusters, criterion='maxclust')  
    return labels

def main():
    file_path = "~/datasets/WineQT.csv" 
    data, wine_scaled = load_and_preprocess_data(file_path)

    reduced_data = reduce_dimensionality(wine_scaled)

    methods = ['ward', 'single', 'complete', 'average']

    plot_dendrograms_subplot(wine_scaled, methods)

    Z = hierarchical_clustering(wine_scaled, method='complete')
    elbow_and_silhouette_subplot(Z, max_clusters=10, wine_scaled=wine_scaled)

    n_clusters_list = [2, 3, 4] 
    plot_clusters_subplot(reduced_data, Z, n_clusters_list, method='complete')

if __name__ == "__main__":
    main()
