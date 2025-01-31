import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def load_and_preprocess_data(file_path):
    glass = pd.read_csv(file_path)
    glass_numeric = glass.drop(columns=["Type"])  
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(glass_numeric)  
    return glass_numeric, scaled_data

def reduce_dimensionality(scaled_data):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    return reduced_data

def hierarchical_clustering(scaled_data, method='ward'):
    return linkage(scaled_data, method=method)

def plot_dendrograms(methods, scaled_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12)) 
    axes = axes.ravel()  
    
    for i, method in enumerate(methods):
        Z = hierarchical_clustering(scaled_data, method)
        dendrogram(Z, ax=axes[i], leaf_font_size=10, color_threshold=5, leaf_rotation=90, truncate_mode='level', p=3)
        axes[i].set_title(f"Dendrogram for {method.capitalize()} Linkage")
        axes[i].set_xlabel("Data points")
        axes[i].set_ylabel("Distance")

    plt.tight_layout()
    plt.show()

def get_clusters(Z, threshold):
    labels = fcluster(Z, threshold, criterion='maxclust')  
    return labels

def plot_clusters(X, labels, ax):
    palette = sns.color_palette("Set2", n_colors=len(set(labels)))  
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette=palette, ax=ax, s=60, alpha=0.7, legend="full")
    ax.set_title('Hierarchical Clustering', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True)

def elbow_method(Z, max_clusters=10, scaled_data=None):
    inertia = []  
    for n_clusters in range(2, max_clusters+1):
        labels = fcluster(Z, n_clusters, criterion='maxclust')  
        if scaled_data is not None:
            inertia.append(silhouette_score(scaled_data, labels))  

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters+1), inertia, marker='o')
    plt.title("Elbow Method - Silhouette Score")
    plt.xlabel("Cluster Count")
    plt.ylabel("Silhouette Score")
    plt.show()

def calculate_silhouette(Z, max_clusters=10, scaled_data=None):
    silhouette_avg = []
    for n_clusters in range(2, max_clusters+1):
        labels = fcluster(Z, n_clusters, criterion='maxclust')  
        if scaled_data is not None:
            silhouette_avg.append(silhouette_score(scaled_data, labels))

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters+1), silhouette_avg, marker='o')
    plt.title("Silhouette Score for Different Cluster Counts")
    plt.xlabel("Cluster Count")
    plt.ylabel("Silhouette Score")
    plt.show()

def main():
    file_path = "~/datasets/glass.csv"  
    data, scaled_data = load_and_preprocess_data(file_path)
    
    reduced_data = reduce_dimensionality(scaled_data)
    
    methods = ['ward', 'single', 'complete', 'average']
    
    plot_dendrograms(methods, scaled_data)

    Z = hierarchical_clustering(scaled_data, method='ward')  
    elbow_method(Z, scaled_data=scaled_data)

    calculate_silhouette(Z, scaled_data=scaled_data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))  
    axes = axes.ravel() 

    n_clusters_list = [2,3, 5, 10]

    for i, n_clusters in enumerate(n_clusters_list):
        labels = get_clusters(Z, n_clusters)
        plot_clusters(reduced_data, labels, axes[i])
        axes[i].set_title(f'{n_clusters} Clusters', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
