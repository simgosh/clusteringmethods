import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def load_and_preprocess_data(file_path):
    breast = pd.read_csv(file_path)
    breast.drop(columns=["id", "Unnamed: 32", "diagnosis"], inplace=True)
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(breast)
    return scaled_data, breast 

def reduce_dimensionality(scaled_data):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    return reduced_data

def hierarchical_clustering(scaled_data, method='ward'):
    Z = linkage(scaled_data, method=method, metric="euclidean")
    return Z

def plot_dendrograms_2x2(scaled_data, methods):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  
    axes = axes.flatten()  
    for i, method in enumerate(methods):
        Z = hierarchical_clustering(scaled_data, method)
        color_threshold = np.max(Z[:, 2]) * 0.7  
        dendrogram(
            Z, leaf_font_size=10, color_threshold=color_threshold, 
            leaf_rotation=90, truncate_mode='level', p=4, ax=axes[i]
        )
        axes[i].set_title(f"Dendrogram for {method.capitalize()} Linkage")
        axes[i].set_xlabel("Data points")
        axes[i].set_ylabel("Distance")
    plt.tight_layout()
    plt.show()

def get_clusters(Z, n_clusters):
    labels = fcluster(Z, n_clusters, criterion='maxclust')  
    return labels

def plot_clusters(X, labels, ax):
    palette = sns.color_palette("Set2", n_colors=len(set(labels)))  
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette=palette, ax=ax, s=60, alpha=0.7, legend="full")
    ax.set_title('Hierarchical Clustering', fontsize=10)
    ax.set_xlabel('Feature 1', fontsize=8)
    ax.set_ylabel('Feature 2', fontsize=8)
    ax.grid(True)

def calculate_silhouette(Z, scaled_data=None, max_clusters=10):
    silhouette_avg = []
    for n_clusters in range(2, max_clusters+1):
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        silhouette_avg.append(silhouette_score(scaled_data, labels))  
    
    # Sonuçları çizelim
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters+1), silhouette_avg, marker='o')
    plt.title("Silhouette Score for Different Cluster Counts")
    plt.xlabel("Cluster Count")
    plt.ylabel("Silhouette Score")
    plt.show()

def plot_clusters_for_different_n_clusters(Z, reduced_data, n_clusters_list):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  
    axes = axes.flatten() 
    
    for i, n_clusters in enumerate(n_clusters_list):
        labels = get_clusters(Z, n_clusters)  
        plot_clusters(reduced_data, labels, ax=axes[i])  
        axes[i].set_title(f"{n_clusters} Clusters", fontsize=8)

    plt.tight_layout()
    plt.show()   

def main():
    file_path = "~/datasets/breast.csv"  
    data, scaled_data = load_and_preprocess_data(file_path)
    
    reduced_data = reduce_dimensionality(scaled_data)
    
    methods = ['ward', 'single', 'complete', 'average']
    
    for method in methods:
        Z = hierarchical_clustering(scaled_data, method)  #
        
        color_threshold = np.max(Z[:, 2]) * 0.7  
        plot_dendrograms_2x2(scaled_data, methods)  # Dendrogram

    Z = hierarchical_clustering(scaled_data, method='ward')  
    
    calculate_silhouette(Z, scaled_data=scaled_data, max_clusters=10)

    
    n_clusters_list = [2, 5, 7, 9]  
    plot_clusters_for_different_n_clusters(Z, reduced_data, n_clusters_list)

 

if __name__ == "__main__":
    main()
