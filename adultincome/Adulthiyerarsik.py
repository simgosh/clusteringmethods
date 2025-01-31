import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def load_and_preprocess_data(file_path):
    adults = pd.read_csv(file_path)
    
    adults_encoded = pd.get_dummies(adults, columns=['income', 'occupation', 'sex', 
                                                     'race', 'workclass', 'marital.status'], drop_first=True)
    
    numeric_cols = adults_encoded.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric Columns: {numeric_cols}")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(adults_encoded[numeric_cols])
    
    return adults, scaled_data
def reduce_dimensionality(scaled_data):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    return reduced_data


def hierarchical_clustering(scaled_data, method='single'):
    Z = linkage(scaled_data, method=method, metric="euclidean")
    return Z


def plot_dendrograms_2x2(scaled_data, methods):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  
    axes = axes.flatten()
    for i, method in enumerate(methods):
        Z = hierarchical_clustering(scaled_data, method)
        color_threshold = np.max(Z[:, 2]) * 0.7
        dendrogram(
            Z, 
            leaf_font_size=10, 
            color_threshold=color_threshold, 
            leaf_rotation=90, 
            truncate_mode='level', 
            p=5, 
            ax=axes[i]
        )
        axes[i].set_title(f"Dendrogram for {method.capitalize()} Linkage")
        axes[i].set_xlabel("Data points")
        axes[i].set_ylabel("Distance")
    plt.tight_layout()
    plt.show()


def calculate_silhouette(Z, scaled_data, max_clusters=10):
    silhouette_avg = []
    for n_clusters in range(2, max_clusters + 1):
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        silhouette_avg.append(silhouette_score(scaled_data, labels))
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_avg, marker='o')
    plt.title("Silhouette Score for Different Cluster Counts")
    plt.xlabel("Cluster Count")
    plt.ylabel("Silhouette Score")
    plt.show()


def plot_clusters_for_different_n_clusters(Z, reduced_data, n_clusters_list):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, n_clusters in enumerate(n_clusters_list):
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        sns.scatterplot(
            x=reduced_data[:, 0], 
            y=reduced_data[:, 1], 
            hue=labels, 
            palette=sns.color_palette("Set2", n_colors=len(set(labels))), 
            ax=axes[i],
            legend=False
        )
        axes[i].set_title(f"{n_clusters} Clusters")
    
    plt.tight_layout()
    plt.show()


def main():
    file_path = "/Users/sim/dev/python/Fuzzy-KmeansCVI/datas/adult.csv"
    adults, scaled_data = load_and_preprocess_data(file_path)
    
    reduced_data = reduce_dimensionality(scaled_data)
    methods = ['ward', 'single', 'complete', 'average']
    plot_dendrograms_2x2(scaled_data, methods)
    
    Z = hierarchical_clustering(scaled_data, method='single')
    calculate_silhouette(Z, scaled_data)
    plot_clusters_for_different_n_clusters(Z, reduced_data, n_clusters_list=[2, 4, 6, 8])


if __name__ == "__main__":
    main()