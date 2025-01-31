import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

# Importing dataset
wine = pd.read_csv("~/datas/WineQT.csv")
wine.drop(columns=["Id", "quality"], inplace=True)

# Preprocessing function
def preprocess_data(wine):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(wine)
    return scaled_data

# KMeans Clustering
def kmeans_clustering(scaled_data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", max_iter=300, algorithm="lloyd")
    kmeans.fit(scaled_data)
    return kmeans

# Validity indices calculation
def calculate_validity_indices(scaled_data, kmeans):
    hard_assignment = kmeans.labels_
    
    # Silhouette Score
    sil_score = silhouette_score(scaled_data, hard_assignment)
    
    # Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(scaled_data, hard_assignment)
    
    # Davies-Bouldin Score
    db_index = davies_bouldin_score(scaled_data, hard_assignment)
    
    # Inertia (the lower the better, represents total within-cluster variance)
    inertia = kmeans.inertia_
    
    return sil_score, calinski_harabasz, db_index, inertia

# Visualizing clusters using PCA
def visualize_clusters(scaled_data, kmeans, data_columns, ax):
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    centers_2d = pca.transform(kmeans.cluster_centers_)
    
    ax.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_, cmap='viridis', s=40, alpha=0.7)  
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=70, label="Cluster Centers")
    
    ax.set_title(f"KMeans Clustering (Clusters = {kmeans.n_clusters})", fontsize=10)
    ax.set_xlabel("PCA1 (Chlorides)", fontsize=8)
    ax.set_ylabel("PCA2 (Sugar)", fontsize=8)
    ax.legend(loc='upper right', fontsize=8)

# Evaluating clusters for different n_clusters
def evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10):
    results = []
    inertia_values = []
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))  # 3x3 grid for clusters 2 to 10
    axes = axes.ravel()  # Flatten the axes array for easy indexing

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)
        
        sil_score, calinski, db_index, inertia = calculate_validity_indices(scaled_data, kmeans)    
        
        # Store results for each number of clusters
        results.append({
            'Clusters': n_clusters,
            'Silhouette Score': sil_score,
            'Calinski Harabasz Score': calinski,
            'Davies Bouldin Score': db_index,
            'Inertia': inertia
        })
        inertia_values.append(inertia) 
        
        # Visualize each clustering result in the subplot
        visualize_clusters(scaled_data, kmeans, wine.columns, axes[n_clusters - 2])

    # Convert results to a DataFrame for easy inspection
    results_df = pd.DataFrame(results)
    print(results_df)
    
    # Plot the inertia values (Elbow Method)
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), inertia_values, marker='o', color='b', linestyle='--')
    plt.title("Elbow Method for Optimal K", fontsize=14)
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.show()

    return results_df

#######################################
scaled_data = preprocess_data(wine)
evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10)
