import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Loading original dataset and check
wine = pd.read_csv("~/datasets/WineQT.csv")
print(wine.info())
print(wine.head())
print(wine.isnull().sum())
print(wine.describe().T)
wine.drop(columns=["Id", "quality"], inplace=True)

def outlierscheck(wine):
    corr = wine.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(data=corr, annot=True, fmt=".2g")
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45)
    plt.show()

    sns.boxplot(data=wine,
            x="fixed acidity",palette="Set2")
    plt.title("Checking Outliers for Fixed Acidity")
    plt.show()

    sns.boxplot(data=wine,
            y="density", palette="Set1")
    plt.title("Checking Outliers for Density")
    plt.show()


# Scaling
def preprocess_data(wine):
    wine_reduced = wine[["fixed acidity", "density"]]
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(wine_reduced)
    return scaled_data

# KMeans clustering
def kmeans_clustering(scaled_data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++",
                    algorithm="lloyd", max_iter=100)
    kmeans.fit(scaled_data)
    return kmeans


# CVI
def calculate_validity_indices(scaled_data, kmeans):
    hard_assignment = kmeans.labels_
    
    # Silhouette Score
    sil_score = silhouette_score(scaled_data, hard_assignment)
    
    # Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(scaled_data, hard_assignment)
    
    # Davies-Bouldin Score
    db_index = davies_bouldin_score(scaled_data, hard_assignment)
    
    # Inertia value (should be low score)
    inertia = kmeans.inertia_
    
    return sil_score, calinski_harabasz, db_index, inertia


def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    intercluster_distances = []
    
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            dist = np.min(cdist(cluster_i, cluster_j)) 
            intercluster_distances.append(dist)
    
    min_intercluster_dist = np.min(intercluster_distances)  
    
    intracluster_distances = []
    for i in unique_labels:
        cluster_i = X[labels == i]
        dist_matrix = pairwise_distances(cluster_i)  
        np.fill_diagonal(dist_matrix, np.nan) 
        max_intracluster_dist = np.nanmax(dist_matrix)  
        intracluster_distances.append(max_intracluster_dist)
    
    max_intracluster_dist = np.max(intracluster_distances)  

    # Dunn Index'i hesapla
    dunn_index_value = min_intercluster_dist / max_intracluster_dist
    return dunn_index_value

# cluster visualize
def visualize_clusters(scaled_data, kmeans, data_columns):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=kmeans.labels_, palette="deep", s=60, alpha=0.7)
    
    # to add cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=100, label="Cluster Centers")
    
    plt.title("KMeans Clustering", fontsize=16)
    plt.xlabel("Fixed Acidity (Feature 1)", fontsize=12)  
    plt.ylabel("Density (Feature 2)", fontsize=12)
    plt.legend(title='Cluster')
    plt.show()

def evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10):
    results = []
    inertia_values =[]
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
        inertia_values.append(inertia) 
        visualize_clusters(scaled_data, kmeans, wine.columns)
    
    results_df = pd.DataFrame(results)
    print(results_df)

    plt.figure(figsize=(8,6))
    plt.plot(range(min_clusters, max_clusters + 1), inertia_values, marker='o', color='b', linestyle='--')
    plt.title("Elbow Method for Optimal K", fontsize=16)
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.show()
    
    return results_df


outlierscheck(wine)
scaled_data = preprocess_data(wine)
kmeans = kmeans_clustering(scaled_data, n_clusters=3)
visualize_clusters(scaled_data, kmeans, wine.columns)

evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10)
