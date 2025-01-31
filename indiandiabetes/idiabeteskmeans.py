import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from fcmeans import FCM
import skfuzzy as fuzz
from sklearn.metrics import rand_score, v_measure_score
from sklearn.cluster import KMeans

#importing original dataset
data = pd.read_csv("~/datasets/diabetes.csv")
print(data.info())
print(data.head())
print(data.isnull().sum())
print(data.describe().T)
data.drop(columns=["Outcome"], inplace=True)

def preprocess_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def kmeans_clustering(scaled_data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)
    return kmeans

def calculate_validity_indices(scaled_data, kmeans):
    hard_assignment = kmeans.labels_
    
    # Silhouette Score
    sil_score = silhouette_score(scaled_data, hard_assignment)
    
    # Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(scaled_data, hard_assignment)
    
    # Davies-Bouldin Score
    db_index = davies_bouldin_score(scaled_data, hard_assignment)

    dunn_idx = dunn_index(scaled_data, hard_assignment)

    #inertia degeri (dusuk olmasi iyidir, toplam içsel varyansı temsil eder.)
    inertia = kmeans.inertia_
    
    return sil_score, calinski_harabasz, db_index, dunn_idx, inertia

def elbow_method(scaled_data, min_clusters=2, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)
        sil_score = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores.append(sil_score)

    # Plotting the Elbow Graph
    plt.figure(figsize=(8, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', color='red')
    plt.title("Elbow Method For Optimal k (Silhouette Score)", fontsize=14)
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.xticks(range(min_clusters, max_clusters + 1))
    plt.grid(True)
    plt.show()

def dunn_index(scaled_data, labels):
        distances = pairwise_distances_argmin_min(scaled_data, scaled_data)
        min_distance = np.min(distances)
        max_distance = np.max(distances)

        return min_distance / max_distance

# Function to visualize clustering in subplots for clusters ranging from min to max clusters
def visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10):
    num_plots = max_clusters - min_clusters + 1
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Ensure enough rows
    fig, axs = plt.subplots(rows, 3, figsize=(15, 12))
    axs = axs.flatten()  # Flatten to treat as 1D array

    cluster_numbers = range(min_clusters, max_clusters + 1)
    
    for i, n_clusters in enumerate(cluster_numbers):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)

        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        # Transform the cluster centers using PCA
        centers_2d = pca.transform(kmeans.cluster_centers_)

        # Scatter plot with the clusters
        sns.scatterplot(ax=axs[i], x=pca_data[:, 0], y=pca_data[:, 1], hue=kmeans.labels_,
                        palette="deep", s=60, alpha=0.7)
        axs[i].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=100, label="Cluster Centers")

        axs[i].set_title(f"KMeans Clustering with {n_clusters} Clusters", fontsize=12)
        axs[i].set_xlabel("Demographics Features PCA1", fontsize=10)
        axs[i].set_ylabel("Metabolic Health PCA2", fontsize=10)
        axs[i].legend(title="Cluster", fontsize=8)

    for j in range(num_plots, len(axs)):  # Remove unused subplots
        fig.delaxes(axs[j])
    plt.tight_layout()

    # Perform PCA components extraction for better visualization
    pca_components = pd.DataFrame(pca.components_ , index=["PCA1", "PCA2"])
    print("PCA Components (Loadings):")
    print(pca_components)

    plt.show()


def evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10):
    results = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)
        sil_score, calinski, db_index, dunn_idx, inertia = calculate_validity_indices(scaled_data, kmeans)    
        results.append({
            'Clusters': n_clusters,
            'Silhouette Score': sil_score,
            'Calinski Harabasz Score': calinski,
            'Davies Bouldin Score': db_index,
            'Dunn Index': dunn_idx,
            'Inertia': inertia
        })
    
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df


#######################################
scaled_data = preprocess_data(data)

# Cluster and visualize results for clusters between 2 and 10
visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10)

elbow_method(scaled_data, min_clusters=2, max_clusters=10)

# Evaluate the clustering for clusters ranging from 2 to 10 and display the validity indices
results_df = evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10)
print(results_df)