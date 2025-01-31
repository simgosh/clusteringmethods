import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

# Import dataset
adults = pd.read_csv("~/datasets/adult.csv")
print(adults.info())
print(adults.head())
print(adults.isnull().sum())
print(adults.describe().T)


def preprocess_data(adults):
    adults_encoded = pd.get_dummies(adults, columns=['income', 'occupation', 'sex', 
                                                     'race', 'workclass', 'marital.status'], drop_first=True)
    
    numeric_cols = adults_encoded.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric Columns: {numeric_cols}")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(adults_encoded[numeric_cols])
    return scaled_data
    
def kmeans_clustering(scaled_data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++")
    kmeans.fit(scaled_data)
    return kmeans

def calculate_validity_indices(scaled_data, kmeans):
    hard_assignment = kmeans.labels_
    
    # Validity indices
    sil_score = silhouette_score(scaled_data, hard_assignment)
    calinski_harabasz = calinski_harabasz_score(scaled_data, hard_assignment)
    db_index = davies_bouldin_score(scaled_data, hard_assignment)
    inertia = kmeans.inertia_
    
    return sil_score, calinski_harabasz, db_index, inertia

def visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10, data_columns=None):
    num_plots = max_clusters - min_clusters + 1
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Calculate number of rows
    fig, axs = plt.subplots(rows, 3, figsize=(15, 12))
    axs = axs.flatten()
    
    cluster_numbers = range(min_clusters, max_clusters + 1)

    for i, n_clusters in enumerate(cluster_numbers):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        centers_2d = pca.transform(kmeans.cluster_centers_)

        # Scatter plot with the clusters
        sns.scatterplot(ax=axs[i], x=pca_data[:, 0], y=pca_data[:, 1], 
                        hue=kmeans.labels_, palette="deep", s=30)
        axs[i].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=40, alpha=0.5, label="Cluster Centers")

        axs[i].set_title(f"KMeans with {n_clusters} Clusters", fontsize=10)
        axs[i].set_xlabel("PCA1", fontsize=8)
        axs[i].set_ylabel("PCA2", fontsize=8)
        axs[i].legend(title="Cluster", fontsize=6)

    for j in range(len(cluster_numbers), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def elbow_method(scaled_data, max_clusters=10):
    silhouette_scores = []  
    for n_clusters in range(2, max_clusters + 1):
        kmeans = kmeans_clustering(scaled_data, n_clusters=n_clusters)
        sil_score = silhouette_score(scaled_data, kmeans.labels_)  
        silhouette_scores.append(sil_score)  
    return silhouette_scores

def plot_elbow_method(scaled_data, max_clusters=10):
    silhouette_scores = elbow_method(scaled_data, max_clusters=max_clusters)
    cluster_counts = list(range(2, max_clusters + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_counts, silhouette_scores, marker='o', linestyle='-', color='b')
    plt.title("Elbow Method - Silhouette Scores")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid()
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
    print(results_df)
    return results_df

# Main process
scaled_data = preprocess_data(adults)
results_df = evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10)
visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10, data_columns=adults.columns)
plot_elbow_method(scaled_data, max_clusters=10)
