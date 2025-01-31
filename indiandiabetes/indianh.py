import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import skfuzzy as fuzz

data = pd.read_csv("~/datasets/diabetes.csv")
print(data.info())
print(data.head())
print(data.isnull().sum())
print(data.describe().T)
data.drop(columns=["Outcome"], inplace=True)


def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def fuzzy_cmeans(scaled_data, n_clusters=2, m=1.4, maxiter=150, error=0.005, init=None):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        scaled_data.T,  
        c=n_clusters,   
        m=m,            
        maxiter=maxiter,  
        error=error,   
        init=init        
    )
    return cntr, u 

def calculate_validity_indices(scaled_data, u):
    pc = np.sum(u**2) / scaled_data.shape[0]  # Partition Coefficient
    pe = -np.sum(u * np.log(u + 1e-10)) / scaled_data.shape[0]  # Partition Entropy
    sil_score = silhouette_score(scaled_data, np.argmax(u, axis=0))  # Silhouette score
    calinski_harabasz = calinski_harabasz_score(scaled_data, np.argmax(u, axis=0))
    db_index = davies_bouldin_score(scaled_data, np.argmax(u, axis=0))
    
    return pc, pe, sil_score, calinski_harabasz, db_index


def visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10):
    num_plots = max_clusters - min_clusters + 1
    rows = (num_plots // 3) + (num_plots % 3 > 0)
    fig, axs = plt.subplots(rows, 3, figsize=(15, 12))
    axs = axs.flatten()

    cluster_numbers = range(min_clusters, max_clusters + 1)
    results = []

    for i, n_clusters in enumerate(cluster_numbers):
        cntr, u = fuzzy_cmeans(scaled_data, n_clusters=n_clusters)

        # Validity metriklerini hesaplama
        pc, pe, sil_score, calinski, db_index = calculate_validity_indices(scaled_data, u)

        results.append({
            'Clusters': n_clusters,
            'PC': pc,
            'PE': pe,
            'Silhouette Score': sil_score,
            'Calinski Harabasz Score': calinski,
            'Davies Bouldin Score': db_index
        })

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        centers_2d = pca.transform(cntr)

        hard_assignment = np.argmax(u, axis=0)

        for cluster in range(n_clusters):
            cluster_points = np.where(hard_assignment == cluster)
            axs[i].scatter(
                pca_data[cluster_points, 0], pca_data[cluster_points, 1], 
                c=plt.cm.tab10(cluster / n_clusters),  
                s=40, alpha=0.5, label=f"Cluster {cluster + 1}"
            )

        axs[i].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=60, label="Centers")
        axs[i].set_title(f"Fuzzy C-Means with {n_clusters} Clusters", fontsize=10)
        axs[i].set_xlabel("PCA1", fontsize=8)
        axs[i].set_ylabel("PCA2", fontsize=8)
        axs[i].legend(fontsize=6)

    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(results)
    return results_df


scaled_data = preprocess_data(data)

results_df = visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10)

print("Cluster Validity Indices:")
print(results_df)
