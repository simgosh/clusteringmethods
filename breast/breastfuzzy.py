import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import skfuzzy as fuzz

# Veriyi yükle ve ön işleme
breast = pd.read_csv("~/datasets/breast.csv")
breast.drop(columns=["id", "Unnamed: 32", "diagnosis"], inplace=True)

# Veriyi standartlaştır
def preprocess_data(breast):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(breast)
    return scaled_data

def fuzzy_cmeans(scaled_data, n_clusters=2, m=1.2, maxiter=100, error=0.005, init=None):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        scaled_data.T,  
        c=n_clusters,   
        m=m,            
        maxiter=maxiter, 
        error=error,   
        init=init       
    )
    return cntr, u 

def calculate_validity_indices_soft(scaled_data, u):
    # Partition Coefficient (PC)
    pc = np.sum(u**2) / scaled_data.shape[0]
    
    # Partition Entropy (PE)
    pe = -np.sum(u * np.log(u + 1e-10)) / scaled_data.shape[0]  # Avoid log(0) error
    
    hard_assignment = np.argmax(u, axis=0)
    calinski = calinski_harabasz_score(scaled_data, hard_assignment)
    db_index = davies_bouldin_score(scaled_data, hard_assignment)
    
    return pc, pe, calinski, db_index


def visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10):
    num_plots = max_clusters - min_clusters + 1
    rows = (num_plots // 3) + (num_plots % 3 > 0)
    fig, axs = plt.subplots(rows, 3, figsize=(15, 12))
    axs = axs.flatten()

    cluster_numbers = range(min_clusters, max_clusters + 1)
    results = []

    for i, n_clusters in enumerate(cluster_numbers):
        cntr, u = fuzzy_cmeans(scaled_data, n_clusters=n_clusters)

        pc, pe, calinski, db_index = calculate_validity_indices_soft(scaled_data, u)

        results.append({
            'Clusters': n_clusters,
            'PC': pc,
            'PE': pe,
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
                s=40, alpha=0.7, label=f"Cluster {cluster + 1}"
            )

        axs[i].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=50, label="Centers")
        axs[i].set_title(f"Fuzzy C-Means with {n_clusters} Clusters", fontsize=10)
        axs[i].set_xlabel("PCA1", fontsize=8)
        axs[i].set_ylabel("PCA2", fontsize=8)
        axs[i].legend(fontsize=6)

    plt.tight_layout()
    plt.show()

    results_df = pd.DataFrame(results)
    return results_df


scaled_data = preprocess_data(breast)

results_df = visualize_clusters_in_subplots(scaled_data, min_clusters=2, max_clusters=10)

print("Cluster Validity Indices:")
print(results_df)
