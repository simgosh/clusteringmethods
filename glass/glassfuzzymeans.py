import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import skfuzzy as fuzz

# Import dataset
glass = pd.read_csv("~/datasets/glass.csv")

def preprocess_data(glass):
    glass_numeric = glass.drop(columns=["Type"])  # Drop categorical column
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(glass_numeric)
    return scaled_data

# Perform fuzzy c-means clustering
def fuzzy_cmeans(scaled_data, n_clusters=2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        scaled_data.T,  # Transpose data for the fuzzy function
        c=n_clusters,   # Number of clusters
        m=1.6,          # Fuzzy exponent (softness)
        maxiter=100,    # Max iterations
        error=0.005,    # Convergence threshold
        init=None       
    )
    return cntr, u

# Calculate cluster validity indices
def calculate_validity_indices(scaled_data, u):
    pc = np.sum(u**2) / scaled_data.shape[0]  # Partition Coefficient
    pe = -np.sum(u * np.log(u + 1e-10)) / scaled_data.shape[0]  # Partition Entropy
    sil_score = silhouette_score(scaled_data, np.argmax(u, axis=0))  # Silhouette score
    calinski_harabasz = calinski_harabasz_score(scaled_data, np.argmax(u, axis=0))
    db_index = davies_bouldin_score(scaled_data, np.argmax(u, axis=0))
    
    return pc, pe, sil_score, calinski_harabasz, db_index

def visualize_clusters_in_single_figure(scaled_data, min_clusters=2, max_clusters=10):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Create a 2x4 grid
    axes = axes.flatten()  
    
    validity_results = []
    
    for idx, n_clusters in enumerate(range(min_clusters, max_clusters + 1)):
        if idx >= 8:  
            break
        cntr, u = fuzzy_cmeans(scaled_data, n_clusters=n_clusters)

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        centers_2d = pca.transform(cntr)

        cluster_assignments = np.argmax(u, axis=0)
        
        scatter = axes[idx].scatter(
            pca_data[:, 0], pca_data[:, 1],
            c=cluster_assignments,  
            cmap='tab10',  
            s=40, alpha=0.7
        )

        axes[idx].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=50, label="Centers")
        axes[idx].set_title(f"Fuzzy C-Means with {n_clusters} Clusters", fontsize=10)
        axes[idx].set_xlabel("PCA1", fontsize=8)
        axes[idx].set_ylabel("PCA2", fontsize=8)
        axes[idx].legend(fontsize=6)

        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        
        # Calculate validity indices for the current clustering
        pc, pe, sil_score, calinski_harabasz, db_index = calculate_validity_indices(scaled_data, u)
        
        validity_results.append({
            'Clusters': n_clusters,
            'Partition Coefficient': pc,
            'Partition Entropy': pe,
            'Silhouette Score': sil_score,
            'Calinski Harabasz Score': calinski_harabasz,
            'Davies Bouldin Score': db_index
        })

    if len(axes) > 8:
        fig.delaxes(axes[8])

    plt.tight_layout()
    plt.show()

    validity_df = pd.DataFrame(validity_results)
    return validity_df


scaled_data = preprocess_data(glass)

validity_df = visualize_clusters_in_single_figure(scaled_data, min_clusters=2, max_clusters=9)

print(validity_df)
