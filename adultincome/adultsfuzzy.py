import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import skfuzzy as fuzz

adults = pd.read_csv("~/datasets/adult.csv")  
print(adults.head())
print(adults.describe())
print(adults.info())

adults_encoded = pd.get_dummies(adults, columns=['income', 'occupation', 'sex', 
                                                'race', 'workclass', 'marital.status'], drop_first=True)

numeric_cols = adults_encoded.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric Columns: {numeric_cols}")

# MinMaxScaler 
scaler = MinMaxScaler()
adult_scaled = scaler.fit_transform(adults_encoded[numeric_cols])
print(adult_scaled.shape)

def fuzzy_cmeans(scaled_data, n_clusters=2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(scaled_data.T, c=n_clusters, m=1.3, maxiter=100, error=0.005, init=None)
    hard_assignment = np.argmax(u, axis=0)
    return cntr, u, hard_assignment  

def calculate_validity_indices(scaled_data, hard_assignment, u):
    # Partition Coefficient (PC)
    pc = np.sum(u**2) / scaled_data.shape[0]
    # Partition Entropy (PE)
    pe = -np.sum(u * np.log(u + 1e-10)) / scaled_data.shape[0]  # Adding epsilon to avoid log(0)
    # Silhouette Score
    sil_score = silhouette_score(scaled_data, hard_assignment)
    # Calinski-Harabasz Score
    calinski_harabasz = calinski_harabasz_score(scaled_data, hard_assignment)
    # Davies-Bouldin Score
    db_index = davies_bouldin_score(scaled_data, hard_assignment)
    return pc, pe, sil_score, calinski_harabasz, db_index


def visualize_clusters(scaled_data, max_clusters=10, min_clusters=2, numeric_cols=None):
    num_plots = max_clusters - min_clusters + 1
    rows = (num_plots // 3) + (num_plots % 3 > 0)
    fig, axs = plt.subplots(rows, 3, figsize=(15, 12))
    axs = axs.flatten()

    cluster_numbers = range(min_clusters, max_clusters + 1)
    results = []

    for i, n_clusters in enumerate(cluster_numbers):
        cntr, u, hard_assignment = fuzzy_cmeans(scaled_data, n_clusters=n_clusters)

        pc, pe, sil_score, calinski, db_index = calculate_validity_indices(scaled_data, hard_assignment, u)
        
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
        
        cmap = plt.colormaps["tab20"]

        for cluster in range(n_clusters):
            cluster_points = np.where(hard_assignment == cluster)
            
            axs[i].scatter(
                pca_data[cluster_points[0], 0], pca_data[cluster_points[0], 1], 
                c=cmap(cluster / n_clusters), 
                s=40, alpha=0.5, label=f"Cluster {cluster + 1}"
            )
        
        axs[i].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=40, label="Centers")

        axs[i].set_title(f"Fuzzy C-Means with {n_clusters} Clusters", fontsize=8)
        axs[i].set_xlabel("Yaş ve Çalışma Haftası PCA1", fontsize=6)
        axs[i].set_ylabel(" Eğitim Seviyesi PCA2", fontsize=6)
        axs[i].legend(fontsize=6)

    plt.tight_layout()
    plt.show()

    print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')
    print(f'Cumulative Explained Variance: {np.cumsum(pca.explained_variance_ratio_)}')

    pca_loadings_df = pd.DataFrame(pca.components_.T, 
                                    columns=['PCA1', 'PCA2'],
                                    index=numeric_cols)  

    print(f"PCA Loadings Table:\n{pca_loadings_df}")


def evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10):
    results = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        cntr, u, hard_assignment = fuzzy_cmeans(scaled_data, n_clusters=n_clusters)
        pc, pe, sil_score, calinski, db_index = calculate_validity_indices(scaled_data, hard_assignment, u)
        results.append({
            'Clusters': n_clusters,
            'PC': pc,
            'PE': pe,
            'Silhouette Score': sil_score,
            'Calinski Harabasz Score': calinski,
            'Davies Bouldin Score': db_index
        })
    return pd.DataFrame(results)


################################################

scaled_data = adult_scaled  
results_df = evaluate_clusters(scaled_data, min_clusters=2, max_clusters=10)

visualize_clusters(scaled_data, max_clusters=10, min_clusters=2, numeric_cols=numeric_cols)

print(results_df)
