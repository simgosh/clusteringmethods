import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load and check dataset
wine = pd.read_csv("~/datasets/WineQT.csv")
wine = wine.drop(columns=["Id", "quality"])  

# Check the first few rows and dataset info
print(wine.head())
print(wine.info())

scaler = RobustScaler()
wine_scaled = scaler.fit_transform(wine)

n_clusters = 3
m = 1.3  # Fuzziness parameter
pca = PCA(n_components=2)
pca_result = pca.fit_transform(wine_scaled)

# Create a DataFrame with the PCA components
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

# Plotting using subplots
fig, axes = plt.subplots(3, 3, figsize=(14, 10))  # 3x3 grid for clusters 2 to 10
axes = axes.ravel()  # Flatten the axes array for easy indexing

indices = []

for c in range(2, 11):  # Küme sayısını 2'den 10'a kadar değiştir
    # Perform Fuzzy C-means clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(wine_scaled.T, c=c, m=m, maxiter=100, error=0.005, init=None)
    
    # Get hard assignment (most likely cluster for each data point)
    hard_assignment = np.argmax(u, axis=0)
    
    # Add clustering results to PCA dataframe
    pca_df['cluster'] = hard_assignment
    
    # Plot the PCA results with clustering labels on the respective subplot
    scatter = axes[c-2].scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['cluster'], cmap='viridis', alpha=0.6)
    axes[c-2].set_title(f"Fuzzy C-means Clustering (Clusters = {c})", fontsize=8)
    axes[c-2].set_xlabel("PCA1", fontsize=8)
    axes[c-2].set_ylabel("PCA2", fontsize=8)
    
    # Project cluster centers to 2D PCA space
    centers_2d = pca.transform(cntr)  # Transform cluster centers into 2D PCA space
    
    # Plot cluster centers
    axes[c-2].scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=100, label="Centroids")
    axes[c-2].legend(loc='upper right', fontsize=9)
    
    # Calculate the cluster validation indices
    sil_score = silhouette_score(wine_scaled, hard_assignment)
    pc = np.sum(u**2) / wine_scaled.shape[0]
    pe = -np.sum(u * np.log(u + 1e-10)) / wine_scaled.shape[0]
    calinski_harabasz = calinski_harabasz_score(wine_scaled, hard_assignment)
    fpd_index = np.sum(u**2) / (c * wine_scaled.shape[0])
    db_index = davies_bouldin_score(wine_scaled, hard_assignment)
    
    indices.append([c, pc, pe, sil_score, calinski_harabasz, fpd_index, db_index])

# Convert indices to DataFrame
indices_df = pd.DataFrame(indices, columns=["Clusters", "PC", "PE", "Silhouette Score", "CH Score", "FPD", "DB Index"])
print(indices_df)

# Plot the cluster validation indices
fig, ax = plt.subplots(figsize=(12, 7))
indices_df.plot(x="Clusters", y=["PC", "PE", "Silhouette Score", "CH Score", "FPD", "DB Index"], marker="o", ax=ax)
plt.title("Cluster Validation Indices Over Different Numbers of Clusters", fontsize=10)
plt.ylabel("Index Value", fontsize=10)
plt.xlabel("Number of Clusters", fontsize=10)
plt.show()
