# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# Load and preprocess dataset
def load_and_preprocess(filepath, drop_columns):
    data = pd.read_csv(filepath)
    data.drop(columns=drop_columns, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title("Ölçeklendirme Öncesi Boxplot")
    plt.show()

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)

    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)  
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=scaled_df)
    plt.title("Ölçeklendirme Sonrası Boxplot")
    plt.show()

    return data, scaled_df

filepath = "/Users/sim/dev/python/Fuzzy-KmeansCVI/datasets/WineQT.csv"
drop_columns = ["Id", "quality"]  
raw_data, scaled_data = load_and_preprocess(filepath, drop_columns)


#check outliers for before scaling
def detect_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    return outliers

#Apply KMeans and return results
def apply_kmeans(data_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, init="k-means++", max_iter=300)
    cluster_labels = kmeans.fit_predict(data_scaled)
    return kmeans, cluster_labels

#PCA analysis (Reduced data and loadings)
def perform_pca(data_scaled, original_columns, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_scaled)
    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(n_components)], index=original_columns)
    explained_variance = pca.explained_variance_ratio_
    return reduced_data, loadings, explained_variance, pca

#Evaluate clustering validity
def evaluate_clustering(data_scaled, cluster_labels):
    return {
        "Silhouette Score": silhouette_score(data_scaled, cluster_labels),
        "Calinski-Harabasz Index": calinski_harabasz_score(data_scaled, cluster_labels),
        "Davies-Bouldin Index": davies_bouldin_score(data_scaled, cluster_labels)
    }

#Loop through K values for metrics and visualization
def analyze_clusters(data_scaled, original_columns, k_range):
    results = {"K": [], "Inertia": [], "Silhouette Score": [], "Calinski-Harabasz Index": [], "Davies-Bouldin Index": []}
    pca_data, loadings, _, pca_model = perform_pca(data_scaled, original_columns)  
    
    plt.figure(figsize=(15, 10))
    for k in k_range:
        kmeans, cluster_labels = apply_kmeans(data_scaled, k)
        validity_scores = evaluate_clustering(data_scaled, cluster_labels)
        
        results["K"].append(k)
        results["Inertia"].append(kmeans.inertia_)
        results["Silhouette Score"].append(validity_scores["Silhouette Score"])
        results["Calinski-Harabasz Index"].append(validity_scores["Calinski-Harabasz Index"])
        results["Davies-Bouldin Index"].append(validity_scores["Davies-Bouldin Index"])
        
        # Transform cluster centers to PCA space
        pca_cluster_centers = pca_model.transform(kmeans.cluster_centers_)

        # Visualize clusters with PCA
        plt.subplot(3, 3, k - 1)
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=cluster_labels, palette='viridis', legend=None)
        plt.scatter(pca_cluster_centers[:, 0], pca_cluster_centers[:, 1], color='red', marker='*', s=200, label="Centroid")
        plt.title(f"K = {k}")
        plt.xlabel("Acidity & Quality Axis PCA1")
        plt.ylabel("Alcohol PCA2")
        plt.grid()
    
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(results), loadings

#Plot validity metrics
def plot_validity_metrics(results):
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(results["K"], results["Inertia"], 'o-', label="Inertia (WCSS)", color='purple')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Inertia vs Number of Clusters")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(results["K"], results["Silhouette Score"], 'o-', label="Silhouette Score")
    plt.plot(results["K"], results["Calinski-Harabasz Index"], 'o-', label="Calinski-Harabasz Index")
    plt.plot(results["K"], results["Davies-Bouldin Index"], 'o-', label="Davies-Bouldin Index")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Score")
    plt.title("Cluster Validity Metrics vs Number of Clusters")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



#####################################################
# Main Execution
if __name__ == "__main__":
    # Filepath and columns to drop
    filepath = "~/datasets/WineQT.csv"
    drop_columns = ["Id", "quality"]

    # Load and scale data
    wine_data, wine_scaled = load_and_preprocess(filepath, drop_columns)
    original_columns = wine_data.columns
    outliers = detect_outliers_iqr(wine_data)
    print(outliers)

    # Analyze clusters and PCA loadings for K = 2 to 10
    k_range = range(2, 11)
    cluster_results, pca_loadings = analyze_clusters(wine_scaled, original_columns, k_range)

    # Display cluster validity results
    print(cluster_results)
    
    # Display PCA Loadings
    print("\nPCA Loadings:")
    print(pca_loadings)

    # Plot validity metrics
    plot_validity_metrics(cluster_results) 
