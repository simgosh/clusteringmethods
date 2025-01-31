import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def load_and_preprocess_data(file_path):
    glass = pd.read_csv(file_path)
    glass_numeric = glass.drop(columns=["Type"])  
    scaler = StandardScaler()  
    scaled_data = scaler.fit_transform(glass_numeric)
    return glass, scaled_data

def plot_distance_curve(X):
    nearest_neighbors = NearestNeighbors(n_neighbors=20, 
                                          leaf_size=15,
                                          metric='minkowski', algorithm='kd_tree', p=5)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    
    distances = np.sort(distances[:, 19], axis=0)
    
    plt.figure(figsize=(5, 5))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel("Distance to 20 Nearest Neighbor")
    plt.title("Distance Curve (20th Nearest Neighbor)")
    plt.show()

    # KneeLocator ile dirsek noktasını bulma
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    knee.plot_knee()
    plt.show()
    
    print(f"Dirsek noktasındaki mesafe (eps için önerilen değer): {distances[knee.knee]}")
    return distances[knee.knee]

def apply_dbscan(scaled_data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)
    return clusters, dbscan

def plot_clusters(X, labels, eps, ax):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=["cluster-{}".format(x) if x != -1 else "noise" for x in labels],
                    palette="Set2", ax=ax, s=60, alpha=0.7, legend="full")
    ax.set_title(f'DBSCAN (eps={round(eps, 2)})', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True)

def main():
    file_path = "~/datasets/glass.csv"  
    data, scaled_data = load_and_preprocess_data(file_path)
    
    pca = PCA(n_components=2)
    X = pca.fit_transform(scaled_data)

    recommended_eps = plot_distance_curve(X)
    
    eps_values = [recommended_eps, 0.2, 0.3, 0.4 ,0.55, 0.70, 0.82, 0.95, 1.1, 1.50]  

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    
    for i, eps in enumerate(eps_values, 1):
        clusters, db = apply_dbscan(X, eps=eps, min_samples=5)  
        ax = fig.add_subplot(2, 5, i)
        plot_clusters(X, clusters, eps, ax)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
