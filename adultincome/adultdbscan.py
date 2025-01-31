import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

def load_and_preprocess_data(file_path):
    adults = pd.read_csv(file_path)
    
    # Kategorik değişkenleri encode etme (one-hot encoding)
    adults_encoded = pd.get_dummies(adults, columns=['income', 'occupation', 'sex', 
                                                     'race', 'workclass', 'marital.status'], drop_first=True)
    
    numeric_cols = adults_encoded.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric Columns: {numeric_cols}")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(adults_encoded[numeric_cols])
    
    return adults, scaled_data

def plot_distance_curve(X):
    nearest_neighbors = NearestNeighbors(n_neighbors=11,
                                         metric="euclidean",
                                        algorithm='ball_tree')  # ball_tree ile hizlanir
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    
    distances = np.sort(distances[:, 10], axis=0)
    plt.figure(figsize=(5, 5))
    plt.plot(distances)
    plt.xlabel("Points")
    plt.ylabel("Distance to 11 Nearest Neighbor")
    plt.title("Distance Curve (11th Nearest Neighbor)")
    plt.show()

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    knee.plot_knee()
    plt.show()
    
    print(f"Dirsek noktasındaki mesafe (eps için önerilen değer): {distances[knee.knee]}")
    return distances[knee.knee]

def apply_dbscan(scaled_data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="minkowski", p=4, algorithm='ball_tree')  # ball_tree ile hızlandırma
    clusters = dbscan.fit_predict(scaled_data)
    return clusters, dbscan

def plot_clusters(X, labels, eps, ax):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=["cluster-{}".format(x) if x != -1 else "noise" for x in labels],
                    palette="Set2", ax=ax, s=25, alpha=0.7, legend="full")
    ax.set_title(f'DBSCAN (eps={round(eps, 2)})', fontsize=10)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True)

def main():
    file_path = "~/datasets/adult.csv"  
    data, scaled_data = load_and_preprocess_data(file_path)
    
    pca = PCA(n_components=2)  
    X_pca = pca.fit_transform(scaled_data)

    recommended_eps = plot_distance_curve(X_pca)
    
    eps_values = [recommended_eps, 0.20, 0.45, 0.55, 0.75, 0.05, 0.01, 1]  
    
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    
    for i, eps in enumerate(eps_values, 1):
        clusters, db = apply_dbscan(X_pca, eps=eps, min_samples=20)  
        ax = fig.add_subplot(2, 4, i)
        plot_clusters(X_pca, clusters, eps, ax)  
    
    plt.tight_layout()
    plt.show()

    min_samples = range(10, 21)
    eps_values = [recommended_eps,0.25, 0.5, 0.75, 1, 2, 2.5, 0.60]
    output = []

    for ms in min_samples:
        for ep in eps_values:
            labels = DBSCAN(min_samples=ms, eps=ep).fit(scaled_data).labels_
            score = silhouette_score(scaled_data, labels)
            output.append((ms, ep, score))

    min_samples, eps, score = sorted(output, key=lambda x: x[-1])[-1]
    print(f"Best silhouette_score: {score}")
    print(f"min_samples: {min_samples}")
    print(f"eps: {eps}")

if __name__ == "__main__":
    main()

