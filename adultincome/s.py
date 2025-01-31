import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Veriyi yükleme ve ön işleme
def load_and_preprocess_data(file_path):
    adults = pd.read_csv(file_path)
    
    # Kategorik değişkenleri encode etme (one-hot encoding)
    adults_encoded = pd.get_dummies(adults, columns=['income', 'occupation', 'sex', 
                                                     'race', 'workclass', 'marital.status'], drop_first=True)
    
    # Sadece sayısal kolonları alıyoruz
    numeric_cols = adults_encoded.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric Columns: {numeric_cols}")
    
    # Veriyi ölçeklendir
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(adults_encoded[numeric_cols])
    
    return adults, scaled_data

# Cosine mesafesi hesapla
def compute_cosine_distance(scaled_data):
    # Cosine mesafesini hesapla
    cosine_distance = pairwise_distances(scaled_data, metric='cosine')
    return cosine_distance

# Mesafe eğrisini çizme ve KneeLocator ile dirsek noktasını bulma
def plot_distance_curve(cosine_distance):
    # 20. en yakın komşuya kadar olan mesafeyi alalım
    distances = np.sort(cosine_distance[:, 20], axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.xlabel("Data points")
    plt.ylabel("Distance to 20th Nearest Neighbor")
    plt.title("Distance Curve (Cosine Distance)")
    plt.show()

    # Dirsek noktasını tespit et
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    knee.plot_knee()
    plt.show()

    # Dirsek noktasındaki mesafe (eps için önerilen değer)
    print(f"Suggested epsilon (eps) from knee: {distances[knee.knee]}")
    return distances[knee.knee]

# DBSCAN parametre kombinasyonları ile kümeleme yapma
def apply_dbscan_with_precomputed_distance(cosine_distance, eps, min_samples):
    # DBSCAN'ı kullanırken 'precomputed' parametresi ile mesafeyi sağlıyoruz
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusters = dbscan.fit_predict(cosine_distance)
    return clusters, dbscan

# Kümeleme sonuçlarını görselleştirme
def plot_clusters(X, labels, eps, ax):
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=["cluster-{}".format(x) if x != -1 else "noise" for x in labels],
                    palette="Set2", ax=ax, s=60, alpha=0.7, legend="full")
    ax.set_title(f'DBSCAN (eps={round(eps, 2)})', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.grid(True)

# Ana fonksiyon
def main():
    # Veriyi yükle ve ölçeklendir
    file_path = "/Users/sim/dev/python/Fuzzy-KmeansCVI/datas/adult.csv" # Verinizin yolunu buraya ekleyin
    data, scaled_data = load_and_preprocess_data(file_path)
    
    # Cosine mesafesini hesapla
    cosine_distance = compute_cosine_distance(scaled_data)

    # Mesafe eğrisini çizme ve önerilen eps değerini bulma
    recommended_eps = plot_distance_curve(cosine_distance)
    
    # DBSCAN'ı değerlendir ve önerilen eps değeri ile kümeleme yapalım
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    
    clusters, db = apply_dbscan_with_precomputed_distance(cosine_distance, eps=recommended_eps, min_samples=15)
    ax = fig.add_subplot(1, 2, 1)
    plot_clusters(scaled_data, clusters, recommended_eps, ax)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
