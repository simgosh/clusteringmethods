import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from fcmeans import FCM
import skfuzzy as fuzz
from scipy import stats
from scipy.stats import chi2_contingency, zscore

#importing original dataset
breast = pd.read_csv("~/datas/breast.csv")
print(breast.isnull().sum())
breast.drop(columns=["id", "Unnamed: 32"], inplace=True)
print(breast.info())
print(breast.diagnosis.value_counts())

def detect_outliers_iqr(breast):
    df_numerical = breast.select_dtypes(include=[np.number])
    Q1 = df_numerical.quantile(0.25)
    Q3 = df_numerical.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numerical < (Q1 - 1.5 * IQR)) | (df_numerical > (Q3 + 1.5 * IQR))).sum()
    return outliers

def count(breast):
    order = breast["diagnosis"].value_counts().index
    palette = sns.color_palette("pastel")
    sns.countplot(data=breast,
                    x="diagnosis",
                    palette=palette,
                    order=order
                    )
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.title("Count of Diagnosis (B:Healthy, M:Unhealthy)")
    plt.show()

def corr(breast):
    breast_numeric = breast.drop(columns=["diagnosis"])
    corr_matrix = breast_numeric.corr()
    threshold = 0.8
    mask = abs(corr_matrix) < threshold
    high_corr = corr_matrix[abs(corr_matrix) >= threshold]
    high_corr = high_corr.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    print("Highly Correlated Features (correlation >= {:.2f}):".format(threshold))
    print(high_corr.dropna(how='all', axis=0).dropna(how='all', axis=1))

    sns.heatmap(corr_matrix, annot=True,
                fmt=".2g",
                cmap="coolwarm",
                mask=mask,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation Matrix (High Correlations)", fontsize=16)
    plt.show()


def scatterplt(breast):
    sns.scatterplot(data=breast,
                    x="radius_mean",
                    y="smoothness_mean",
                    hue="diagnosis")
    plt.title("Relationship by Radius vs Smoothness")
    plt.xlabel("Radius Mean")
    plt.ylabel("Smoothness Mean")
    plt.show()
    sns.scatterplot(data=breast,
                    x="radius_mean",
                    y="perimeter_mean",
                    hue="diagnosis")
    plt.title("Relationship by Radius vs Parameter")
    plt.xlabel("Radius Mean")
    plt.ylabel("Parameter Mean")
    plt.show()
    sns.scatterplot(data=breast,
                    x="radius_mean",
                    y="area_mean",
                    hue="diagnosis")
    plt.title("Relationship by Radius vs Area")
    plt.xlabel("Radius Mean")
    plt.ylabel("Area Mean")
    plt.show()
    sns.scatterplot(data=breast,
                    x="concavity_worst",
                    y="concavity_mean",
                    hue="diagnosis")
    plt.title("Relationship by Concavitiy Mean vs Worst")
    plt.xlabel("concavity worst")
    plt.ylabel("concavity mean")
    plt.show()

def hist(breast):
    sns.histplot(data=breast,
                 x="concavity_worst",
                 kde=True)
    plt.title("Dist. of concavity worst")
    plt.xlabel("concavity worst")
    plt.ylabel("Count")
    plt.show()
    sns.histplot(data=breast,
                 x="symmetry_mean",
                 kde=True)
    plt.title("Dist. of symmetry mean")
    plt.xlabel("symmetry mean")
    plt.ylabel("Count")
    plt.show()
    sns.histplot(data=breast,
                 x="radius_mean",
                 kde=True)
    plt.title("Dist. of radius mean")
    plt.xlabel("radius mean")
    plt.ylabel("Count")
    plt.show()

def details(breast):
    # Filter rows with diagnosis 'M'
    diagnosiss = breast[(breast["diagnosis"] == 'M') & (breast["concavity_worst"]>0.75)]
    numeric_columns = diagnosiss.select_dtypes(include=["float64", "int64"]).columns
    # Group by "concavity_worst" and "radius_worst", then compute mean for other numeric columns
    grouping = diagnosiss.groupby(["concavity_worst", "radius_worst", "texture_mean"])[numeric_columns].mean()
    print(grouping)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=grouping, x="concavity_worst", y="radius_worst", size="texture_mean", palette="viridis", sizes=(20, 200), legend=False)
    # Update title and axis labels to reflect the correct columns
    plt.title('Concavity vs Radius(for Unhealthies)', fontsize=14)
    plt.xlabel('Concavity Worst', fontsize=12)
    plt.ylabel('Radius Worst', fontsize=12)
    plt.show()

def details1(breast):
    # Filter rows with diagnosis 'M'
    diagnosiss1 = breast[(breast["diagnosis"] == 'B') & (breast["concavity_worst"]<0.25)]
    # Group by "concavity_worst" and "radius_worst", then compute mean for other numeric columns
    grouping1 = diagnosiss1.groupby(["concavity_worst", "radius_worst", "texture_mean"])[["fractal_dimension_worst",
                                                                                         "symmetry_worst",
                                                                                         "smoothness_mean",
                                                                                         "perimeter_mean"]].mean()
    print(grouping1.head(20))
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=grouping1, x="concavity_worst", y="radius_worst", size="texture_mean", palette="viridis", sizes=(20, 200), legend=False)
    # Update title and axis labels to reflect the correct columns
    plt.title('Concavity vs Radius (for Helathies)', fontsize=14)
    plt.xlabel('Concavity Worst', fontsize=12)
    plt.ylabel('Radius Worst', fontsize=12)
    plt.show()


def statsc(breast):
    correlation = breast['concavity_worst'].corr(breast['radius_worst'])
    print(f"Correlation between of Concavity and Radius: {correlation}")
    correlation1 = breast["fractal_dimension_worst"].corr(breast["perimeter_mean"])
    print(f"Correlation between of Fractal and Perimeter: {correlation1}")
    correlation2 = breast["smoothness_mean"].corr(breast["concavity_worst"])
    print(f"Correlation between of Smoothness and Concavity: {correlation2}")
    correlation3 = breast["area_mean"].corr(breast["compactness_se"])
    print(f"Correlation between of Area and Compactness: {correlation3}")
    correlation4= breast["concavity_mean"].corr(breast["perimeter_mean"])
    print(f"Correlation between of Concavity and Parameter: {correlation4}")
    correlation5= breast["area_mean"].corr(breast["perimeter_mean"])
    print(f"Correlation between of Area and Parameter: {correlation5}")    
    grouping=breast.groupby("diagnosis")[["concavity_worst", "radius_worst", "perimeter_mean"]].mean()
    print(f"Mean of Concavity & Radius for every diagnosis: {grouping}")
    return correlation, correlation1, grouping




count(breast)
outliers = detect_outliers_iqr(breast)
print(outliers)
corr(breast)
scatterplt(breast)
hist(breast)
details(breast)
details1(breast)
statsc(breast)





###### Malignant tumors (M) tend to have:
#Larger radius, perimeter, and area.
#More irregular and concave boundaries (higher concavity and concave points).
#Less smooth boundaries (lower smoothness).
#More complex boundaries (higher fractal dimension).