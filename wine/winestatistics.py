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
wine = pd.read_csv("~/datasets/WineQT.csv")
print(wine.info())
print(wine.head())
print(wine.isnull().sum())
print(wine.describe().T)
wine.drop(columns=["Id"], inplace=True)
print(wine.alcohol.describe().T)
print(wine.quality.value_counts())

def corr(wine):
    plt.figure(figsize=(12,10))
    wine_numeric = wine.drop(columns=["quality"])
    corr = wine_numeric.corr()
    sns.heatmap(data=corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm")
    plt.xticks(rotation=45)
    plt.title("Correlation Matrix")
    plt.show()

def scatter(wine):
    order = wine["quality"].value_counts().index
    palette = sns.color_palette("pastel")
    sns.countplot(data=wine,
                    x="quality",
                    palette=palette,
                    order=order
                    )
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.title("Count of Quality")
    plt.show()

def boxplot(wine):
    sns.boxplot(data=wine)
    plt.show()

def hist(wine):
    sns.histplot(data=wine,
                 x="pH",
                 kde=True)
    plt.title("Dist. of pH")
    plt.xlabel("pH")
    plt.ylabel("Count")
    plt.show()
    sns.histplot(data=wine,
                 x="density",
                 kde=True)
    plt.title("Dist. of Density")
    plt.xlabel("Density")
    plt.ylabel("Count")
    plt.show()
    sns.histplot(data=wine,
                 x="alcohol",
                 kde=True)
    plt.title("Dist. of Alcohol")
    plt.xlabel("Alcohol")
    plt.ylabel("Count")    
    plt.show()
    sns.histplot(data=wine,
                 x="fixed acidity",
                 kde=True)
    plt.title("Dist. of Fixed Acidity")
    plt.xlabel("Fixed Acidity")
    plt.ylabel("Count")    
    plt.show()    
    sns.histplot(data=wine,
                 x="residual sugar",
                 kde=True)
    plt.title("Dist. of Residual Sugar")
    plt.xlabel("Residual Sugar")
    plt.ylabel("Count")    
    plt.show()  

def quality(wine):
    qual = wine[(wine["quality"]==5) & (wine["alcohol"]>12)]
    grouping = qual.groupby(["density", "pH", "citric acid"]).mean()
    print(grouping)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=grouping, x="density", y="pH", size="citric acid", hue="citric acid", palette="viridis", sizes=(20, 200), legend=False)
    plt.title('Density vs pH (Size by Citric Acid)', fontsize=14)
    plt.xlabel('Density', fontsize=12)
    plt.ylabel('pH', fontsize=12)
    plt.show()

def quality1(wine):
    qual1= wine[(wine["quality"]==5) & (wine["alcohol"]<9)]
    grouping1 = qual1.groupby(["density", "pH", "citric acid"]).mean()
    print(grouping1)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=grouping1, x="density", y="pH", size="citric acid", hue="citric acid", 
                    palette="viridis", sizes=(20, 200), legend=False)
    plt.title('Density vs pH (Size by Citric Acid)', fontsize=14)
    plt.xlabel('Density', fontsize=12)
    plt.ylabel('pH', fontsize=12)
    plt.show()

def sugarvsalcol(wine):
    palette = sns.color_palette("pastel")
   
    sns.scatterplot(data=wine,
                    x="residual sugar",
                    y="alcohol",
                    hue="quality",
                    size="quality",
                    palette=palette)
    plt.show()

def statsc(wine):
    correlation = wine['citric acid'].corr(wine['pH'])
    print(f"Correlation between of Citric acid and pH: {correlation}")
    correlation1 = wine["citric acid"].corr(wine["alcohol"])
    print(f"Correlation between of Citric acid and alcohol: {correlation1}")
    correlation2 = wine["pH"].corr(wine["alcohol"])
    print(f"Correlation between of pH and alcohol: {correlation2}")
    grouping=wine.groupby("quality")[["alcohol", "density", "pH"]].mean()
    print(f"Mean of Alcohol & Density for every quality: {grouping}")
    return correlation, correlation1, grouping


###############################################
corr(wine)
scatter(wine)
hist(wine)
boxplot(wine)
quality(wine)
quality1(wine)
sugarvsalcol(wine)
statsc(wine)

scaled_df = pd.DataFrame(wine, columns=wine.columns)

# Pairplot kullanmak
sns.pairplot(scaled_df)
plt.xticks(rotation=90)
plt.suptitle("Pairplot of Scaled Data", y=1.02)
plt.show()
