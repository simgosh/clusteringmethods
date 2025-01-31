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
glass = pd.read_csv("~/datasets/glass.csv")
print(glass.info())
print(glass.head())
print(glass[glass["Type"]==6])
print(glass[glass["Type"]==5].head(9))
print(glass.describe().T)
print(glass.Type.value_counts())
print(glass.duplicated().sum())
#check duplicated rows
duplicates = glass[glass.duplicated()]
print(duplicates)
all_duplicates = glass[glass.duplicated(keep=False)]
print(all_duplicates)
cleaned_glass = glass.drop_duplicates()
print(cleaned_glass) #remove duplicated rows
num_features = len(glass.columns)
n_cols = 4  
n_rows = (num_features // n_cols) + (num_features % n_cols > 0)  

for i, column in enumerate(glass.columns):
    plt.subplot(n_rows, n_cols, i + 1)  
    sns.boxplot(glass[column])
    plt.title(f"Boxplot: {column}")  

plt.tight_layout()  
plt.show()

def corr(glass):
    plt.figure(figsize=(12,10))
    corr = glass.corr()
    sns.heatmap(data=corr,
                annot=True,
                fmt=".2f",
                cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def scatplot(glass):
    plt.figure(figsize=(12,10))
    sns.scatterplot(data=glass,
                    x="RI",
                    y="Ca",
                    hue="Type")
    plt.title("Relationship by RI vs Ca")
    plt.show()

def count(glass):
    palette = sns.color_palette("pastel")
    glass_order = glass["Type"].value_counts().index
    sns.countplot(data=glass,
                  x="Type",
                  palette=palette,
                  order=glass_order)
    plt.title("Count of Types")
    plt.show()

def hist(glass):
    sns.histplot(data=glass,
                 x="Fe",
                 kde=True)
    plt.title("Dist. of Fe")
    plt.xlabel("Fe")
    plt.ylabel("Count")
    plt.show()
    sns.histplot(data=glass,
                 x="Si",
                 kde=True)
    plt.title("Dist. of Si")
    plt.xlabel("Si")
    plt.ylabel("Count")
    plt.show()
    sns.histplot(data=glass,
                 x="Na",
                 kde=True)
    plt.title("Dist. of Na")
    plt.xlabel("Na")
    plt.ylabel("Count")    
    plt.show()
    sns.histplot(data=glass,
                 x="Ca",
                 kde=True)
    plt.title("Dist. of Ca")
    plt.xlabel("Ca")
    plt.ylabel("Count")    
    plt.show()    

def plot1(glass):
    glassy = glass[glass["Type"] == 2] 
    sns.kdeplot(data=glassy, x="RI", fill=True, palette="Set2")
    plt.title("KDE Plot of RI for Type == 2")
    plt.show()

def statsc(glass):
    correlation = glass['RI'].corr(glass['Ca'])
    print(f"Correlation between of Ri and Ca: {correlation}")
    correlation1 = glass["RI"].corr(glass["Si"])
    print(f"Correlation between of Ri and Si: {correlation1}")
    grouping=glass.groupby("Type")[["Ca", "Si"]].mean()
    print(f"Mean of Ca & Si for every Type: {grouping}")
    return correlation, correlation1, grouping

def relationship(glass):
    grouping1= glass.groupby("Type").agg({
        "K": "mean",  # Calculate mean for 'Glucose'
        "Al": "mean"  # Calculate mean for 'Insulin'      
    }).reset_index()
    print(grouping1)

def stat(glass):
    group1 = glass[glass["Type"]==2]["RI"]
    group2= glass[glass["Type"]==6]["RI"]

    t_stat, p_value = stats.ttest_ind(group1, group2) #independence 2 different group
    print(f"T-Statistics: {t_stat}")
    print(f"P-Value: {p_value}")
    if p_value<0.05:
        print("There are significant differences between Types in terms of RI value.")
    else:
        print("There are NO significant differences between Types in terms of RI value.")
    return t_stat, p_value   

def statistical(glass):
    covariance = glass[['Ca', 'RI']].cov().iloc[0, 1]
    print(f"Covariance: {covariance}")

    correlation = glass[['Ca', 'RI']].corr().iloc[0, 1]
    print(f"Pearson Correlation: {correlation}")

    z_scores = zscore(glass[['Ca', 'RI']])
    print(f"Z-Scores:\n{z_scores}") 

    outliers = (z_scores > 3) | (z_scores < -3)  
    outliers_data = glass[outliers.any(axis=1)]  # choose the outliers rows from all values
    print("Outliers Values:")
    print(outliers_data)
     
def scaling(glass):
    glass_numeric = glass.drop(columns=["Type"])
    corr_numeric = glass_numeric.corr()
    sns.heatmap(data=corr_numeric, annot=True, fmt=".2g")
    plt.show()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(glass_numeric)
    scaled_df = pd.DataFrame(scaled, columns=glass_numeric.columns)
    print(scaled_df.head())
    return scaled_df

def si_type_chi2_test(scaled_df):
    # Chi-Square Test: Type and SI between relationships.
    contingency_table = pd.crosstab(scaled_df['Type'], glass['Si'])
    # Chi-Square Test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-Square Stats: {chi2_stat}")
    print(f"P-Value: {p_value}")
    if p_value < 0.05:
        print("There are significiant relationship between Type and SI.")
    else:
        print("There are no significiant relationship between Type and SI.")
    # Chi-Square Test: Type and CA between relationships.
    contingency_table1 = pd.crosstab(scaled_df['Type'], glass['Ca'])
    # Chi-Square Test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table1) 
    print(f"Chi-Square Stats: {chi2_stat}")
    print(f"P-Value: {p_value}")
    if p_value < 0.05:
        print("There are significiant relationship between Type and Ca.")
    else:
        print("There are no significiant relationship between Type and Ca.")   
    # Chi-Square Test: Type and AL between relationships.
    contingency_table2 = pd.crosstab(scaled_df['Type'], glass['Al'])
    # Chi-Square Test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table2) 
    print(f"Chi-Square Stats: {chi2_stat}")
    print(f"P-Value: {p_value}")
    if p_value < 0.05:
        print("There are significiant relationship between Type and Al.")
    else:
        print("There are no significiant relationship between Type and Al.") 


######################################
corr(glass)
scatplot(glass)
count(glass)
hist(glass)
plot1(glass)
statsc(glass)
relationship(glass)
stat(glass)
statistical(glass)
scaling(glass)
si_type_chi2_test(glass)
