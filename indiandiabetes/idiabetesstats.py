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
data = pd.read_csv("~/datasets/diabetes.csv")
print(data.info())
print(data.head())
print(data.isnull().sum())
print(data.describe().T)
print(data.Age.value_counts())
num_features = len(data.columns)
n_cols = 4  
n_rows = (num_features // n_cols) + (num_features % n_cols > 0)  
for i, column in enumerate(data.columns):
    plt.subplot(n_rows, n_cols, i + 1)  
    sns.boxplot(data[column])
    plt.title(f"Boxplot: {column}")  

plt.tight_layout()  
plt.show()

#examining Statistical calculations
def plot(data):
    sns.scatterplot(data=data,
                    x="BMI",
                    y="Age",
                    hue="Outcome")
    plt.title("Scatterplot by Age vs Pregnancy")
    plt.show()

def plot2(data):
    pregnancy = data[data["Pregnancies"] > 10] 
    sns.countplot(data=pregnancy,
                    x="Age",
                    hue="Outcome")
    plt.title("Pregnancy More Than 10 by Age")
    plt.show()

def plot3(data):
    pregnancy = data[data["Pregnancies"] == 0] 
    sns.countplot(data=pregnancy,
                    x="Age",
                    hue="Outcome")
    plt.title("Pregnancy==0 by Age")
    plt.show()

def plot4(data):
    ages = data[data["Age"]<30]
    grouping = ages.groupby("Pregnancies").agg({
        "Glucose": "mean",  # Calculate mean for 'Glucose'
        "Insulin": "mean"  # Calculate mean for 'Insulin'      
    }).reset_index()
    print(grouping)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Pregnancies", y="Glucose", data=grouping, color='blue', label='Glucose Mean')
    sns.barplot(x="Pregnancies", y="Insulin", data=grouping, color='orange', label='Insulin Mean')
    plt.title("Mean Glucose and Insulin Levels by Number of Pregnancies (Age < 29)")
    plt.xlabel("Number of Pregnancies")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.show()

def plot5(data):
    ages1 = data[data["Age"]>50]
    grouping1 = ages1.groupby("Pregnancies").agg({
        "Glucose": "mean",  # Calculate mean for 'Glucose'
        "Insulin": "mean"  # Calculate mean for 'Insulin'      
    }).reset_index()
    print(grouping1)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Pregnancies", y="Glucose", data=grouping1, color='green', label='Glucose Mean')
    sns.barplot(x="Pregnancies", y="Insulin", data=grouping1, color='pink', label='Insulin Mean')
    plt.title("Mean Glucose and Insulin Levels by Number of Pregnancies (Age < 29)")
    plt.xlabel("Number of Pregnancies")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.show()

def plot6(data):
    ages2 = data[data["Age"]==22]
    grouping2 = ages2.groupby("Pregnancies").agg({
        "Glucose": "mean",  # Calculate mean for 'Glucose'
        "Insulin": "mean"  # Calculate mean for 'Insulin'      
    }).reset_index()
    print(grouping2)

def corr(data):
    data_numeric = data.drop(columns=["Outcome"])
    corr = data_numeric.corr()
    sns.heatmap(data=corr,
                annot=True,
                fmt=".2g",
                cmap="coolwarm")    
    plt.title("Correlation Matrix")
    plt.show()

def stat(data):
    group1 = data[data["Outcome"]==1]["Insulin"]
    group2= data[data["Outcome"]==0]["Insulin"]

    t_stat, p_value = stats.ttest_ind(group1, group2) #independence 2 different group
    print(f"T-Statistics: {t_stat}")
    print(f"P-Value: {p_value}")
    return t_stat, p_value

def dens(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'], kde=True, color='purple')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frekans")
    plt.show()

def boxplot(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Outcome", y="Glucose", data=data) #glucose levels according to outcomes
    plt.title("Glucose Levels by Outcome")
    plt.xlabel("Outcome (0:Healthy, 1:Diabetic)")
    plt.ylabel("Glucose")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Outcome", y="Insulin", data=data)
    plt.title("Insulin Levels by Outcome")
    plt.xlabel("Outcome (0: Healthy, 1: Diabetic)")
    plt.ylabel("Insulin")
    plt.show()

def normdist(data):
    stat, p_value = stats.shapiro(data['Glucose'].dropna())
    print(f"Shapiro-Wilk Test Result: Test Stats = {stat}, P-değeri = {p_value}") #if p value < 0.05, data doesnt show normal dist.
    if p_value < 0.05:
        print(f"Dataset doesnt appropriate for normal distribution.")
    else:
        print(f"Dataset appropriate for normal distribution.")


def statistical(data):
    covariance = data[['Glucose', 'Insulin']].cov().iloc[0, 1]
    print(f"Covariance: {covariance}")

    correlation = data[['Glucose', 'Insulin']].corr().iloc[0, 1]
    print(f"Pearson Correlation: {correlation}")

    z_scores = zscore(data[['Glucose', 'Insulin']])
    print(f"Z-Scores:\n{z_scores}") 

    outliers = (z_scores > 3) | (z_scores < -3)  # Hem pozitif hem negatif outliers
    outliers_data = data[outliers.any(axis=1)]  # choose the outliers rows from all values
    print("Outliers Values:")
    print(outliers_data)

def stat1(data):
    group1 = data[data["Outcome"]==1]["SkinThickness"]
    group2= data[data["Outcome"]==0]["SkinThickness"]

    t_stat, p_value = stats.ttest_ind(group1, group2) #independence 2 different group
    print(f"T-Statistics: {t_stat}")
    print(f"P-Value: {p_value}")
    return t_stat, p_value

def agegroup(data):
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['19-30', '31-40', '41-50', '51-60', '60+'], right=False)
    f_stat, p_value = stats.f_oneway(
        data[data['Age_Group'] == '19-30']['Glucose'],
        data[data['Age_Group'] == '31-40']['Glucose'],
        data[data['Age_Group'] == '41-50']['Glucose'],
        data[data['Age_Group'] == '51-60']['Glucose'],
        data[data['Age_Group'] == '60+']['Glucose']
    )
    print(f"F-Statistiği: {f_stat}, P-Değeri: {p_value}")

    if p_value<0.05:
        print("Between age groups have been significal differences.")
    else:
        print("Between age groups have not been significal differences")

def plot_dpf_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['DiabetesPedigreeFunction'], kde=True, color='purple')
    plt.title('DiabetesPedigreeFunction Distribution')
    plt.xlabel('DiabetesPedigreeFunction')
    plt.ylabel('Frequency')
    plt.show()

def dpf_statistics(data):
    print("DiabetesPedigreeFunction - Statistical Summary:")
    print(data['DiabetesPedigreeFunction'].describe())
    # Skewness ve Kurtosis
    from scipy.stats import skew, kurtosis
    print(f"Skewness: {skew(data['DiabetesPedigreeFunction']):.3f}")
    print(f"Kurtosis: {kurtosis(data['DiabetesPedigreeFunction']):.3f}")

def plot_dpf_vs_outcome(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Outcome', y='DiabetesPedigreeFunction', data=data)
    plt.title('DiabetesPedigreeFunction vs Outcome')
    plt.xlabel('Outcome (0: Healthy, 1: Diabetic)')
    plt.ylabel('DiabetesPedigreeFunction')
    plt.show()

def dpf_agegroup_anova(data):
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['19-30', '31-40', '41-50', '51-60', '60+'], right=False)

    f_stat, p_value = stats.f_oneway(
        data[data['Age_Group'] == '19-30']['DiabetesPedigreeFunction'],
        data[data['Age_Group'] == '31-40']['DiabetesPedigreeFunction'],
        data[data['Age_Group'] == '41-50']['DiabetesPedigreeFunction'],
        data[data['Age_Group'] == '51-60']['DiabetesPedigreeFunction'],
        data[data['Age_Group'] == '60+']['DiabetesPedigreeFunction']
    )

    print(f"F-Statistiği: {f_stat}, P-Değeri: {p_value}")
    if p_value < 0.05:
        print("There are significant differences between age groups in terms of Diabetes Pedigree Function.")
    else:
        print("There are NO significant differences between age groups in terms of Diabetes Pedigree Function.")

def bmi_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['BMI'], kde=True, color='pink')
    plt.title('BMI Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Frequency')
    plt.show()

def bmi(data):
    conditions = [
        (data['BMI'] < 18.5),
        (data['BMI'] >= 18.5) & (data['BMI'] < 24.9),
        (data['BMI'] >= 25) & (data['BMI'] < 29.9),
        (data['BMI'] >= 30)
    ]
    categories = ['Underweight', 'Normal weight', 'Overweight', 'Obesity']
    data['BMI Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, float('inf')], labels=categories)
    print(data["BMI Category"].value_counts())
    return data

def bmi_outcome_analysis(data):
    bmi_outcome_counts = pd.crosstab(data['BMI Category'], data['Outcome'], margins=True, margins_name="Total")
    
    print("BMI Category and Outcome Dist:")
    print(bmi_outcome_counts)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='BMI Category', hue='Outcome', data=data, palette='Set1')
    plt.title('BMI Category and Outcome Dist.')
    plt.xlabel('BMI Category')
    plt.ylabel('People Count')
    plt.legend(title='Outcome', labels=['Healthy (0)', 'Diabetic (1)'])
    plt.show()

def bmi_outcome_chi2_test(data):
    # Chi-Square Test: BMI Category and Outcome between relationships.
    contingency_table = pd.crosstab(data['BMI Category'], data['Outcome'])
    # Chi-Square Test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-Square Stats: {chi2_stat}")
    print(f"P-Value: {p_value}")
    if p_value < 0.05:
        print("There are significiant relationship between BMI and Outcome.")
    else:
        print("There are no significiant relationship between BMI and Outcome.")


################################################################################################################

plot(data)
plot2(data)
plot3(data)
plot4(data)
plot5(data)
plot6(data)
corr(data)
statcal=stat(data)
print(statcal) # p value<0.5 , the difference between 2 group is significant. 
dens(data)
boxplot(data)
normdist(data)
statistical(data)
stat1(data)
agegroup(data)
plot_dpf_distribution(data)
dpf_statistics(data)
plot_dpf_vs_outcome(data)
dpf_agegroup_anova(data)
bmi_distribution(data)
categorized_data = bmi(data)
print(categorized_data)
bmi_outcome_analysis(data)
bmi_outcome_chi2_test(data) #there are significant relationship between BMI and OUTCOME.