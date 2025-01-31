import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from fcmeans import FCM
import skfuzzy as fuzz

#import original dataset 
adults = pd.read_csv("~/datasets/adult.csv")
print(adults.head())
print(adults.describe())
print(adults.info())

col = adults.select_dtypes(include=['object']).columns.tolist()
for i in col:
    adults[i] = adults[i].str.replace('?', 'Unknown')


#average income by countries
def details(adults):
    adults.head()
    adults.describe().T
    adults.info()
    print(adults["sex"].value_counts())
    print(adults["native.country"].value_counts())
    print(adults["income"].value_counts())
    print(adults["education"].value_counts())
    print(adults["workclass"].value_counts())
    print(adults["relationship"].value_counts())
    return adults

def detect_outliers_iqr(adults):
    df_numerical = adults.select_dtypes(include=[np.number])
    Q1 = df_numerical.quantile(0.25)
    Q3 = df_numerical.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numerical < (Q1 - 1.5 * IQR)) | (df_numerical > (Q3 + 1.5 * IQR))).sum()
    return outliers


def hist(adults):
    sns.histplot(data=adults,
                 x="age",
                 kde=True)
    plt.title("Dist. of Age")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()
    sns.histplot(data=adults,
                 x="education",
                 kde=True)
    plt.title("Dist. of Education")
    plt.xlabel("Education")
    plt.xticks(rotation=45)
    plt.ylabel("Frequency")
    plt.show()
    sns.histplot(data=adults,
                 x="workclass",
                 kde=True)
    plt.title("Dist. of Workclass")
    plt.xticks(rotation=45)
    plt.xlabel("Workclass")
    plt.ylabel("Frequency")
    plt.show()
    sns.histplot(data=adults,
                 x="marital.status",
                 kde=True)
    plt.title("Dist. of Marital Status")
    plt.xticks(rotation=45)
    plt.xlabel("Marital Status")
    plt.ylabel("Frequency")
    plt.show()
    sns.histplot(data=adults,
                 x="relationship",
                 kde=True,
                 color="pink")
    plt.title("Dist. of Relationship")
    plt.xticks(rotation=90)
    plt.xlabel("Relationship")
    plt.ylabel("Frequency")
    plt.show()
    sns.histplot(data=adults,
                 x="race",
                 kde=True,
                 color="green")
    plt.title("Dist. of Race")
    plt.xticks(rotation=90)
    plt.xlabel("Race")
    plt.ylabel("Frequency")
    plt.show()
    sns.histplot(data=adults,
                 x="hours.per.week",
                 kde=True,
                 color="purple",
                 bins=30)
    plt.title("Dist. of Hours per Week")
    plt.xlabel("Hour per week")
    plt.ylabel("Frequency")
    plt.show()    

def barplotting(adults):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='income', data=adults, palette='pastel')
    plt.title('Distribution of Income')
    plt.xlabel('Income')
    plt.ylabel('Count')
    plt.show()
    ####box plotting###
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='income', y='hours.per.week',  palette='coolwarm', data=adults)
    plt.title('Hours Per Week by Income')
    plt.xlabel('Income')
    plt.ylabel('Hours Per Week')
    plt.show()
    #####count plot education by income #####
    sns.countplot(y='education', hue='income', data=adults, order=adults['education'].value_counts().index)
    plt.title('Education Level by Income')
    plt.xlabel('Count')
    plt.ylabel('Education Level')
    plt.legend(title='Income')
    plt.show()
    #########################
    sns.countplot(data=adults,
                  y="occupation",
                  palette="muted",
                  order=adults['occupation'].value_counts().index)
    plt.title('Occupation Distribution')
    plt.title('Count')
    plt.show()
    #########################
    sns.countplot(data=adults,
                  y="workclass",
                  palette="muted",
                  order=adults['workclass'].value_counts().index)
    plt.title('Workclass Distribution')
    plt.title('Count')
    plt.show()
    sns.countplot(y='education', hue='income', data=adults, palette='bright', 
                  order=adults['education'].value_counts().index)
    plt.title('Education Level vs. Income')
    plt.xlabel('Count')
    plt.ylabel('Education Level')
    plt.show()
    sns.countplot(y='education', hue='sex', data=adults, palette='bright', 
                  order=adults['education'].value_counts().index)
    plt.title('Education Level by Sex')
    plt.xlabel('Count')
    plt.ylabel('Education Level')
    plt.show()
    sns.countplot(x='education', hue='race', data=adults, palette='bright', 
                  order=adults['education'].value_counts().index)
    plt.xticks(rotation=90)
    plt.title('Education Level by Race')
    plt.xlabel('Count')
    plt.ylabel('Education Level')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.countplot(x='workclass', data=adults, hue="race", 
                  palette='dark', order=adults['workclass'].value_counts().index)
    plt.title('Workclass Distribution')
    plt.xlabel('Count')
    plt.ylabel('Workclass')
    plt.show()

def corrmatrix(adults):
    adults_numeric = adults.select_dtypes(include=['number'])
# Correlation heatmap for numeric variables
    corr = adults_numeric.corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(data=corr, annot=True, cmap='coolwarm',
                fmt=".2g")
    plt.title('Correlation Heatmap')
    plt.show()

## i just decided randomly to examine Japan
def analyze_japan(adults):
    japan_data = adults[adults["native.country"] == "Japan"]
    print(f"Data counts that has for Japan: {len(japan_data)}")
    print(japan_data.head())  # a few rows

    #income distribution
    income_distribution = japan_data['income'].value_counts()
    print(f"Income distribution in Japan: \n{income_distribution}")

    #age and hours of work relationship
    japan_age_hours = japan_data[['age', 'hours.per.week']].describe()
    print(f"Age and Work for Week relationship in  Japan: \n{japan_age_hours}")

    #education and income ratio
    japan_education_income = japan_data.groupby('education')['income'].value_counts().reset_index()
    print(f"Education and Income relationsjip in Japan: \n{japan_education_income}")

    #gender and income rel.
    gender_income = japan_data.groupby("sex")["income"].value_counts().unstack()
    print(f"Gender and Income rel. in Japan: \n{gender_income}")

    plt.figure(figsize=(10,6))
    sns.boxplot(data=japan_data, x='sex', y='hours.per.week')
    plt.title("Gender ve Hours of Work Relationship (Japonya)")
    plt.show()

## i just decided randomly to examine USA
def analyze_usa(adults):
    usa_data = adults[adults["native.country"] == "United-States"]
    print(f"Data counts that has for USA: {len(usa_data)}")
    print(usa_data.head())  # a few rows

    #income distribution
    income_distribution = usa_data['income'].value_counts()
    print(f"Income distribution in USA: \n{income_distribution}")

    #age and hours of work relationship
    usa_age_hours = usa_data[['age', 'hours.per.week']].describe()
    print(f"Age and Work for Week relationship in  USA: \n{usa_age_hours}")

    #education and income ratio
    usa_education_income = usa_data.groupby('education')['income'].value_counts().reset_index()
    print(f"Education and Income relationsjip in USA: \n{usa_education_income}")

    #gender and income rel.
    gender_income = usa_data.groupby("sex")["income"].value_counts().unstack()
    print(f"Gender and Income rel. in USA: \n{gender_income}")

#details in Iran
def analyze_iran(adults):
    iran_data = adults[adults["native.country"] == "Iran"]
    print(f"Data counts that has for Iran: {len(iran_data)}")
    print(iran_data.head())  # a few rows

    #income distribution
    income_distribution = iran_data['income'].value_counts()
    print(f"Income distribution in Iran: \n{income_distribution}")

    #gender and income rel.
    gender_income = iran_data.groupby(["marital.status","sex"])["income"].value_counts().unstack()
    print(f"Gender and Income rel. in Iran: \n{gender_income}") #i just wanted to look how many women work in Iran##

def analyze_age_hours(adults):
    age_hours = adults.groupby('age')['hours.per.week'].mean()
    print(f"Avg Hours of Week by Age:\n{age_hours}")
    age_hours.plot(kind='line', marker='o')
    plt.title("Avg Hours of Week by Age")
    plt.xlabel('Age')
    plt.ylabel('Avg Working Hours')
    plt.grid(True)
    plt.show()

def analyze_age_income(adults):
    bins = [18, 30, 40, 50, 60, 100]
    labels = ['18-29', '30-39', '40-49', '50-59', '60+']
    adults['age_group'] = pd.cut(adults['age'], bins=bins, labels=labels)   
    age_income = adults.groupby('age_group')['income'].value_counts().unstack().fillna(0)  
    print(f"Income Dist. by Age Group:\n{age_income}") 
    age_income.plot(kind='bar', stacked=True, figsize=(12,6))
    plt.title('Income Dist. by Age Group')
    plt.ylabel('Count')
    plt.xlabel('Age Group')
    plt.show()

########################################################################

details(adults)
outliers = detect_outliers_iqr(adults)
print(outliers)
hist(adults)
barplotting(adults)
corrmatrix(adults)
analyze_japan(adults)
analyze_usa(adults)
analyze_iran(adults)
analyze_age_hours(adults)
analyze_age_income(adults)
