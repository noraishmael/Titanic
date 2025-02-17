## import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
## import dataset
df= pd.read_csv("C:/Users/nora_/OneDrive/NORA/DATASCIENCE BOOTCAMP/train.csv")
## visualise five first lines of the ED with header**
df.head(5)
#check the type of values
df.info()
#check if there are values missing
df.isnull().sum()
##list the descriptive stats for each of the columns
df.describe()
## change data for Cabin to numerical value

## first convert all the values to strings
df['Cabin'] = df['Cabin'].astype(str)

## then Map the beginning letter to a number
mapping = {'A': 1, 'B': 2, 'C': 3}  

## get first letter and convert
df['Cabin'] = df['Cabin'].str[0].map(mapping)

## replace missing values (NaN) with 0
df['Cabin'] = df['Cabin'].fillna(0)

## as a last step convert everything into integers
df['Cabin'] = df['Cabin'].astype(int)

## show the outliers through boxplot so we can determine statistical method to use
df.boxplot(figsize=(5,5))
plt.tight_layout

## do the same but with a histogram
df.hist(bins=10, figsize=(5,5))
plt.show()

## fill the missing values in age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

## fill the missing values in cabin with median
df['Cabin'] = df['Cabin'].fillna(df['Cabin'].median())

## print dataset after inputting missing values
df.isnull().sum()
## convert also gender to numeric 
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})  

##convert ports to numbers
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  

## replace missing values (NaN) with 0
df['Embarked'] = df['Embarked'].fillna(0)

## drop columns that do not have numeric data
df = df.drop(columns=['Name','Ticket']) 

## look into correlations i.e. which factors are most likely associated with survival
correlation_matrix = df.corr()

# Plot heatmap of correlation
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# values close to +1 → Strong positive correlation 
# Values close to -1 → Strong negative correlation 
# Values close to 0 → No strong relationship

# show a scatter plot to indicate correlation between age, fare and survival

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Age'], y=df['Fare'], hue=df['Survived'], palette=['red', 'green'], alpha=0.6)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Fare vs. Age (Survival Colored)")
plt.show()

# show survived passengers by gender in a pie chart
survival_by_gender = df.groupby('Sex')['Survived'].sum()

# Create pie chart
plt.figure(figsize=(6,6))
plt.pie(survival_by_gender, labels=['Female', 'Male'], autopct='%1.1f%%', colors=['pink', 'blue'], startangle=90)
plt.title("Survival Rate by Gender")
plt.show()

# for the logistic regression we need to look at the variables 
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]  
y = df['Survived'] 

# then we need to split the data into test data and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)
# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
