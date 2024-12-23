#Importing the Dependencies
#We take 80 or 90 percent of data as training data and 10 or 20 percent data as test data.
#We do so because the ml model have already seen the training data so for testing it we need to use some new data which we call test data.


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  #Used to standardize all the data.
from sklearn.model_selection import train_test_split  #This will automatically shift our data to train and tests.
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data Collection and Analysis

#PIMA Diabetes Dataset


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('/content/diabetes.csv')


# printing the first 5 rows of the dataset
print(diabetes_dataset.head())


# number of rows and Columns in this dataset
print(diabetes_dataset.shape)


# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()


#0 --> Non-Diabetic

#1 --> Diabetic



diabetes_dataset.groupby('Outcome').mean()



# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


print(X)



print(Y)


#Data Standardization


scaler = StandardScaler()


scaler.fit(X)

StandardScaler(copy=True, with_mean=True, with_std=True)

standardized_data = scaler.transform(X)


print(standardized_data)



X = standardized_data
Y = diabetes_dataset['Outcome']


print(X)
print(Y)


#SPLITTING THE DATA INTO TRAINING DATA & TESTING DATA


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)  #We can give any integer value to random_state but the way data is split varies depending on the integer.


print(X.shape, X_train.shape, X_test.shape)

