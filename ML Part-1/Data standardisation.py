#Data Standardization:

#The process of standardizing the data to a common format and common range.
#We need to standardize the data before feeding it to the machine learning model.
#It is usually used to handle outliers.
#Standardization does not affect the data but changes the range.


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# loading the dataset
dataset = sklearn.datasets.load_breast_cancer()


print(dataset)


# loading the data to a pandas dataframe
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)


print(df.head())


print(df.shape)


X = df
Y = dataset.target


print(X)


print(Y)

#Splitting the data into training data and test data
#Generally we take 10(0.1) or 20(0.2) percentage of data as test data.
#X_train and Y_train contains data for training.
#We need to standardize the data before splitting it as after splitting it may lose some of its data.


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


print(X.shape, X_train.shape, X_test.shape)

#Standardize the data


print(dataset.data.std())  #This code will give standard deviation of the data. If all the data lies in the same range the standard deviation will be 1 or close to 1.


scaler = StandardScaler()  #This will give the standardized data.


scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

X_train_standardized = scaler.transform(X_train) #This will change the data into the standardized data.


print(X_train_standardized)



X_test_standardized = scaler.transform(X_test)


print(X_train_standardized.std())


print(X_test_standardized.std())

