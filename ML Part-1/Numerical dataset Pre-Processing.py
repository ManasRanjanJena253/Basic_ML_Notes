#Importing the Dependencies


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Data Collection & Pre-Processing


# loading the data from csv file to a pandas dataframe
diabetes_data = pd.read_csv('../Important Datasets/diabetes.csv')


# first 5 rows of the dataframe
print(diabetes_data.head())



# number of rows & columns
print(diabetes_data.shape)


diabetes_data.describe() #Will give the statistical measures of the data.

#Separating Features and Target

X = diabetes_data.drop(columns='Outcome', axis =1)  #Will store all the data expect the outcome column.

Y = diabetes_data['Outcome'] #Will store all the data present in outcome column.


print(X)


print(Y)

#0 --> Non - Diabetic

#1 --> Diabetic

#Data Standardization ::
#We standardize the data before splitting it as after splitting it may lose some of its data.


scaler = StandardScaler()


standardized_data = scaler.fit_transform(X)


print(standardized_data)



X = standardized_data


print(X)

print(Y)


#Splitting the dataset into Training data & Testing Data


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


print(X.shape, X_train.shape, X_test.shape)




