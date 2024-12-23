#Importing the Dependencies


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Data Collection and Processing


# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('../Important Datasets/heart_disease_data.csv')


# print first 5 rows of the dataset
print(heart_data.head())


# print last 5 rows of the dataset
print(heart_data.tail())


# number of rows and columns in the dataset
print(heart_data.shape)

# checking for missing values
print(heart_data.isnull().sum())


# statistical measures about the data
print(heart_data.describe())

# checking the distribution of Target Variable
print(heart_data['target'].value_counts())


#1 --> Defective Heart

#0 --> Healthy Heart

#Splitting the Features and Target


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


print(X)


print(Y)

#Splitting the Data into Training data & Test Data


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


print(X.shape, X_train.shape, X_test.shape)

#Model Training


model = LogisticRegression(max_iter=1000)


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

LogisticRegression(max_iter=1000)
#Model Evaluation

#Accuracy Score


from sklearn.metrics import accuracy_score


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print(training_data_accuracy)


print('Accuracy on Training data : ', round(training_data_accuracy*100, 2), '%')

#Accuracy on Training data :  85.54 %

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print(test_data_accuracy)


print('Accuracy on Test data : ', round(test_data_accuracy*100, 2), '%')

#Accuracy on Test data :  80.33 %
#Confuaion Matrix


from sklearn.metrics import confusion_matrix


cf_matrix = confusion_matrix(Y_test, X_test_prediction)

print(cf_matrix)


tn, fp, fn, tp = cf_matrix.ravel()  # This function used to find all the values of tn,fp,tp,tn.
# tn = true negative
# fp = false positive
# fn = false negative
# tp = true positive

print(tn, fp, fn, tp)


import seaborn as sns
sns.heatmap(cf_matrix, annot=True)  # The annot = true means that we want the labels on our heatmap.
plt.show()
