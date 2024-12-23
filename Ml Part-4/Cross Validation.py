#Importing the Dependencies


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score   # This function used for cross validation of models.
from sklearn.metrics import accuracy_score

#Importing the models


from sklearn.linear_model import LogisticRegression  # Importing the logistic regression ml model.
from sklearn.svm import SVC  # Importing support vector classifier ml model.
from sklearn.neighbors import KNeighborsClassifier  # Importing the KNeighbours classifier ml model.
from sklearn.ensemble import RandomForestClassifier    # Importing RandomForestClassifier  ml model.

#Data Collection and Processing


# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('../Important Datasets/heart_disease_data.csv')


# print first 5 rows of the dataset
# We shouldn't standardise the categorical data types i.e. the datapoints having value either 1 or 0.
print(heart_data.head())


# number of rows and columns in the dataset
print(heart_data.shape)


# checking for missing values
print(heart_data.isnull().sum())

# checking the distribution of Target Variable
print(heart_data['target'].value_counts())


#1 --> Defective Heart

#0 --> Healthy Heart

#Splitting the Features and Target


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


print(X)


print(Y)


#Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=3)


print(X.shape, X_train.shape, X_test.shape)

#Comparing the performance of the models


# list of models
models = [LogisticRegression(max_iter=1000), SVC(kernel='linear'), KNeighborsClassifier(), RandomForestClassifier()]


def compare_models_train_test():

    for model in models:

        # training the model
        model.fit(X_train, Y_train)

        # evaluating the model
        test_data_prediction = model.predict(X_test)

        accuracy = accuracy_score(Y_test, test_data_prediction)

        print('Accuracy score of the ', model, ' = ', accuracy)




print(compare_models_train_test())


#Cross Validation
#Logistic Regression


cv_score_lr = cross_val_score(LogisticRegression(max_iter=1000), X, Y, cv=5)

print(cv_score_lr)

mean_accuracy_lr = sum(cv_score_lr)/len(cv_score_lr)

mean_accuracy_lr = mean_accuracy_lr*100

mean_accuracy_lr = round(mean_accuracy_lr, 2)

print(mean_accuracy_lr)

#Support Vector Classifier


cv_score_svc = cross_val_score(SVC(kernel='linear'), X, Y, cv=5)

print(cv_score_svc)

mean_accuracy_svc = sum(cv_score_svc)/len(cv_score_svc)

mean_accuracy_svc = mean_accuracy_svc*100

mean_accuracy_svc = round(mean_accuracy_svc, 2)

print(mean_accuracy_svc)

#Creating a Function to compare the models


# list of models
models = [LogisticRegression(max_iter=10000), SVC(kernel='linear'), KNeighborsClassifier(), RandomForestClassifier()]

x = []
def compare_models_cross_validation():

    for model in models:


        cv_score = cross_val_score(model, X,Y, cv=5)  # This function will perform cross validation on model by dividing the data into equal no. of times as specified in the cv parameter of the function and find accuracy each no. of times and store each accuracy in the form of a list.

        mean_accuracy = sum(cv_score)/len(cv_score)  # Getting the sum of all the accuracies present in the list created by the above function.

        mean_accuracy = mean_accuracy*100

        mean_accuracy = round(mean_accuracy, 2)  # This function will round off the accuracy scores upto two decimals.
        x.append(mean_accuracy)
        print('Cross Validation accuracies for ', model, '=  ', cv_score)
        print('Accuracy % of the ', model, mean_accuracy)
        print('----------------------------------------------')
        print(x)
    return x
a = compare_models_cross_validation()
a = np.array(a)

# Plotting the different mean accuracies with their corresponding graphs.

x = np.array(x)
y = np.array(['LogisticRegression', 'SVC', 'KNeighborsClassifier', 'RandomForestClassifier'])
colours = np.array(['yellow','blue','red','orange'])
plt.barh(y,x,color = colours )
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()





