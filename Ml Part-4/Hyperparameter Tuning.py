# importing the dependencies
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV  # Importing the GridSearchCV method used for hyperparameter tuning.
from sklearn.model_selection import RandomizedSearchCV # Importing the RandomizedSearchCv method used for hyperparameter tuning.

#We will be working on the breast cancer dataset


# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


print(breast_cancer_dataset)

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)


# print the first 5 rows of the dataframe
print(data_frame.head())

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target


# print the first 5 rows of the dataframe
data_frame.head()

# number of rows and Columns in this dataset
print(data_frame.shape)


# checking for missing values
print(data_frame.isnull().sum())


# checking the distribution of Target Variable
data_frame['label'].value_counts()


#1 --> Benign

#0 --> Malignant

#Separating the features and target


X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']


print(X)


print(Y)


X = np.asarray(X)  # Converting the features of our dataframe into a numpy array.
Y = np.asarray(Y)  # Converting the labels of our dataframe into a numpy array.

#GridSearchCV is used for determining the best parameters for our model


# loading the SVC model
model = SVC()


# hyperparameters

parameters = {
    'kernel':['linear','poly','rbf','sigmoid'],
    'C':[1, 5, 10, 20]
}   # Creating a dictionary containing the name of parameter as key and their types as their values.


# grid search
classifier = GridSearchCV(model, parameters, cv=5)
# Meaning of each parameter passed into the grid search ::
# 1. model :: The ml model we want to find the best parameters for.
# 2. parameters :: A dictionary containing all the important hyperparameters we are wanting to optimise.
# 3. cv :: Total no. of folds of data for cross validation.

# fitting the data to our model
classifier.fit(X, Y)


print(classifier.cv_results_)  # Function to see all the combination of parameters made by the grid search .


# best parameters

best_parameters = classifier.best_params_   # Function to find the best parameters which the grid search found.
print(best_parameters)


# highest accuracy

highest_accuracy = classifier.best_score_   # This function will give the highest accuracy provided by the best parameter combination.
print(highest_accuracy)


# loading the results of all different parameter combination found through grid search function to pandas dataframe.
result = pd.DataFrame(classifier.cv_results_)


print(result.head())

grid_search_result = result[['param_C','param_kernel','mean_test_score']]



#RandomizedSearchCV  (All the steps similar to grid search )

# loading the SVC model
model = SVC()


# hyperparameters

parameters = {
    'kernel':['linear','poly','rbf','sigmoid'],
    'C':[1, 5, 10, 20]
}


# grid search
classifier = RandomizedSearchCV(model, parameters, cv=5)


# fitting the data to our model
classifier.fit(X, Y)


# best parameters

best_parameters = classifier.best_params_
print(best_parameters)


# highest accuracy

highest_accuracy = classifier.best_score_
print(highest_accuracy)


# loading the results to pandas dataframe
result = pd.DataFrame(classifier.cv_results_)


print(result.head())

randomized_search_result = result[['param_C','param_kernel','mean_test_score']]





