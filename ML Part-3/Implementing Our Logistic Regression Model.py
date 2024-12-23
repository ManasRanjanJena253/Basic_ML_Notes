#Importing the Dependencies


# importing numpy library
import numpy as np

#Logistic Regression


class Logistic_Regression():


    # declaring learning rate & number of iterations (Hyper parameters)
    def __init__(self, learning_rate, no_of_iterations):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations



    # fit function to train the model with dataset
    def fit(self, X, Y):

        # number of data points in the dataset (number of rows)  -->  m
        # number of input features in the dataset (number of columns)  --> n
        self.m, self.n = X.shape


        #initiating weight & bias value

        self.w = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y


        # implementing Gradient Descent for Optimization

        for i in range(self.no_of_iterations):
            self.update_weights()



    def update_weights(self):

        # Y_hat formula (sigmoid function)

        Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b ) ))


        # derivatives

        dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))

        db = (1/self.m)*np.sum(Y_hat - self.Y)


        # updating the weights & bias using gradient descent

        self.w = self.w - self.learning_rate * dw

        self.b = self.b - self.learning_rate * db


    # Sigmoid Equation & Decision Boundary

    def predict(self, X):

        Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) ))
        Y_pred = np.where( Y_pred > 0.5, 1, 0)
        return Y_pred

#Importing the Dependencies


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Data Collection and Analysis

#PIMA Diabetes Dataset


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('../Important Datasets/diabetes.csv')


# printing the first 5 rows of the dataset
diabetes_dataset.head()


# number of rows and Columns in this dataset
print(diabetes_dataset.shape)


# getting the statistical measures of the data
diabetes_dataset.describe()


diabetes_dataset['Outcome'].value_counts()


#0 --> Non-Diabetic

#1 --> Diabetic


diabetes_dataset.groupby('Outcome').mean()


# separating the data and labels
features = diabetes_dataset.drop(columns = 'Outcome', axis=1)
target = diabetes_dataset['Outcome']


print(features)


print(target)


#Data Standardization


scaler = StandardScaler()


scaler.fit(features)

StandardScaler(copy=True, with_mean=True, with_std=True)

standardized_data = scaler.transform(features)


print(standardized_data)

features = standardized_data
target = diabetes_dataset['Outcome']


print(features)
print(target)

#Train Test Split


X_train, X_test, Y_train, Y_test = train_test_split(features,target, test_size = 0.2, random_state=2)


print(features.shape, X_train.shape, X_test.shape)

#Training the Model


classifier = Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

#Model Evaluation

#Accuracy Score


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score( Y_train, X_train_prediction)


print('Accuracy score of the training data : ', training_data_accuracy)


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score( Y_test, X_test_prediction)


print('Accuracy score of the test data : ', test_data_accuracy)

#Making a Predictive System


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')



