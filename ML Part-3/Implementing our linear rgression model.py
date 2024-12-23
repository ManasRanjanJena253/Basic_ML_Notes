#Linear Regression:

#Y = wX + b

#Y --> Dependent Variable

#X --> Independent Variable

#w --> weight

#b --> bias

#Gradient Descent:

#Gradient Descent is an optimization algorithm used for minimizing the loss function in various machine learning algorithms. It is used for updating the parameters of the learning model.

#w = w - α*dw

#b = b - α*db

# In most cases weight will be an array and bias will be an int/float value.

#Importing the Dependencies


# Importing numpy library
import numpy as np

#Linear Regression


class Linear_Regression():

    def __init__( self, learning_rate, no_of_iterations ) :

        self.learning_rate = learning_rate

        self.no_of_iterations = no_of_iterations

    # fit function to train the model

    def fit( self, X, Y ) :

        # no_of_training_examples, no_of_features

        self.m, self.n = X.shape

        # initiating the weight and bias

        self.w = np.zeros( self.n )

        self.b = 0

        self.X = X

        self.Y = Y


        # implementing Gradient Descent for Optimization

        for i in range( self.no_of_iterations ) :

            self.update_weights()



    # function to update weights in gradient descent

    def update_weights( self ) :

        Y_prediction = self.predict( self.X )

        # calculate gradients

        dw = - ( 2 * ( self.X.T ).dot( self.Y - Y_prediction )  ) / self.m

        db = - 2 * np.sum( self.Y - Y_prediction ) / self.m

        # updating the weights

        self.w = self.w - self.learning_rate * dw

        self.b = self.b - self.learning_rate * db


    # Line function for prediction:

    def predict( self, X ) :

        return X.dot( self.w ) + self.b   # Performs similar operation as wx+b.


#Using Linear Regression model for Prediction


# importing the dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Data Pre-Processing


# loading the data from csv file to a pandas dataframe

salary_data = pd.read_csv('../Important Datasets/salary_data.csv')


# printing the first 5 columns of the dataframe
print(salary_data.head())



# last 5 rows of the dataframe
print(salary_data.tail())


# number of rows & columns in the dataframe
print(salary_data.shape)


# checking for missing values
salary_data.isnull().sum()


#Splitting the feature & target


X = salary_data.iloc[:,:-1].values
Y = salary_data.iloc[:,1].values


print(X)


print(Y)


#Splitting the dataset into training & test data


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 2)

#Training the Linear Regression model


model = Linear_Regression(0.02, 1000)  # The learning rate and the no. of iteration values are experimental, and we have to personally check which value works best for our model.


model.fit(X_train, Y_train)


# printing the parameter values ( weights & bias)

print('weight = ', model.w[0])
print('bias = ', model.b)


#Predict the salary value for test data


test_data_prediction = model.predict(X_test)


print(test_data_prediction)


#Visualizing the predicted values & actual Values


plt.scatter(X_test, Y_test, color = 'red')  # Plotting the test data points.
plt.plot(X_test, test_data_prediction, color='blue')  # Plotting the predicted values.
plt.xlabel(' Work Experience')
plt.ylabel('Salary')
plt.title(' Salary vs Experience')
plt.show()

