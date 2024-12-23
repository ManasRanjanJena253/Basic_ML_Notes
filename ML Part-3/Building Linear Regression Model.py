
# importing numpy library
import numpy as np

#Linear Regression Model ::


class Linear_Regression():

    # initiating the parameters (learning rate & no. of iterations)
    # Learning rate determines the step size of the change in parameters in each iteration.
    def __init__(self, learning_rate, no_of_iterations):  # The self parameter is provided as most models are called using other variables and those variables then take up the place of the self.
        # __init__ is an initiation function.
        # For example lets take the variable with which our model is lr.

        self.learning_rate = learning_rate  # This represents the calling of the model as lr.learning_rate.
        self.no_of_iterations = no_of_iterations


    def fit(self, X, Y ):
         # fit function is used to train the ML model.

        # number of training examples & number of features means the number of data points we are going to use.

        self.m, self.n = X.shape  # number of rows & columns
         # In the above function the self.m will store the no. of rows of the training data and self.n will store the no. of columns of the training data.

        # initiating the weight and bias

        self.w = np.zeros(self.n)  # Assigning weight/slope as array of no. of elements = no. of columns with all values = 0.
        self.b = 0     # Initially setting the bias/intercept set as 0.
        self.X = X
        self.Y = Y

        # implementing Gradient Descent

        for i in range(self.no_of_iterations):
            self.update_weights()


    def update_weights(self):
        # This function is being used to update the parameters.

        Y_prediction = self.predict(self.X)

        # calculate gradients

        dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m

        db = - 2 * np.sum(self.Y - Y_prediction)/self.m

        # updating the weights

        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db


    def predict(self, X):

        return X.dot(self.w) + self.b
