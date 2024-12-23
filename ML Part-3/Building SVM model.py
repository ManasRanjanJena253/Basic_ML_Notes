# importing numpy library
import numpy as np

#Support Vector Machine Classifier


class SVM_classifier():


    # initiating the hyperparameters
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter



    # fitting the dataset to SVM Classifier
    def fit(self, X, Y):

        # m  --> number of Data points --> number of rows
        # n  --> number of input features --> number of columns
        self.m, self.n = X.shape

        # initiating the weight value and bias value

        self.w = np.zeros(self.n)

        self.b = 0

        self.X = X

        self.Y = Y

        # implementing Gradient Descent algorithm for Optimization

        for i in range(self.no_of_iterations):
            self.update_weights()



    # function for updating the weight and bias value
    def update_weights(self):

        # label encoding
        y_label = np.where(self.Y <= 0, -1, 1)  # This function will replace any value in the Y column having values less than 0 with -1 or else any other values with 1.



        # gradients ( dw, db)
        for index, x_i in enumerate(self.X):

            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1  # Similar as the formula Y(wx + b)

            if (condition == True):

                dw = 2 * self.lambda_parameter * self.w   # Similar as the formula 2λw.
                db = 0

            else:

                dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])  # Similar as the formula 2λw - YX.
                db = y_label[index]


            self.w = self.w - self.learning_rate * dw

            self.b = self.b - self.learning_rate * db



    # predict the label for a given input value
    def predict(self, X):

        output = np.dot(X, self.w) - self.b

        predicted_labels = np.sign(output)

        y_hat = np.where(predicted_labels <= -1, 0, 1)

        return y_hat

