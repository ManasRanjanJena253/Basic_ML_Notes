#Imbalanced Dataset:

#A dataset with an unequal class distribution or a dataset where the distribution of data available for a single condition is much more than the data available for other conditions.
#For example a dataset for diabetes have 1000 data for non-diabetic people and only 100 data for diabetic people.


# importing the dependencies
import numpy as np
import pandas as pd


# loading the dataset to pandas DataFrame
credit_card_data = pd.read_csv('../Important Datasets/credit_data.csv')


# first 5 rows of the dataframe
print(credit_card_data.head())


print(credit_card_data.tail()) #Will give last 5 rows


# distribution of the two classes
credit_card_data['Class'].value_counts()


#This is Highly Imbalanced Dataset

#0 --> Legit Transactions

#1 --> Fraudulent Transactions


# separating the legit and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


print(legit.shape)
print(fraud.shape)


#Under-sampling

#Build a sample dataset containing similar distribution of Legit & Fraudulent Transactions

#Number of Fraudulent Transactions --> 492


legit_sample = legit.sample(n=492)


print(legit_sample.shape)

#Concatenate the Two DataFrames


new_dataset = pd.concat([legit_sample, fraud], axis = 0)


new_dataset.head()


new_dataset.tail()



new_dataset['Class'].value_counts()




