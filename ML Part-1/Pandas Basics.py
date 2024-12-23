#Pandas Library:

#Useful for Data Processing & Analysis

#Pandas Data Frame:

#Pandas DataFrame is two-dimensional tabular data structure with labeled axes (rows and columns).


# importing the pandas library
import pandas as pd
import numpy as np

#Creaating a Pandas DataFrame


# importing the boston house price data
from sklearn.datasets import fetch_california_housing


california_dataset = fetch_california_housing()


type(california_dataset)
#Output of the above code ::

#sklearn.utils.Bunch

print(california_dataset)


# pandas DataFrame
california_df = pd.DataFrame(california_dataset.data, columns = california_dataset.feature_names)
# In the above code we are converting the california .

california_df.head()

print(california_df.shape)

type(california_df)

#Output of the above code ::

#pandas.core.frame.DataFrame


#Importing the data from a CSV file to a pandas DataFrame


# csv file to pandas df
#diabetes_df = pd.read_csv('/content/diabetes.csv')


#type(diabetes_df)
#Output of the above code ::

#pandas.core.frame.DataFrame

#diabetes_df.head()


#print(diabetes_df.shape)

#Loading the data from a excel file to a Pandas DataFrame:

# To read an Excel file :: pd.read_excel('file path')

#Exporting a DataFrame to a csv file


california_df.to_csv('boston.csv')

#Exporting the Pandas DataFrame to an Excel File:

#boston_df.to_excel('file path')


# creating a DatFrame with random values
random_df = pd.DataFrame(np.random.rand(20,10))  #20 and 10 are the dimensions of the dataframe created.


random_df.head()  #Gives output of first five rows present in the dataframe.


print(random_df.shape)  #Finding the dimensions of the dataframe.

#Inspecting a DataFrame


#finding the number of rows & columns
print(california_df.shape)


# first 5 rows in a DataFrame
california_df.head()

# last 5 rows of the DataFrame
california_df.tail()


# information about the DataFrame
california_df.info()


# finding the number of missing values
california_df.isnull().sum()



# diabetes dataframe
#diabetes_df.head()


# counting the values based on the labels
#diabetes_df.value_counts('Outcome')  #Gives the number of values for each unique values in outcome column .


# group the values based on the mean
#diabetes_df.groupby('Outcome').mean() #Mean of unique values of column 'outcome'.



# count or number of values
california_df.count()


# mean value - column wise
california_df.mean()  # Mean value for each column .


# standard deviation - column wise
california_df.std()



# minimum value
california_df.min()



# maximum value
california_df.max()


# all the statistical measures about the dataframe
california_df.describe()


#Manipulating a DataFrame


# adding a column to a dataframe
california_df['Price'] = california_dataset.target


california_df.head()

# removing a row
california_df.drop(index=0, axis=0)




# drop a column
california_df.drop(columns='ZN', axis=1)



# locating a particular column
print(california_df.iloc[:,0])  # first column
print(california_df.iloc[:,1])  # second column
print(california_df.iloc[:,2])  # third column
print(california_df.iloc[:,-1]) # last column

#Correlation are of two types ::
#1. Positive correlation(Directly proportional)
#2. Negative correlation(Indirectly proportional)

california_df.corr()  #Used to find the correlation between each column with other columns.


