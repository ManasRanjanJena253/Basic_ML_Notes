#We cannot feed our machine learning model with datasets having missing values .

#Methods to Handle Missing Values:

#Imputation (It means replacing the null values with mean , median or mode values.)
#Dropping (It means deleting the all the rows that have null values.(this method is not preferred.))

#Importing the libraries


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# loading the dataset to a Pandas DataFrame
dataset = pd.read_csv('/content/Placement_Dataset.csv')


print(dataset.head())


print(dataset.shape)


dataset.isnull().sum()


#Central Tendencies:

#Mean
#Median
#Mode
#These three are called as central tendencies.
#When there are skew distribution(outliers) we use median and mode to fill up the null values.

# analyse the distribution of data in the salary
fig, ax = plt.subplots(figsize=(8,8))
sns.displot(dataset.salary)
plt.show()

#Replace the missing values with Median value


dataset['salary'].fillna(dataset['salary'].median(),inplace=True)
#fillna() function will fill all the null values with the given value.


dataset.isnull().sum()



# filling missing values with Mean value:
dataset['salary'].fillna(dataset['salary'].mean(),inplace=True)


# filling missing values with Mean value:
dataset['salary'].fillna(dataset['salary'].mode(),inplace=True)

#Dropping Method


salary_dataset = pd.read_csv('/content/Placement_Dataset.csv')


print(salary_dataset.shape)


print(salary_dataset.isnull().sum())

# drop the missing values
salary_dataset = salary_dataset.dropna(how='any')


print(salary_dataset.isnull().sum())



print(salary_dataset.shape)

