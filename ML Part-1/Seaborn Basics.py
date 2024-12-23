#Seaborn:

#Data Visualization Library
#Importing the Libraries


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Note : Seaborn has some built-in datasets


# total bill vs tip dataset
tips = sns.load_dataset('tips')


print(tips.head())


# setting a theme for the plots
sns.set_theme()


# visualize the data
sns.relplot(data=tips, x ='total_bill',y='tip',col='time',hue='smoker',style='smoker',size='size')
plt.show() #We need to use this function from matplotlib to show graphs made from seaborn library .

# load the iris dataset
iris = sns.load_dataset('iris')


iris.head()

#Scatter Plot


sns.scatterplot(x='sepal_length',y='petal_length',hue='species',data=iris)
plt.show()


sns.scatterplot(x='sepal_length',y='petal_width',hue='species',data=iris)
plt.show()


# loading the titanic dataset
titanic = sns.load_dataset('titanic')


print(titanic.head())



print(titanic.shape)


#Count Plot


sns.countplot(x='class',data=titanic) #Plots the number of people classified by how many peoples in each class column as given in titanic dataset.
plt.show()


sns.countplot(x='survived',data=titanic) #Plots the number of people classified by how many peoples in each survived column as given in titanic dataset.
plt.show()

#Bar Chart


sns.barplot(x='sex',y='survived',hue='class',data=titanic)
plt.show()

# house price dataset
from sklearn.datasets import fetch_california_housing
house_california = fetch_california_housing()

house = pd.DataFrame(house_california.data, columns=house_california.feature_names)
house['PRICE'] = house_california.target


print(house_california)


house.head()

#Distribution Plot


sns.displot(house['PRICE'])
plt.show()

#Correlation:

#Positive Correlation
#Negative Correlation
#Heat Map


correlation = house.corr()


# constructing a Heat Map
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt=".if", annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()



