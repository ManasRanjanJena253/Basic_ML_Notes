#Importing the Dependencies


import numpy as np
import pandas as pd
import re  #re is regular expression library. It is useful for scanning and going through some text in the document.
import nltk  #nltk is natural language tool kit.
from nltk.corpus import stopwords #Corpus means anything containing any sort of text , example : essays,paragraph.
from nltk.stem.porter import PorterStemmer #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


nltk.download('stopwords')  #This statement will download the stopwords.
#stopwords are repetitive words which occur in a paragraph but these words don't convey much meaning.


# printing the stopwords
print(stopwords.words('english')) #Stopewords of english language.

#Data Pre-Processing


# load the data to a pandas dataframe
news_data = pd.read_csv('../Important Datasets/fake_news_dataset.csv')


# first 5 rows of the dataset
news_data.head()

#0 --> Real News

#1 --> Fake News


print(news_data.shape)
#The more the data the better performance of your ml model.

# checking for missing values
news_data.isnull().sum()



# replacing the missing values with null string because the initial values in those rows are strings.
# If the initial values of the rows would have been numerical we would have replaced the null values with mean,median or mode.
news_data = news_data.fillna('')


# merging th author name and news title
news_data['content'] = news_data['author']+' '+news_data['title'] #Creating a new column called " Content ".

# We are merging only the author and title column for prediction and not the text column as it is very large texts, and it will take a long time to process it.

# first 5 rows of the dataset
news_data.head()



# separating feature and target.
# target :: The column which contains the end conclusion that if the data is fake or not.
# Feature :: The column through which we are predicting if the data is fake or not.(The input for the ml model).

X = news_data.drop(columns='label', axis =1)  # Separating the target from feature.
Y = news_data['label']


print(X)


print(Y)


#Stemming :: It is the process of reducing a word to its root word. Example :: Words like enjoying,enjoyable,enjoyed all have the same root word enjoy.
port_stem = PorterStemmer() #Loading of function to a new variable.


def stemming(content):  #Defining a function to do stemming.
    stemmed_content = re.sub('[^a-zA-Z]',' ',content) # '[^a-zA-Z]' This will ensure that all other string values except the alphabets will be removed and replaced with space (' ')form the content column.
    stemmed_content = stemmed_content.lower() # Will convert all the alphabets into lower case.
    stemmed_content = stemmed_content.split() # Will split all the words where there are spaces.
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]  # This will remove all the stopwords from the stemmed_content.
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


news_data['content'] = news_data['content'].apply(stemming)


print(news_data['content'])


X = news_data['content'].values
Y = news_data['label'].values


print(X)


print(Y)


print(Y.shape)


# converting the textual data to feature vectors
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


print(X)



