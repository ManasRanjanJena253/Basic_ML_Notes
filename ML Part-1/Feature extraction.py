# Feature extraction :: The mapping from textual data to real valued vectors is called feature extraction.
# Feature extraction is done as it will be hard for the machine to learn directly from the text.
# Feature vectors are numerical representation of the data given.
# Bag of words(BOW) :: list of unique words in the text corpus.
# Term Frequency-Inverse Document Frequency(TF-IDF) :: To count the no. of times each word appears in a document.
# TF = (No. of times term t appears in a document) / (No. of terms in the document.)
# IDF = log(N/n),where,N is the number of documents and n is the number of documents  term t has appeared in.
# IDF value of a rare word is high,whereas the IDF of a frequent word is low.
# TF-IDF value of a term = TF * IDF.


#About the Dataset:

#id: unique id for a news article
#title: the title of a news article
#author: author of the news article
#text: the text of the article; could be incomplete
#label: a label that marks whether the news article is real or fake: 1: Fake news 0: real News
#Importing the Dependencies


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


import nltk
nltk.download('stopwords')



# printing the stopwords in English
print(stopwords.words('english'))


#Data Pre-processing


# loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv('/content/train.csv')


print(news_dataset.shape)


# print the first 5 rows of the dataframe
news_dataset.head()


# counting the number of missing values in the dataset
news_dataset.isnull().sum()


# replacing the null values with empty string
news_dataset = news_dataset.fillna('')


# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']


print(news_dataset['content'])


# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']


print(X)
print(Y)


#Stemming:

#Stemming is the process of reducing a word to its Root word

#example: actor, actress, acting --> act


port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content
                       if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


news_dataset['content'] = news_dataset['content'].apply(stemming)


print(news_dataset['content'])


#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values


print(X)


print(Y)


print(Y.shape)


#Tf-Idf


# convert the textual data to Feature Vectors
vectorizer = TfidfVectorizer()


vectorizer.fit(X)

X = vectorizer.transform(X)


print(X)
