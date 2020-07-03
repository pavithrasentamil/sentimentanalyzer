import csv
import numpy as np 
import pandas as pd 
import re
import nltk 
import pickle
#import matplotlib.pyplot as plt
#%matplotlib inline
#data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
# tweets = pd.read_csv("E:\trains\train.csv")
# df=pd.DataFrame(tweets)
#airline_tweets.head()
#plot_size = plt.rcParams["figure.figsize"] 
#print(plot_size[0]) 
#print(plot_size[1])

#plot_size[0] = 8


##plt.rcParams["figure.figsize"] = plot_size 
# airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
# airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
# airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
# airline_sentiment.plot(kind='bar')
#import seaborn as sns

#sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=airline_tweets)

#features = tweets.iloc[:, 1].values
data_source_url = "E:\\imdb1.xlsx"
airline_tweets = pd.read_excel(data_source_url)
features = airline_tweets.iloc[:, 0].values
#print(features[0])
labels = airline_tweets.iloc[:, 1].values
# labels = tweets.iloc[:, 2].values
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTree
text_classifier = DecisionTree()
text_classifier.fit(X_train, y_train)
#filename='actualmod.sav'
#pickle.dump(text_classifier,open(filename,'wb'))
#loadmod=pickle.load(open(filename,'rb'))
predictions = text_classifier.predict(X_test)#here make alt like api help
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print("=====")
print(classification_report(y_test,predictions))
print("=====")
print(accuracy_score(y_test, predictions))


