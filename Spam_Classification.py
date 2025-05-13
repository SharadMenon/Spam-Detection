import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("email.csv")
df['Category_num'] = df['Category'].map({"ham":0, "spam":1})
df = df.iloc[:-1] #As last row is not needed and is giving NAN value
y = df['Category_num']

import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    stop_words = set(stopwords.words("english")) 
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df['cleaned_message'] = df['Message'].apply(preprocess_text)
#Feature Extraction, Converting the text into numbers for the machine to understand
#Two methods
#Count Vectorizer - creates matrix where row - msg column - word
#TF-IDF Vectorizer(Used here and recommended)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])


#Building the model now, We will use Naive Bayed - MultinomialNB as its best for text data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2)
classifier = MultinomialNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))


