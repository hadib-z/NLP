# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 02:09:10 2018

@author: Hadib
"""
import testsets
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import textPreprocessor as tp
import time
import nltk
import numpy as np

tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)
start_time = time.time()


df = pd.read_csv("twitter-training-data.txt", encoding="utf-8", sep="\t", names = ["ID", "Sentiment", "Tweet"])

#df.head(5)

df["Tidy Tweet"] = df["Tweet"].apply(tp.preproc)

#df["Tweet Length"] = df["Tidy Tweet"].apply(len)

#df["Tidy Tweet 2"] = df["Tweet"].apply(tp.test)

#df["Tweet 2 Length"] = df["Tweet"].apply(tp.test).apply(len) - df["Tidy Tweet"].apply(len)

#positives = df['Sentiment'][df.Sentiment == "positive"]
#negatives = df['Sentiment'][df.Sentiment == "negative"]
#neutrals = df['Sentiment'][df.Sentiment == "neutral"]
#
#print("\n", len(positives) + len(negatives) + len(neutrals), "\n")
#
#print('number of positve tagged sentences is:  {}'.format(len(positives)))
#print('number of negative tagged sentences is: {}'.format(len(negatives)))
#print('number of neutral tagged sentences is: {}'.format(len(neutrals)))
#print('total length of the data is:            {}'.format(df.shape[0]))

X, y = df["Tidy Tweet"], df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.99)

count_vect = CountVectorizer(tokenizer=tokenizer.tokenize) 
classifier = LogisticRegression()

pipeline = Pipeline([
        ('vectorizer', count_vect),
        ('classifier', classifier)
    ])

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

print(np.mean(preds == y_test))

for testset in testsets.testsets:
        dft = pd.read_csv(testset, sep="\t", names = ["ID", "Sentiment", "Tweet"])

        dft["Tidy Tweet"] = dft["Tweet"].apply(tp.preproc)
        
        preds = pipeline.predict(dft["Tidy Tweet"])
        
        print(np.mean(preds == dft["Sentiment"]))

print("\n", "--- {} seconds ---".format(time.time() - start_time))

