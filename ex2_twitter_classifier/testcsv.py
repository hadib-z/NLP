# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 02:29:47 2018

@author: Hadib
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import csv

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



#with open("twitter-training-data.txt", "r", encoding="utf-8") as file:
#    f = csv.reader(file, delimiter='\t')
#    idf = [i[0] for i in f]
#    sentiments = [i[1] for i in f]
#    tweets = [i[2] for i in f]
#    

      
id_gts2 = {}
cleantweets2 = []
sentiments2 = []
ids2 =[]
with open("twitter-training-data.txt", 'r', encoding="utf-8") as fh:
    for line in fh:
      fields2 = line.split('\t')
      tweetid2 = fields2[0]
      gt2 = fields2[1]
      tweet2 = fields2[2]
      cleantweet2 = tp.preproc(tweet2)
      cleantweets2.append(cleantweet2)
      sentiments2.append(gt2)
      ids2.append(tweetid2)

      id_gts2[tweetid2] = gt2



for classifier in ['myclassifier1', 'myclassifier2', 'myclassifier3']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'myclassifier1':
        print('Training ' + classifier)
        X, y = cleantweets2, sentiments2
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.99)
        
        count_vect = CountVectorizer(tokenizer=tokenizer.tokenize) 
        classif = LogisticRegression()
        
        pipeline = Pipeline([
                ('vectorizer', count_vect),
                ('classifier', classif)
            ])
        
        pipeline.fit(X_train, y_train)
        
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        
        print(np.mean(preds == y_test))
        
    elif classifier == 'myclassifier2':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
    elif classifier == 'myclassifier3':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3

    for testset in testsets.testsets:
        cleantweets = []
        sentiments = []
        ids =[]
        stuff = {}
        with open(testset, 'r', encoding="utf-8") as fh:
            for line in fh:
              fields = line.split('\t')
              tweetid = fields[0]
              gt = fields[1]
              tweet = fields[2]
              cleantweet = tp.preproc(tweet)
              cleantweets.append(cleantweet)
              sentiments.append(gt)
              ids.append(tweetid)
              
              stuff[tweetid] = cleantweet
        
        
        predictions = dict(zip(stuff.keys(), pipeline.predict(stuff.values())))
#        predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)


print("\n", "--- {} seconds ---".format(time.time() - start_time))
