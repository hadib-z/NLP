#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation

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


df = pd.read_csv("twitter-training-data.txt", sep="\t", names = ["ID", "Sentiment", "Tweet"])

df["Tidy Tweet"] = df["Tweet"].apply(tp.preproc)


for classifier in ['myclassifier1', 'myclassifier2', 'myclassifier3']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'myclassifier1':
        print('Training ' + classifier)
        X, y = df["Tidy Tweet"], df["Sentiment"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.99)
        
        count_vect = CountVectorizer(tokenizer=tokenizer.tokenize) 
        classif = LogisticRegression()
        
        pipeline = Pipeline([
                ('vectorizer', count_vect),
                ('classifier', classif)
            ])
        
        pipeline.fit(X_train, y_train)
        
#    elif classifier == 'myclassifier2':
#        print('Training ' + classifier)
#        # TODO: extract features for training classifier2
#        # TODO: train sentiment classifier2
#    elif classifier == 'myclassifier3':
#        print('Training ' + classifier)
#        # TODO: extract features for training classifier3
#        # TODO: train sentiment classifier3

    for testset in testsets.testsets:
        dft = pd.read_csv(testset, sep="\t", names = ["ID", "Sentiment", "Tweet"])

        dft["Tidy Tweet"] = dft["Tweet"].apply(tp.preproc)
        
        predictions = dict(zip(dft.iloc[dft["Tidy Tweet"].index, 0], pipeline.predict(dft["Tidy Tweet"])))
#        predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)


print("\n", "--- {} seconds ---".format(time.time() - start_time))
