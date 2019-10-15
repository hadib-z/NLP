# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 01:11:43 2018

@author: Hadib
"""

""" Test """

import csv
import re
from textPreprocessor import preproc as proc


#with open("twitter-training-data.txt", encoding='utf-8') as f, open("out","w", encoding='utf-8') as t:
#    tsvreader = csv.reader(f, delimiter="\t")
#    temp = csv.writer(t, delimiter="\t")
#    for row in tsvreader:
#        tweet = row[2]
#        tweet = tweet.lower()
#        tweet = re.sub("(@[A-Za-z0-9_]+)", "USERMENTION", tweet)
#        temp.writerow(row)
#        
#        

        
#with open("twitter-training-data.txt", encoding='utf-8') as f:
#    tsvreader = csv.reader(f, delimiter="\t")
#    for row in tsvreader:
#        tweet = row[2]
#        newtweet = tweet.lower()
#        newtweet = re.sub("(@[A-Za-z0-9_]+)", "USERMENTION", newtweet)
#        print(tweet + '\n' + newtweet + '\n')
        
        
        
#with open("twitter-training-data.txt", encoding='utf-8') as f:
#    tsvreader = csv.reader(f, delimiter="\t")
#    for row in tsvreader:
#        tweet = row[2]
#        proc(tweet)
#        print(tweet + '\n' + newtweet + '\n')
#        



tweet = "hiHI @teset dsfgsd 565 343gggg"
newtext = proc(tweet)
print(tweet + '\n' + newtext + '\n')
