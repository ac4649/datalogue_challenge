import sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv
import numpy as np
from tqdm import *

class randomForestModel():

    def loadEmbedds(self,filename,embedLength):
        gloveEmbedds = pd.read_table(filename,sep=" ",quoting=csv.QUOTE_NONE,header=None)
        gloveEmbedds.columns = ['word'] +  ['dim'+str(i) for i in range(embedLength)]
        return gloveEmbedds

    def __init__(self,vocab):
        self.word2idx = {w:i for i, w in  enumerate(vocab)}
        self.truth2idx = {'flagged':1, 'not_flagged':0}
        self.idx2truth = {1:'flagged',0:'not_flagged'}

        self.model = RandomForestClassifier()

    def computeX(self,series):
        for i in series:
            print(i)
        return series.sum(axis=1) # just doing summation of the embeddings

    def train(self,trainDataFrame):
        computedX = computeX(trainDataFramedata['response_text_array'])
        self.model.fit(computedX,trainDataFrame['class'])

