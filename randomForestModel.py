from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv
import numpy as np
from tqdm import *

class randomForestModel():

    def loadEmbedds(self,filename,embedLength):
        gloveEmbedds = pd.read_table(filename,sep=" ",quoting=csv.QUOTE_NONE,header=None)
        gloveEmbedds.columns = ['word'] +  ['dim'+str(i) for i in range(embedLength)]
        return gloveEmbedds

    def __init__(self, embedding_size = 200, embedding_file = 'glove/glove.6B.200d.txt', numEstimators = 10):

        self.EMBEDDING_SIZE = embedding_size
        gloveEmbedds = self.loadEmbedds(embedding_file,self.EMBEDDING_SIZE)
        # set vocab to be the vocab from the glove embeddings
        vocab = gloveEmbedds['word'].values
        self.embeddings = gloveEmbedds.drop('word',axis=1)

        self.word2idx = {w:i for i, w in  enumerate(vocab)}
        self.truth2idx = {'flagged':1, 'not_flagged':0}
        self.idx2truth = {1:'flagged',0:'not_flagged'}

        self.model = RandomForestClassifier(n_estimators=numEstimators)

    def computeX(self,sentences):
        returnFrame = pd.DataFrame(index=sentences.index,columns=['dim'+str(i) for i in range(self.EMBEDDING_SIZE)])
        for index, sentence in sentences.iteritems():
            # print(index)
            sentenceFrame = pd.DataFrame(index = [i for i in range(len(sentence))],columns = ['dim'+str(i) for i in range(self.EMBEDDING_SIZE)])
            for curIndex, word in enumerate(sentence):
                if (word in self.word2idx):
                    sentenceFrame.iloc[curIndex] = pd.Series(self.embeddings.iloc[self.word2idx[word]])
            
            # returnFrame.loc[index] = sentenceFrame.sum() # just doing summation of the embeddings
            returnFrame.loc[index] = sentenceFrame.mean() # just doing summation of the embeddings

        return returnFrame

    def train(self,trainDataFrame):
        computedX = self.computeX(trainDataFrame['response_text_array'])
        # print(trainDataFrame['class'].describe())
        self.model.fit(computedX,trainDataFrame['class'])
    
    def predict(self,testDataFrame):
        computedX = self.computeX(testDataFrame['response_text_array'])
        predictions = self.model.predict(computedX)
        return predictions


    def computeDevAcc(self,dev_data,printStats = True):
        # print(dev_data['class'].iloc[0])
        predictions = self.predict(dev_data)
        # print(predictions)
        # print(dev_data['class'])
        acc = (predictions == dev_data['class']).sum() / len(predictions)

        trueFlagged = dev_data['class'][dev_data['class'] == self.idx2truth[1]]
        trueFlaggedPredictions = predictions[dev_data['class'] == self.idx2truth[1]]

        trueNotFlagged = dev_data['class'][dev_data['class'] == self.idx2truth[0]]
        trueNotFlaggedPredictions = predictions[dev_data['class'] == self.idx2truth[0]]

        truePos = np.sum(trueFlagged == trueFlaggedPredictions)
        trueNeg = np.sum(trueNotFlagged == trueNotFlaggedPredictions)
        falsePos = np.sum(trueFlagged != trueFlaggedPredictions)
        falseNeg = np.sum(trueNotFlagged != trueNotFlaggedPredictions)
        if printStats:
            print("Accuracy: " +str(acc))
            print("True Positive #: " + str(truePos))
            print("True Negative #: " + str(trueNeg))
            print("False Positive #: " + str(falsePos))
            print("False Negative #: " + str(falseNeg))

        return predictions, [acc, truePos, trueNeg, falsePos, falseNeg]



