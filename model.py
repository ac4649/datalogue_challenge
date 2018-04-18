import dynet as dy
import pandas as pd
import csv
import numpy as np
from tqdm import *

class DLmodel():

    def loadEmbedds(self,filename,embedLength):
        gloveEmbedds = pd.read_table(filename,sep=" ",quoting=csv.QUOTE_NONE,header=None)
        gloveEmbedds.columns = ['word'] +  ['dim'+str(i) for i in range(embedLength)]
        return gloveEmbedds

    # def __init__(self,vocab,numLayers = 1): # no need for vocab if using glove
    def __init__(self):
        # vocab will be used to train the embeddings in the parameter collection

        # load the glove pre-trained embeddings
        print("Loading Embeddings")
        EMBEDDING_SIZE = 50
        gloveEmbedds = self.loadEmbedds('glove/glove.6B.50d.txt',EMBEDDING_SIZE)
        # set vocab to be the vocab from the glove embeddings
        vocab = gloveEmbedds['word'].values
        embeddings = gloveEmbedds.drop('word',axis=1)
        print(vocab)
        # load the embeddings themselves as lookup parameters
        

        self.word2idx = {w:i for i, w in  enumerate(vocab)}
        self.truth2idx = {'flagged':1, 'not_flagged':0}
        self.idx2truth = {1:'flagged',0:'not_flagged'}

        OUTPUT_DIM = 2

        HIDDEN_DIM = 10
        NUM_LAYERS = 1
        self.paramCollection = dy.ParameterCollection()
        self.wordEmbeddings = self.paramCollection.add_lookup_parameters((len(vocab),EMBEDDING_SIZE),init=dy.NumpyInitializer(embeddings.values))
        self.Weights = self.paramCollection.add_parameters((OUTPUT_DIM,HIDDEN_DIM))
        self.bias = self.paramCollection.add_parameters((OUTPUT_DIM,))
        self.rnnBuilder = dy.SimpleRNNBuilder(NUM_LAYERS,EMBEDDING_SIZE,HIDDEN_DIM,self.paramCollection)

    def computeSoftMax(self,output):

        return dy.softmax(dy.parameter(self.Weights)*output + dy.parameter(self.bias))

    def forwardSequenceWithLoss(self,sequence,truth):
        dy.renew_cg()
        state = self.rnnBuilder.initial_state()
        loss = []
        unkWords = []
        for word in sequence:
            if (word in self.word2idx):
                wordEmbedd = self.wordEmbeddings[self.word2idx[word]]
                state = state.add_input(wordEmbedd)
                # exit()
            else:
                unkWords.append(word)
        # softMax = dy.softmax(dy.parameter(self.Weights)*state.output() + dy.parameter(self.bias))
        softMax = self.computeSoftMax(state.output())
        # print(softMax.value())
        # print(self.truth2idx[truth])
        # print(-dy.pickneglogsoftmax(softMax,self.truth2idx[truth]).value())
        loss.append(-dy.pickneglogsoftmax(softMax,self.truth2idx[truth]))
        return dy.esum(loss), unkWords

    def train(self,train_data,maxEpochs = 10):
        print("Training")
        trainer = dy.SimpleSGDTrainer(self.paramCollection)
        allUnk = []
        allLosses = []
        for j in trange(maxEpochs):
            epochLoss = 0
            for i, data in train_data.iterrows():
                #compute the loss
                loss, unkWords = self.forwardSequenceWithLoss(data['response_text_array'],data['class'])
                loss.backward()
                trainer.update()
                allUnk.extend(unkWords)
                # if i % 10:
                #     print(loss.value())
                epochLoss = loss.value()
                # exit()
            allLosses.append(epochLoss)

        # print(allLosses)
        # print(allUnk)
        return allUnk, allLosses

    def forwardSequenceNoLoss(self,sequence):
        dy.renew_cg()
        state = self.rnnBuilder.initial_state()
        loss = []
        for word in sequence:
            if (word in self.word2idx):
                wordEmbedd = self.wordEmbeddings[self.word2idx[word]]
                state = state.add_input(wordEmbedd)
        
        
        softMax = self.computeSoftMax(state.output())
        argMax = np.argmax(softMax)
        # print(softMax.value())
        # print(np.argmax((softMax.value())))
        # print(self.idx2truth[np.argmax((softMax.value()))])
        # exit()
        # print(self.idx2truth[argMax])
        return self.idx2truth[argMax]

    def predict(self,test_data):
        outputSeries = pd.Series(index=test_data.index)
        for i, data in test_data.iterrows():
            outputSeries.loc[i] = self.forwardSequenceNoLoss(data['response_text_array'])
        
        # print(outputSeries)
        return outputSeries

    def computeDevAcc(self,dev_data):
        # print(dev_data['class'].iloc[0])
        predictions = self.predict(dev_data)
        # print(predictions)
        # print(dev_data['class'])
        acc = (predictions == dev_data['class']).sum() / len(predictions)

        # truePos = np.sum([predictions == True and dev_data['class'] == True ])
        # trueNeg = np.sum([predictions == False and dev_data['class'] == False ])
        # falsePos = np.sum([predictions == True and dev_data['class'] == False ])
        # falseNeg = np.sum([predictions == False and dev_data['class'] == True ])
        print(acc)
        # print(truePos)
        # print(trueNeg)
        # print(falsePos)
        # print(falseNeg)

