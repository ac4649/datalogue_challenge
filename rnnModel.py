import dynet as dy
import pandas as pd
import csv
import numpy as np
from tqdm import *

class RNNmodel():

    def loadEmbedds(self,filename,embedLength):
        gloveEmbedds = pd.read_table(filename,sep=" ",quoting=csv.QUOTE_NONE,header=None)
        gloveEmbedds.columns = ['word'] +  ['dim'+str(i) for i in range(embedLength)]
        return gloveEmbedds

    # def __init__(self,vocab,numLayers = 1): # no need for vocab if using glove
    def __init__(self, hidden_dim = 4, num_layers = 1, embedding_size = 200, embedding_file = 'glove/glove.6B.200d.txt'):
        # vocab will be used to train the embeddings in the parameter collection

        # load the glove pre-trained embeddings
        # print("Loading Embeddings")
        self.EMBEDDING_SIZE = embedding_size
        gloveEmbedds = self.loadEmbedds(embedding_file,self.EMBEDDING_SIZE)
        # set vocab to be the vocab from the glove embeddings
        vocab = gloveEmbedds['word'].values
        embeddings = gloveEmbedds.drop('word',axis=1)
        # print(vocab)
        # load the embeddings themselves as lookup parameters

        self.word2idx = {w:i for i, w in  enumerate(vocab)}
        self.truth2idx = {'flagged':1, 'not_flagged':0}
        self.idx2truth = {1:'flagged',0:'not_flagged'}

        OUTPUT_DIM = 2

        self.HIDDEN_DIM = hidden_dim
        self.NUM_LAYERS = num_layers
        self.paramCollection = dy.ParameterCollection()
        # self.wordEmbeddings = self.paramCollection.add_lookup_parameters((len(vocab),EMBEDDING_SIZE),init=dy.NumpyInitializer(embeddings.values))
        self.wordEmbeddings = self.paramCollection.lookup_parameters_from_numpy(embeddings.values)
        self.Weights = self.paramCollection.add_parameters((OUTPUT_DIM,self.HIDDEN_DIM))
        self.bias = self.paramCollection.add_parameters((OUTPUT_DIM,))
        self.rnnBuilder = dy.SimpleRNNBuilder(self.NUM_LAYERS,self.EMBEDDING_SIZE,self.HIDDEN_DIM,self.paramCollection)


    def getModelParams(self):
        return [self.trainEpochs,self.NUM_LAYERS, self.EMBEDDING_SIZE, self.HIDDEN_DIM]

    def loadModel(self,filePath):
        self.paramCollection.populate(filePath)

    def saveModel(self,filePath):
        self.paramCollection.save(filePath)

    def computeScore(self,output):

        return dy.parameter(self.Weights)*output + dy.parameter(self.bias)

    def forwardSequence(self,sequence):
        dy.renew_cg()
        state = self.rnnBuilder.initial_state()

        # loss = []
        unkWords = []
        losses = []
        for word in sequence:
            if (word in self.word2idx):
                wordEmbedd = self.wordEmbeddings[self.word2idx[word]]
                # wordEmbedd = dy.const_lookup(self.paramCollection,self.wordEmbeddings,self.word2idx[word])
                state = state.add_input(wordEmbedd)
            else:
                unkWords.append(word)

        score = self.computeScore(state.output())

        return score, unkWords

    def train(self,train_data,maxEpochs = 10):
        self.trainEpochs = maxEpochs
        # print("Training")
        trainer = dy.SimpleSGDTrainer(self.paramCollection)
        allUnk = []
        allLosses = []
        for j in trange(self.trainEpochs,desc="Training: "):
            epochLoss = 0
            for i, data in train_data.sample(frac=1).iterrows(): # randomly shuffle the data
                #compute the loss
                score, unkWords = self.forwardSequence(data['response_text_array'])
                # print(self.truth2idx[data['class']])
                # print(dy.softmax(score).value())
                loss = dy.pickneglogsoftmax(score,self.truth2idx[data['class']])
                epochLoss += loss.value()
                loss.backward()
                trainer.update()

                allUnk.extend(unkWords)
                # exit()
            allLosses.append(epochLoss)
            # print(epochLoss)

        # print("Final Loss: " + str(allLosses[maxEpochs-1]))
        self.trainLoss = allLosses[maxEpochs-1]
        # print(allUnk)
        return allUnk, allLosses

    def predict(self,test_data):
        outputSeries = pd.Series(index=test_data.index)
        for i, data in test_data.iterrows():
            probs, unkWords = self.forwardSequence(data['response_text_array'])
            # print(dy.softmax(probs).value())
            # print(np.argmax(dy.softmax(probs).value()))
            outputSeries.loc[i] = self.idx2truth[np.argmax(dy.softmax(probs).value())]
        
        # print(outputSeries)
        return outputSeries

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
            print("Training Loss: " + str(self.trainLoss))

        return predictions, [acc, truePos, trueNeg, falsePos, falseNeg, self.trainLoss]

