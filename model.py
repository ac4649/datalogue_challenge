import dynet as dy
import pandas as pd


class DLmodel():

    def __init__(self,vocab,numLayers = 1):
        # vocab will be used to train the embeddings in the parameter collection

        self.word2idx = {w:i for i, w in  enumerate(vocab)}
        self.truth2idx = {'flagged':1, 'not_flagged':0}

        EMBEDDING_SIZE = 100
        HIDDEN_DIM = 2
        paramCollection = dy.ParameterCollection()
        self.wordEmbeddings = paramCollection.add_lookup_parameters((len(vocab),EMBEDDING_SIZE))
        self.Weights = paramCollection.add_parameters((EMBEDDING_SIZE,HIDDEN_DIM))
        self.bias = paramCollection.add_parameters((EMBEDDING_SIZE,))
        self.rnnBuilder = dy.SimpleRNNBuilder(1,EMBEDDING_SIZE,HIDDEN_DIM,paramCollection)

    def forwardSequenceWithLoss(self,sequence,truth):
        dy.renew_cg()
        state = self.rnnBuilder.initial_state()
        loss = []
        for word in sequence:
            wordEmbedd = self.wordEmbeddings[self.word2idx[word]]
            state = state.add_input(wordEmbedd)
            loss.append(-dy.pickneglogsoftmax(state.output(),self.truth2idx[truth]))
        return dy.esum(loss)

    def train(self,train_data):

        for i, data in train_data.iterrows():
            #compute the loss
            loss = self.forwardSequenceWithLoss(data['response_text_array'],data['class'])
            loss.backward()
            # if i % 10:
            #     print(loss.value())
        return

    def forwardSequenceNoLoss(self,sequence):
        dy.renew_cg()
        state = self.rnnBuilder.initial_state()
        loss = []
        for word in sequence:
            if (word in self.word2idx):
                wordEmbedd = self.wordEmbeddings[self.word2idx[word]]
                state = state.add_input(wordEmbedd)
        print(dy.log_softmax(state.output()))
        return 

    def predict(self,train_data):
        outputSeries = pd.Series(index=train_data.index)
        for i, data in train_data.iterrows():
            #compute the loss
            outputSeries.loc[i] = self.forwardSequenceNoLoss(data['response_text_array'])
        return
