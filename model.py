import dynet as dy


class model():

    def __init__(self,vocab,numLayers = 1):
        # vocab will be used to train the embeddings in the parameter collection

        self.word2idx = {w:i for i, w in  enumerate(vocab)}

        EMBEDDING_SIZE = 10
        HIDDEN_DIM = 2
        paramCollections = dy.ParameterCollection()
        wordEmbeddings = paramCollections.add_lookup_parameters((len(vocab),EMBEDDING_SIZE)
        # self.builder = dy.SimpleRNNBuilder(1,EMBEDDING_SIZE,HIDDEN_DIM,paramCollections)

        return 0

    def train(self,train_data):

        return 0
