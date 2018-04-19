# Adrien Cogny
# Started Tuesday Apr 17 2018
# Due Tuesday Apr 24 2018
import pandas as pd
import dynet as dy
import nltk
import re

from model import DLmodel

# ----------------------------- DATA LOADING ----------------------------- #
# load the data from the file
data = pd.read_csv('deepnlp/Sheet_1.csv')

# remove extraneous columsn from the dataframe
data = data.drop(['Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7'],axis=1)


def prepareString(string):
    # print(string)
    string = re.sub(r'[^A-Za-z0-9\s\']', ' ', string)
    string = string.replace('\'',' \'')
    string = string.lower()
    stringArr = string.split(' ')

    # print(stringArr)
    newArray = []
    # numRemovedWords = 0
    # numWordsTotal = len(stringArr)
    # now do some stemming, lemmatization and removal of stop words
    for word in stringArr:
        includeWord = True
        if word == '':
            includeWord = False

        if word == '\'t':
            includeWord = False

        if word in nltk.corpus.stopwords.words('english'):
            includeWord = False
            # numRemovedWords += 1

        if includeWord == True:
            newArray.append(word)

    # print(numRemovedWords/numWordsTotal)
    # exit()
    return newArray

data['response_text_array'] = data['response_text'].apply(lambda x: prepareString(x))


# ----------------------------- DATA DESCRIPTION ----------------------------- #
# column: response_id
# description: id of the response (we have 80 unique ids)

# column: class
# description: whether the response_text was flagged for referal to a specialist 
#               (we have 2 unique values: not_flagged or flagged)

# column: response_text
# description: natural language response a user inputed to the chat bot, 
#               must be classified as flagged for assistance or not_flagged.


# looking at statistics for the class column in the dataset to see if it is a roughly even dataset 
print(data['class'].describe())
# the fact that the frequency of the not_flagged is 55 over the 80 shows an
#  imbalance toward the not_flagged class.


# ------------------- SEPARATION INTO TRAIN AND TEST SPLITS -------------------#
# train on a certain percentage of the data
percentTrain = 0.7

# sample a fraction equivalent to the percent rain from the data
data_train = data.sample(frac=percentTrain) 
# print(data_train.shape)
# the rest of the data will be used as dev
data_dev = data.drop(data_train.index)
# print(data_dev.shape)


# ------------------- CREATION OF VOCABULARY (INCLUDING UNKNOWN TOKENS) -------------------#
# Generate the vocabulary for the train data
def generateVocab(pandasSeries):
    vocab = []
    for element in pandasSeries:
        for word in element:
            vocab.append(word)
    vocab.append('UNK')
    return list(set(vocab))

# ------------------- CREATION OF MODEL -------------------#
# print(data_train['response_text_array'])
# train_vocab = generateVocab(data_train['response_text_array']) # remove if using glove

# model = DLmodel(train_vocab) # change to not need train_vocab when using glove
model = DLmodel()
unknowns, losess = model.train(data_train,maxEpochs = 100)
# print(unknowns)

pd.DataFrame(losess).to_csv('lossesEpoch.csv')

# results = model.predict(data_dev)
predictions, stats = model.computeDevAcc(data_dev)

params = model.getModelParams()

model.saveModel('curModel.model')
print(stats)
print(params)
