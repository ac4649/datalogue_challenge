# Adrien Cogny
# Started Tuesday Apr 17 2018
# Due Tuesday Apr 24 2018
import pandas as pd
# import dynet as dy
import nltk
import re
from tqdm import *
from rnnModel import RNNmodel
from randomForestModel import randomForestModel

import pickle

# ----------------------------- DATA LOADING ----------------------------- #

def prepareString(string):
    string = re.sub(r'[^A-Za-z0-9\s\']', ' ', string)
    string = string.replace('\'',' \'')
    string = string.lower()
    stringArr = string.split(' ')

    newArray = []
    for word in stringArr:
        includeWord = True
        if word == '':
            includeWord = False

        if word == '\'t':
            includeWord = False

        if word in nltk.corpus.stopwords.words('english'):
            includeWord = False

        if includeWord == True:
            newArray.append(word)

    return newArray

def loadData(filepath):
    # load the data from the file
    data = pd.read_csv(filepath)

    # remove extraneous columsn from the dataframe
    data = data.drop(['Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7'],axis=1)
    data['response_text_array'] = data['response_text'].apply(lambda x: prepareString(x))

    return data

def convertSentence(sentence):
    outputFrame = pd.DataFrame(index=[0],columns=['response_text_array'])
    outputFrame.iloc[0]['response_text_array'] = prepareString(sentence)
    return outputFrame

def convertSentences(sentences):
    outputFrame = pd.DataFrame(index=[i for i in range(len(sentences))],columns=['response_text_array'])
    for i in range(len(sentences)):
        outputFrame.iloc[i]['response_text_array'] = prepareString(sentences[i])

    return outputFrame




# ----------------------------- DATA DESCRIPTION ----------------------------- #
# column: response_id
# description: id of the response (we have 80 unique ids)

# column: class
# description: whether the response_text was flagged for referal to a specialist 
#               (we have 2 unique values: not_flagged or flagged)

# column: response_text
# description: natural language response a user inputed to the chat bot, 
#               must be classified as flagged for assistance or not_flagged.


def describeData(frame):
    # looking at statistics for the class column in the dataset to see if it is a roughly even dataset 
    print(frame['class'].describe())
    # the fact that the frequency of the not_flagged is 55 over the 80 shows an
    #  imbalance toward the not_flagged class.


# ------------------- SEPARATION INTO TRAIN AND TEST SPLITS -------------------#
def splitDataSet(data):
    # train on a certain percentage of the data
    percentTrain = 0.7

    # sample a fraction equivalent to the percent rain from the data
    data_train = data.sample(frac=percentTrain) 
    # print(data_train.shape)
    # the rest of the data will be used as dev
    data_dev = data.drop(data_train.index)
    # print(data_dev.shape)
    return data_train, data_dev


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

# model = RNNmodel(train_vocab) # change to not need train_vocab when using glove

def runRNNModel(data, hidden_dim = 4, num_layers = 1, embedding_size = 200, embedding_file = 'glove/glove.6B.200d.txt', maxEpochs = 200, saveModel = False):
    
    data_train, data_dev = splitDataSet(data)
    model = RNNmodel(hidden_dim = hidden_dim, num_layers = num_layers, embedding_size = embedding_size, embedding_file = embedding_file)
    unknowns, losess = model.train(data_train,maxEpochs = maxEpochs)
    # print(unknowns)

    pd.DataFrame(losess).to_csv('lossesEpoch.csv')

    # results = model.predict(data_dev)
    predictions, stats = model.computeDevAcc(data_dev,printStats=False)

    params = model.getModelParams()

    if saveModel:
        model.saveModel('curModel.model')
    # print(stats)
    # print(params)
    return predictions, stats, params, model

def runRNNTrials(data):
    # Run the model multiple times with a given set of parameters to get the best parameters on average 
    # (no matter what the training )
    numRuns = 15
    # for j in tqdm([300],desc='Param Variation'):
    for j in trange(23,26,desc='Param Variation'):
        overallModelStats = pd.DataFrame(index=[i for i in range(numRuns)],columns=['maxEpochs','num_layers','embeddingSize','hiddenDim','train_loss','accuracy','truePos','trueNeg','falsePos','falseNeg'])
        for i in trange(0,numRuns,desc="Param Runs "):
            # print("Run: " +str(i))
            preds, stats, params = runRNNModel(data,hidden_dim = 3, num_layers = j, embedding_size = 100, embedding_file = 'glove/glove.6B.100d.txt',maxEpochs = 400)
            # overallModelStats.append((stats,params))
            # curRun = pd.Series([param for param in params]+[stats for i in stats])
            overallModelStats.iloc[i]['maxEpochs'] = params[0]
            overallModelStats.iloc[i]['num_layers'] = params[1]
            overallModelStats.iloc[i]['embeddingSize'] = params[2]
            overallModelStats.iloc[i]['hiddenDim'] = params[3]
            overallModelStats.iloc[i]['train_loss'] = stats[5]

            overallModelStats.iloc[i]['accuracy'] = stats[0]
            overallModelStats.iloc[i]['truePos'] = stats[1]
            overallModelStats.iloc[i]['trueNeg'] = stats[2]
            overallModelStats.iloc[i]['falsePos'] = stats[3]
            overallModelStats.iloc[i]['falseNeg'] = stats[4]
        overallModelStats.to_csv('layers_stats'+str(j)+'.csv')

    return overallModelStats

def runRandomForestModel(data,n_estimators = 2,embedding_size = 100, embedding_file = 'glove/glove.6B.100d.txt'):
    data_train, data_dev = splitDataSet(data)
    rfModel = randomForestModel(numEstimators=n_estimators,embedding_size = embedding_size, embedding_file = embedding_file)

    rfModel.train(data_train)

    predictions = rfModel.predict(data_dev)
    # print(predictions)

    predictions, stats = rfModel.computeDevAcc(data_dev,printStats=False)

    # if saveModel:
    #     model.saveModel('curModel.model')
    # # print(stats)
    # # print(params)
    return stats, rfModel

def runRandomForestTrials(data):
    numRuns = 15
    randForestStats = pd.DataFrame(index=[i for i in range(numRuns)],columns = ['embeddingSize','n_estimators','accuracy','truePos','trueNeg','falsePos','falseNeg'])

    # for j in trange(1,21,desc='Changing parameters'):
    for j in tqdm([50, 100, 200, 300],desc='Param Variation'):
        for i in trange(numRuns,desc='Running Models'):
            curStats = runRandomForestModel(data,n_estimators = 10,embedding_size = j, embedding_file = 'glove/glove.6B.' + str(j) +'d.txt')
            randForestStats.iloc[i]['embeddingSize'] = j
            randForestStats.iloc[i]['accuracy'] = curStats[0]
            randForestStats.iloc[i]['truePos'] = curStats[1]
            randForestStats.iloc[i]['trueNeg'] = curStats[2]
            randForestStats.iloc[i]['falsePos'] = curStats[3]
            randForestStats.iloc[i]['falseNeg'] = curStats[4]
            randForestStats.iloc[i]['n_estimators'] = 10

        # print(randForestStats)
        # print(randForestStats.mean())
        randForestStats.to_csv('Results/RandomForest/embedding_dimm/randomForest_embedd_size_'+str(j)+'_estimators.csv')


def getRNNFinalModel(data):
    numRuns = 15
    overallModelStats = pd.DataFrame(index=[i for i in range(numRuns)],columns=['maxEpochs','num_layers','embeddingSize','hiddenDim','train_loss','accuracy','truePos','trueNeg','falsePos','falseNeg'])
    
    currentBestTPR = 0
    bestModel = None
    for i in trange(0,numRuns,desc="Param Runs "):
        # print("Run: " +str(i))
        preds, stats, params, model = runRNNModel(data,hidden_dim = 3, num_layers = 15, embedding_size = 100, embedding_file = 'glove/glove.6B.100d.txt',maxEpochs = 400)
        if (stats[1] + stats[4]) > 3: # only check if the data has more than 3 positive examples (approx mean of the trials).
            if (stats[1] + stats[4]) != 0:
                bestTPR = stats[1]/(stats[1] + stats[4])
                if bestTPR > currentBestTPR:
                    bestModel = model
                    bestModel.saveModel('RNNFinalModel/bestRNN.model')
                    currentBestTPR = bestTPR

    return bestModel, bestTPR


def getRandomForestFinalModel(data):
    numRuns = 15
    overallModelStats = pd.DataFrame(index=[i for i in range(numRuns)],columns=['maxEpochs','num_layers','embeddingSize','hiddenDim','train_loss','accuracy','truePos','trueNeg','falsePos','falseNeg'])
    
    currentBestTPR = 0
    bestModel = None
    for i in trange(0,numRuns,desc="Param Runs "):
        # print("Run: " +str(i))
        stats, model = runRandomForestModel(data,n_estimators = 10,embedding_size = 200, embedding_file = 'glove/glove.6B.200d.txt')
        if (stats[1] + stats[4]) > 3:  # only check if the data has more than 3 positive examples (approx mean of the trials).
            if (stats[1] + stats[4]) != 0:
                bestTPR = stats[1]/(stats[1] + stats[4])
                if bestTPR > currentBestTPR:
                    bestModel = model
                    bestModel.saveModel('RandomForestFinalModel/bestRandomForest.model')
                    currentBestTPR = bestTPR

    return bestModel, bestTPR


def loadRandomForestModel(filename):
    model = pickle.load(open(filename,'rb'))
    return model

data = loadData('deepnlp/Sheet_1.csv')

# runRandomForestTrials(data) # runs the tests defined in the function

# runRNNTrials(data) # runs the tests defined in the function

finalRNN, finalTPR = getRNNFinalModel(data) # given the optimal parameters found in the tests, gets the final model
print("\n")
print(finalTPR)

finalRF, finalRFTPR = getRandomForestFinalModel(data) # given the optimal parameters found in the tests, gets the final model
print("\n")
print(finalRFTPR)

convertedSentence = convertSentence('Roommate when he was going through death and loss of a gf. Did anything to get him out of his bedroom.')
convertedSentences = convertSentences(
    ['This is an example of multiple sentences.',
    'Where each sentence is an index in the array.'])


curRNN = RNNmodel(model_filename='RNNFinalModel/bestRNN.model')
curRFM = loadRandomForestModel('bestRandomForest.model')

print(curRNN.predict(convertedSentence))
print(curRFM.predict(convertedSentence))

print(curRNN.predict(convertedSentences))
print(curRFM.predict(convertedSentences))