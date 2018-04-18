import pandas as pd



# load the data from the file
data = pd.read_csv('deepnlp/Sheet_1.csv')
# remove extraneous columsn from the dataframe
data = data.drop(['Unnamed: 3','Unnamed: 4','Unnamed: 5','Unnamed: 6','Unnamed: 7'],axis=1)

# train on a certain percentage of the data
percentTrain = 0.6

# sample a fraction equivalent to the percent rain from the data
data_train = data.sample(frac=percentTrain) 
# the rest of the data will be used as dev
data_dev = data.drop(data_train.index)



print(data.head())