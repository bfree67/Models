'''
LSTM RNN for predicting timeseries


'''
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
import math
import easygui
from pandas import ExcelWriter
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import time

# convert an array of values into a dataset matrix
def TensorForm(data, look_back):
    #determine number of data samples
    rows_data,cols_data = np.shape(data)
    
    #determine # of batches based on look-back size
    tot_batches = int(rows_data-look_back)+1
    
    #initialize 3D tensor
    threeD = np.zeros(((tot_batches,look_back,cols_data)))
    
    # populate 3D tensor
    for sample_num in range(tot_batches):
        for look_num in range(look_back):
            threeD[sample_num,:,:] = data[sample_num:sample_num+(look_back),:]
    
    return threeD
	
# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
title = 'Choose a data file...'
data_file_name = easygui.fileopenbox(title)
df = read_csv(data_file_name, engine='python', skipfooter=3)

a = list(df)

for i in range (len(a)):
    print i, a[i]

last_col = np.shape(df)[1] - 1

# pick column to predict
try:
    target_col = int(raw_input("Select the column number to predict (default = " + a[last_col] + " : "))
except ValueError:
    target_col = last_col   #choose last column as default

# choose look-ahead to predict   
try:
    lead_time =  int(raw_input("How many hours ahead to predict (default = 24)?: "))
except ValueError:
    lead_time = 24
    
#convert to floating numpy arrays
dataset1 = df.fillna(0).values
dataset1 = dataset1.astype('float32')
dataplot1 = dataset1[lead_time:,target_col]  #shift training data
dataplot1 = dataplot1.reshape(-1,1)
    
# normalize the dataset
try:
    process = raw_input("Does the data need to be pre-preprocessed Y/N? (default = y) ")
except ValueError:
    process = 'y'
    
if process == 'Y' or 'y':
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))
    
    dataset = scalerX.fit_transform(dataset1)
    dataplot = scalerY.fit_transform(dataplot1)
    
    print'\nData processed using MinMaxScaler'
else:
    print'\nData not processed'

# split into train and test sets
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# prepare output arrays
trainY, testY = dataplot[0:train_size], dataplot[train_size:len(dataset)]

n,p = np.shape(trainY)
if n < p:
    trainY = trainY.T
    testY = testY.T

# resize input sets
trainX1 = train[:len(trainY),]
testX1 = test[:len(testY),]
  
# get number of epochs (default = 100)
try:
    n_epochs = int(raw_input("Number of epochs? (Default = 10)? "))
except ValueError:
    n_epochs = 10
    
# prepare input Tensors
try:
    look_back = int(raw_input("Number of recurrent (look-back) units? (Default = 8)? "))
except ValueError:
    look_back = 8
trainX = TensorForm(trainX1, look_back)
testX = TensorForm(testX1, look_back)
input_nodes = trainX.shape[2]

# trim target arrays to match input lengths
if len(trainX) < len(trainY):
    trainY = np.asmatrix(trainY[:len(trainX)])
    
if len(testX) < len(testY):
    testY = np.asmatrix(testY[:len(testX)])

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(input_nodes, activation='sigmoid', recurrent_activation='tanh', 
                input_shape=(testX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='nadam')
model.fit(trainX, trainY, epochs=n_epochs, batch_size=4, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scalerY.inverse_transform(trainPredict)
trainY = scalerY.inverse_transform(trainY)
testPredict = scalerY.inverse_transform(testPredict)
testY = scalerY.inverse_transform(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# make timestamp for unique filname
stamp = str(time.clock())  #add timestamp for unique name
stamp = stamp[0:2] 

# generate filename and remove extra periods
filename = 'FinErr_lstm_'+ str(n_epochs) + '_' + stamp + '.xlsx'    #example output file
if filename.count('.') == 2:
    filename = filename.replace(".", "",1)

#write results to file    
writer = ExcelWriter(filename)
pd.DataFrame(trainPredict).to_excel(writer,'Train-predict') #save prediction output
pd.DataFrame(trainY).to_excel(writer,'obs-train') #save observed output
pd.DataFrame(testPredict).to_excel(writer,'Test-predict') #save output training data
pd.DataFrame(testY).to_excel(writer,'obs_test') 
writer.save()
print'File saved in ', filename

# plot baseline and predictions
plt.close('all')
plt.plot(testY)
plt.plot(testPredict)
plt.show()
