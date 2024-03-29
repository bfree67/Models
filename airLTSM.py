'''
LSTM RNN using Keras libraries for predicting an individual timeseries based 
on multivariate inputs. Takes data in csv column format (each col is variable 
with first row header). 

Provides user input to:
    - select time series to be predicted, 
    - amount of horizon units to predict, 
    - # of epochs to train, and 
    - # of look-back recurrent cells.

Also asks if the data should be processed - if yes, converts data to a 0-1 scale
based on the column for training. Cyclic data such as wind direction should be
pre-coverted to sin/cosine components prior to loading.

Partitions data into training (80%) and testing (20%) sets.

Takes output data (Y) and leads it for future prediction.
Takes input data (X) and converts into a 3D tensor for Keras based RNN training.
The tensor dimensions are based on (# of samples, # of look_backs, and # of input variables)
A sample is the # of variables x # of look_backs - creating a 2D array. Samples
are prepared by sliding down the list of observations.

X and Y sets (training and test) are adjusted for equal length.
Input nodes = # of input variables, batch training is set for online (1)

The model is trained and RMSE calculated for training and test sets. The observed 
and predicted sets are saved in an xlsx file

Brian Freeman 6 Sep 2017
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
print(data_file_name + ' has ' + str(df.shape[0]) + ' observations and ' + str(df.shape[0]) + ' variables')

# pick column to predict
try:
    target_col = int(raw_input("Select the column number to predict (default = " + a[last_col] + "): "))
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
  
# get number of epochs
try:
    n_epochs = int(raw_input("Number of epochs? (Default = 10)? "))
except ValueError:
    n_epochs = 10
    
# prepare input Tensors by requesting # of recurrent look-backs. Default should be the # of variable in data
try:
    look_back = int(raw_input("Number of recurrent (look-back) units? (Default = " + str(lead_time + 2) + ")? "))
except ValueError:
    look_back = lead_time + 2
    
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
model.fit(trainX, trainY, epochs=n_epochs, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scalerY.inverse_transform(trainPredict)
trainY = scalerY.inverse_transform(trainY)
testPredict = scalerY.inverse_transform(testPredict)
testY = scalerY.inverse_transform(testY)

# calculate root mean squared error
print'Prediction horizon = '+ str(lead_time),'Look back = ' + str(look_back)
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# make timestamp for unique filname
stamp = str(time.clock())  #add timestamp for unique name
stamp = stamp[0:2] 

# generate filename and remove extra periods
filename = 'FinErr_lstm_'+ str(n_epochs) + str(lead_time) + '_' + stamp + '.xlsx'    #example output file
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

# make coordinates for line to plot 
x = [0.,round(testY.max(),0)]

# plot baseline and predictions
plt.close('all')
plt.scatter(testY, testPredict)
plt.plot(x,x, color = 'r')
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.title("Prediction horizon = "+ str(lead_time) + "/Look back = " + str(look_back))
plt.show()
