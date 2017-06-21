'''
LSTM classifier that cycles through a list of data files using a set number 
of epochs. Uses 2 each LSTM layers and a FF output layer. 
Dropout layers are added between 
Brian Freeman - 20 June 2017
'''
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Nadam
import numpy as np
import time
import pandas as pd
from pandas import ExcelWriter
import copy
import os

#Training data file names
data_files = ["outputDates.xlsx", "outputDatesO3.xlsx", "outputDatesFAH.xlsx",
              "outputDatesJAR.xlsx","outputDatesMAN.xlsx"]
timesteps = 12
newpath = 'C:\\\TARS\Phd\Keras\\'   #example output path
os.chdir(newpath)
activation_type = 'relu'
drop_out = .15

### - Load data from excel file function
def load_file(datafile,worksheet=0,type_data="Input Training"):
    #### Start the clock
    start = time.clock()   
    
    data_fil = pd.read_excel(datafile, 
                sheetname=worksheet, 
                header=0,         #assumes 1st row are header names
                skiprows=None, 
                skip_footer=0, 
                index_col=0,    #first column is index
                parse_cols=None, #default = None
                parse_dates=False, 
                date_parser=None, 
                na_values=None, 
                thousands=None, 
                convert_float=True, 
                has_index_names=None, 
                converters=None, 
                engine=None)
    
    data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
    
    # stop clock
    end = time.clock()    
    if (end-start > 60):
        print type_data, "data loaded in {0:.2f} minutes".format((end-start)/60.)
    else:
        print type_data, "data loaded in {0:.2f} seconds".format((end-start)/1.)
    return data

def ThreeDdata(data):
    '''#reshape a 2D dataset into a super 2D with time steps and then into a 3D matrix
    #where (total sample rows, delay, feature size)
    #a data set of 500 samples with 6 features (500 x 6 = 3,000 elements) that has 3 time inputs
    #t = 0, t = -1, t = -2 will first be transformed into a 497 x 9 matrix
    #and have a dimension of (497, 3, 6) or 8,946 elements - have to remove the last 
    #rows to keep dimensionality as you shift columns up
    '''
    n,p = np.shape(data)    # get dimensions of data matrix 
    a = np.zeros((n,p*timesteps))  # initialize a matrix with multiples based on total delay
    dataX = copy.copy(data[0:(n-timesteps),:])  #deep copy the data
    for i in range(1,timesteps):
        a = data[i:(n-timesteps+i), :]       # create delayed matrix for each increment
        dataX = np.concatenate((dataX, a), axis = 1)
    dataXre=np.reshape(dataX,((n-timesteps),timesteps,p))   
    return dataXre

############### Build RNN model
# expected input data shape: (batch_size, timesteps, data_dim)
def buildmodel(layer_out, data_dim, num_classes):
    model = Sequential()

    ######### create RNN model with 2 LSTM layers and 1 FF layer with dropout between layers
    # setup LSTM layers
    model.add(LSTM(layer_out, return_sequences=True,
               input_shape=(timesteps, data_dim),
               bias_initializer="zeros",
               activation=activation_type))  # returns a sequence of vectors of dimension 32
    model.add(Dropout(drop_out))
    model.add(LSTM(layer_out, activation=activation_type)) # return a single vector of dimension 32
    model.add(Dropout(drop_out))
    #add feed forward layer
    model.add(Dense(num_classes,
                use_bias = True, bias_initializer="zeros",
                activation='sigmoid'))

    optimizer_type = Nadam(lr=0.002, beta_1=0.9, 
                beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    model.compile(loss='binary_crossentropy',
              optimizer=optimizer_type,
              metrics=['accuracy'])
    return model

########## function to fit model with data based on epochs
def mod_fit(x_train, y_train, n_epochs):
    model.fit(x_train, y_train, batch_size=24, epochs=n_epochs, shuffle = False)
    out = (model.predict_proba(x_test) > .5) + 0.
    tot_err = np.absolute(out - y_test)
    return tot_err.sum(axis=0)

#########Select data source - Use random data for architecture development
epochs_text = raw_input('How many epochs? ')
n_epochs = int(epochs_text)
data_select = raw_input('Use random data (y) or imported data (n)? ')

if data_select == 'y':
    ######### Generate random dummy training data if yes
    data_dim = 13  #set dimensions for data
    num_classes = 5
    x_train = np.random.random((1000, timesteps, data_dim))
    y_train = (np.random.random((1000, num_classes)) > .5) + 0.
    layer_out = len(x_train.T) * 2
    
    x_test = np.random.random((100, timesteps, data_dim))
    y_test = (np.random.random((100, num_classes)) > .5) + 0.
    data_file_name = 'RANDRANDRAND'
    
    model = buildmodel(layer_out, data_dim, num_classes)  
    
    print"\nModel prepared using {0} samples with {1} features from file {2}".format(len(x_train),data_dim,data_file_name)
    print"\nThere are {0} outputs.\n".format(num_classes)

    #n_epochs = int(raw_input('Number of epochs: '))
    n_outputs = len(y_train.T)
    N = len(x_train)
    N_test = len(x_test)
    ################## Record error at different parameters

    tot_err_mod = mod_fit(x_train, y_train, n_epochs)
    tot_err_summary = tot_err_mod/N_test    
    tot_err_per = np.sum(tot_err_mod)/(N*n_outputs)
    print "\n Composite Error = {0:.2f}% after {1} epochs".format(tot_err_per * 100, n_epochs)

    cErr = np.squeeze(tot_err_summary)
    
    ### make extension for output file
    filename='RNNFinalRAND.xlsx'
    writer = ExcelWriter(filename)
    pd.DataFrame(cErr).to_excel(writer,'FinalError')
    writer.save()
    print'File saved in ', newpath + filename
    
else:
    #### Import data from file if No. Convert input data into 3D matrices
    #### Asks which files to use for training/testing - can select 1 or all
    
    print"\nWhich data file(s) would you like to use?"  # print list of files
    print"\n0"," - ","Select all files"
    for data_i in range (len(data_files)):
        print data_i + 1, " - file:", data_files[data_i]
    
    d_num = int(raw_input('Choose a number or select 0 for ALL: '))-1
    # a value of d_num = -1 represents all files to be used
    if abs(d_num) > len(data_files): #if out of range, use a default
        d_num = -1
        print"\nValue out of range. Using all data files."
    
    ### set parameters for file selection cycling
    if d_num == -1:  #if all files are selected, cycle through all files
        l_start = 0
        l_end = len(data_files)
        l_step = 1
    else:           #if only one file is selected, cycle only through one 
        l_start = d_num
        l_end = d_num-1
        l_step = -1
        
    bErr = []
    for data_i in range (l_start, l_end, l_step):
        
        data_file_name = data_files[data_i]
        print"\nUsing data in file {0}".format(data_file_name)
        
        '''
        x_train = ThreeDdata(load_file(data_file_name,0,"Input Training"))
        n_train = len(x_train)
        y_train = load_file(data_file_name,1,"Output Training")
        y_train = y_train[0:n_train,:] 
        layer_out = len(x_train.T) * 2
        
        x_test = ThreeDdata(load_file(data_file_name,4,"Input Test"))
        n_test = len(x_test)
        y_test = load_file(data_file_name,5,"Output Test") 
        y_test = y_test[0:n_test,:]

        data_dim = len(x_test.T) 
        num_classes = len(y_test.T)
        
        model = buildmodel(layer_out, data_dim, num_classes) 
        print"\nModel prepared using {0} samples with {1} features from file {2}".format(len(x_train),data_dim,data_file_name)
        print"\nThere are {0} outputs.\n".format(num_classes)

        #n_epochs = int(raw_input('Number of epochs: '))
        n_outputs = len(y_train.T)
        N = len(x_train)
        N_test = len(x_test)
        ################## Record error at different parameters

        tot_err_mod = mod_fit(x_train, y_train, n_epochs)
        tot_err_summary = tot_err_mod/N_test    
        tot_err_per = np.sum(tot_err_mod)/(N*n_outputs)
        print "\n Composite Error = {0:.2f}% after {1} epochs".format(tot_err_per * 100, n_epochs)
        cErr = tot_err_summary
        cErr = np.squeeze(cErr)
    
        ### make extension for output file
        fname=data_file_name[-9:-1]
        fname = fname[1:4]
        filename = 'RNNFinalError'+fname+'_3.xlsx'    #example output file
        writer = ExcelWriter(filename)
        pd.DataFrame(cErr).to_excel(writer,'FinalError')
        writer.save()
        print'File saved in ', newpath + filename
        '''
############## End program
