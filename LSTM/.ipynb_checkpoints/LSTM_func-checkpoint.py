import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.cuda.is_available())

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from bokeh.plotting import figure, show
from bokeh.models import Title
from bokeh.io import output_notebook
from bokeh.layouts import row

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from LSTM.LSTM_fit import LSTM_Fit


def gen_dataset(data_set, look_back=10, dim_out=1):
    '''
    Function used to generate the dataset for training.
    '''
    data_X = []
    data_Y = []
    
    for i in range(len(data_set)-look_back-1):
        data_X.append(data_set[i:i+look_back, :-dim_out])
        data_Y.append(data_set[i+look_back, -dim_out:])
    #print('X : %s, Y : %s' % (data_X[-1], data_Y[-1])) # Show the form of (x,y).
    return np.array(data_X), np.array(data_Y)


def cut_dataset(data_set, look_back=10):
    if len(data_set) < look_back:
        raise('Err, len(data_set < look_back)')
        
    data_X = [data_set[i:i+look_back] for i in range(len(data_set)-look_back)]
    return data_X


def LSTM_train(dataset, dim_in=26, dim_out=2, look_back=10, batch_size=1280, num_epoch=50, bokeh=False, use_gpu=True):
    
    # Scaler
    #scaler = MinMaxScaler()
    #data_XY = scaler.fit_transform(dataset)
    
    # Split the datas
    #data_X, data_Y = gen_dataset(data_XY, look_back, dim_out)
    #num_data = len(data_X)
    #num_train = int(2*num_data/3)
    # Generate data    
    X_list = [data[0] for data in dataset]
    Y_list = [data[1] for data in dataset]
    X_list = np.array(X_list)
    Y_list = np.array(Y_list)
    # Seperate train data from test data.
    msk = (np.random.rand(len(Y_list)) < 0.7)
    X_list_train = X_list[msk]
    Y_list_train = Y_list[msk]
    X_list_test = X_list[~msk]
    Y_list_test = Y_list[~msk]
    num_train = len(X_list_train)
    
    data_X = X_list_train
    data_Y = Y_list_train
    
    if use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else :
        device = torch.device('cpu')
    print('Using ', device) 
    
    # Define the model
    model = LSTM_Fit(look_back, dim_in, dim_out, use_gpu=True)
    model.to(device) # To GPU
    #func_loss = nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    func_loss = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
        
       
    # Now do the trainning.
    loss_list = []
    for epoch in tqdm(range(num_epoch)):
        loss_total = 0
        num_it = 0
        for i in tqdm(range(0, num_train, batch_size)):
            model.zero_grad()
            #model.init_hid()
            
            # get the batched datas.
            # The upper bound.
            j = i+batch_size if i+batch_size<num_train else num_train
            X = data_X[i:j]
            Y = torch.tensor(data_Y[i:j]).to(device).long().view(j-i)
            
            Y_p = model(X)
            #Y_p = F.log_softmax(Y_p, dim=1)
            loss = func_loss(Y_p, Y)
            loss.backward()
            optimizer.step()
            loss_total += loss
            num_it += 1
            
            
        loss_mean = loss_total/num_it
        loss_list.append(loss_mean)
        print('Epoch : %4d/%4d, loss : %.3f' % (epoch, num_epoch, loss_mean))
        
    
    # Then calculate the predcited values.
    with torch.no_grad():
        train_Yp = model(X_list_train)
        test_Yp  = model(X_list_test)
        train_Y = torch.tensor(Y_list_train).to(device).long().view(-1)
        test_Y  = torch.tensor(Y_list_test).to(device).long().view(-1)
        
        err_train = func_loss(train_Yp, train_Y)
        err_test  = func_loss(test_Yp , test_Y )
        
        train_Yp = train_Yp.exp().cpu().numpy()
        test_Yp  = test_Yp.exp().cpu().numpy()
        train_Y = np.array(Y_list_train, dtype=int)
        test_Y = np.array(Y_list_test, dtype=int)
        
        train_Y_label = np.argmax(train_Yp, axis=1)
        test_Y_label  = np.argmax(test_Yp, axis=1)
        num_train_err = np.sum(train_Y_label != train_Y)
        num_test_err  = np.sum(test_Y_label != test_Y)
        train_err_rate = num_train_err / len(train_Y)
        test_err_rate  = num_test_err / len(test_Y)
        
        print('train err : %.2f, train err_rate : %.2f' % (err_train, train_err_rate))
        print('test  err : %.2f, test  err_rate : %.2f' % (err_test , test_err_rate ))
        print('Out put example :', train_Yp[-1], test_Y[-1])
    
    return model#, scaler
    
    
def model_pred(model, scaler, dataA):
    dim_in = dataA.shape[1]
    # Check dim in:
    if dim_in != model.dim_in:
        raise('Err, the dimension of given data does not match with input dim of the model.')
    # find the required num_col of dataset.
    dim_out = scaler.min_.shape[0]-dim_in
    if dim_out != model.dim_out:
        raise('Err, the dimension of scaler does not match with input dim of the model.')
        
    look_back = model.look_back
    # pad the data.
    dataset = np.column_stack([dataA, np.zeros((len(dataA),dim_out))])
    # scale the data
    dataXY = scaler.transform(dataset)
    # seperate data X and Y
    dataX = cut_dataset(dataXY[:,:dim_in], look_back)
    # precess the model
    with torch.no_grad():
        dataY_p = model(dataX).cpu().numpy()
        
    # combine the predicted data.
    dataXY_p = np.column_stack([dataXY[look_back:, :dim_in], dataY_p])
    # Reverse
    data_pred = scaler.inverse_transform(dataXY_p)
    # Seperate the predicted data.
    tar_pred = data_pred[:, -dim_out:]
    
    return tar_pred