'''
This file defines a linear classify model.
'''

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class linear_model(nn.Module):
    
    def __init__(self, dim_in = 26*15, dim_out = 2, dim_hid = [258, 128], use_gpu = True):
        super(linear_model, self).__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        assert len(dim_hid) == 2
        #assert isinstance(dim_hid, list)
        #self.dim_hid_list = [dim_in] + dim_hid + [dim_out]
        #self.num_layer = len(self.dim_hid_list)-1
        
        if use_gpu :
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else :
            self.device = torch.device('cpu')
        
        #self.linear_list = []
        #for i in range(self.num_layer):
        #    self.linear_list.append(nn.Linear(self.dim_hid_list[i],self.dim_hid_list[i+1]))
        self.linear_1 = nn.Linear(self.dim_in, self.dim_hid[0])
        self.linear_2 = nn.Linear(self.dim_hid[0], self.dim_hid[1])
        self.linear_3 = nn.Linear(self.dim_hid[1], self.dim_out)

    # Overwrite the forward function
    def forward(self,X):
        num_batch = len(X)
        X = torch.tensor(X).view(num_batch, self.dim_in).to(self.device).float()
        X_tmp = X
        #for i in (self.num_layer-1):
        #    linear = self.linear_list[i]
        #    X_tmp = F.ReLu(linear(X_tmp))
        X_tmp = F.relu(self.linear_1(X_tmp))
        X_tmp = F.relu(self.linear_2(X_tmp.view(num_batch, self.dim_hid[0])))
        X_tmp = self.linear_3(X_tmp.view(num_batch, self.dim_hid[1]))

        X_tmp = F.log_softmax(X_tmp, dim=1)
        return X_tmp
    
def train(dataset, dim_in, dim_out, dim_hid, use_gpu, num_epoch = 64, batch_size = 64):
#     dataset是一个list，前一项n*26，后一项0或1
    
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
    
    # Define the model
    model = linear_model(dim_in = dim_in, dim_out = dim_out, dim_hid = dim_hid, use_gpu = use_gpu)
    
    # Train the model
    if use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else :
        device = torch.device('cpu')
    print('Using ', device)
    model.to(device)
    func_loss = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    
    data_X = X_list_train
    data_Y = Y_list_train
    # Now do the training
    loss_list = []
    for epoch in tqdm(range(num_epoch)):
        loss_total = 0
        num_it = 0
        for i in tqdm(range(0, num_train, batch_size)):
            model.zero_grad()

            j = i+batch_size if i+batch_size<num_train else num_train
            X = data_X[i:j]
            Y = torch.tensor(data_Y[i:j]).to(device).long().view(j-i)

            prob_log = model(X)
            loss = func_loss(prob_log, Y)
            loss.backward()
            optimizer.step()
            
            loss_total += loss
            num_it += 1
        
        scheduler.step()
        loss_mean = loss_total/num_it
        loss_list.append(loss_mean)
        print('Epoch %4d/%4d, loss : %.3f' % (epoch,num_epoch,loss_mean))
        
    plt.plot(loss_list)
    plt.legend(['loss'])
    
    # Then show the result
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
    
    model.cpu()
    return model
    
    
# vector a
def predict(a, model):
    
    with torch.no_grad():
        prob = model(a).exp()
    if (prob[0]<prob[1]):
        return 1
    else:
        return 0
    
        