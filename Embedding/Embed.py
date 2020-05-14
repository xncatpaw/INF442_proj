'''
Define the embedding class to classify the sequence.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
#%matplotlib inline

HAS_TQDM = True
try:
    from tqdm import tqdm_notebook as tqdm
except ImportError:
    HAS_TQDM = False



class Embed_Model(nn.Module):
    '''
    The embed model used to classify.
    '''
    def __init__(self, size_contxt=15, size_voc=26, dim_emb=8, dim_hid=64, dim_out=2, use_gpu=True):
        '''
        Constructor.
        - Param(s):
            size_contxt : int, the size of a word sequence. Optional, default is 15.
            size_voc :    int, the size of vocabulary. Optional, default is 26.
            dim_emb  :    int, the embedding dimension. Optional, default is 8.
            dim_hid  :    int, the hidden layer dimension. Optional, default is 64.
            dim_out  :    int, the output dimension. Optional, default is 2.
            use_gpu  :    bool, whether use gpu or not in this model. Optional, default is True.
        '''
        super(Embed_Model, self).__init__()
        self.size_contxt = size_contxt
        self.size_voc = size_voc
        self.dim_emb = dim_emb
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        
        if use_gpu :
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else :
            self.device = torch.device('cpu')
        print('Using device', self.device, '.')
            
        # Embedding layer.
        self.embed = nn.Embedding(self.size_voc, self.dim_emb)
        self.linear_1 = nn.Linear(self.size_contxt*self.dim_emb, self.dim_hid)
        self.linear_2 = nn.Linear(self.dim_hid, self.dim_out)
        
    
    def forward(self, inputs):
        '''
        Forward function of the model.
        - Param(s):
            inputs : Input values. Shall of the form (num_sample * size_contxt), 
                     i.e., [[word_ind_0, word_ind_1, ..., word_ind_size_contxt], ...]
        '''
        # get the batch size.
        batch_size = len(inputs)
        # Reshap and convert to tensor.
        X = torch.tensor(inputs).view(batch_size, self.size_contxt).long().to(self.device)
        # The embedding layer.
        embds = self.embed(X).view((batch_size,-1))
        # 1st linear layer.
        out = F.relu(self.linear_1(embds))
        # 2nd linear layer.
        out = self.linear_2(out).view((batch_size,self.dim_out))
        # log soft max.
        log_probs = F.log_softmax(out, dim=1)
        
        return log_probs
    
    
    
    
    def train(self, dataset, batch_size=1280, num_epoch=50, random=True, print_loss=False):
        '''
        Train this model using the given dataset.
        - Parma(s):
            dataset : List/np.ndarray, the dataset, of shape (num_sample * size_contxt).
            batch_size : int, batch size. Optional, default is 1280.
            num_epoch  : int, the number of epoches. Optional, default is 50.
        '''
        X_list = [data[0] for data in dataset]
        Y_list = [data[1] for data in dataset]
        X_list = np.array(X_list, dtype=int)
        Y_list = np.array(Y_list, dtype=int)
        num_data = len(dataset)
        # Seperate the train data from test data.
        if random:
            msk = (np.random.rand(num_data)<0.7)
            X_list_train = X_list[msk]
            Y_list_train = Y_list[msk]
            X_list_test  = X_list[~msk]
            Y_list_test  = Y_list[~msk]
            num_train = len(X_list_train)
        else :
            num_train = int(num_data*2/3)
            X_list_train = X_list[:num_train]
            Y_list_train = Y_list[:num_train]
            X_list_test  = X_list[num_train:]
            Y_list_test  = Y_list[num_train:]
            
        data_X = X_list_train
        data_Y = Y_list_train
        
        self.to(self.device)
        
        func_loss = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1.3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Train
        loss_list = []
        it_epoch = tqdm(range(num_epoch)) if HAS_TQDM else range(num_epoch)
        for epoch in it_epoch:
            loss_total = 0
            num_it = 0
            it_train_loop = range(0, num_train, batch_size)
            #if HAS_TQDM:
            #    it_train_loop = tqdm(it_train_loop)
            
            for i in it_train_loop:
                self.zero_grad()
                
                # Batched data
                j = i+batch_size if i+batch_size<num_train else num_train
                X = data_X[i:j]
                Y = torch.tensor(data_Y[i:j]).to(self.device).long().view(j-i)
                
                Y_p = self(X)
                loss = func_loss(Y_p, Y)
                loss.backward()
                optimizer.step()
                loss_total += loss
                num_it += 1
        
            scheduler.step()
            loss_mean = loss_total/num_it
            loss_list.append(loss_mean)
            if print_loss:
                print('Epoch : %4d/%4d, loss : %.3f.' % (epoch, num_epoch, loss_mean))
        plt.plot(loss_list)
        plt.legend(['loss'])
        plt.show()
        
        # After the training, calculate the test err.
        with torch.no_grad():
            train_Yp = self(X_list_train)
            test_Yp  = self(X_list_test)
            train_Y = torch.tensor(Y_list_train).to(self.device).long().view(-1)
            test_Y  = torch.tensor(Y_list_test ).to(self.device).long().view(-1)
            
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
            print('Out put example :', train_Yp[-1], train_Y_label[-1], train_Y[-1])
