#!/usr/bin/env python
# coding: utf-8

# ### In this notebook, we calculate the vector $c$ that used in the calculation of corresponding score.

# In[20]:


import numpy as np
import sys
sys.path.insert(1, 'C:/Users/zbxol/OneDrive/PSC/psc')
from confiance import conf
from confiance import cdf
import time
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead
from bokeh.transform import linear_cmap
from bokeh.palettes import RdYlGn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm

output_notebook()

## Define the model
class CorrScoreModel (nn.Module):
    '''
    The deep learning model used to calculate
    the corresponding score using the linear layer
    and LogSoftMax function.
    '''
    
    def __init__(self, dim_in=3, dim_out=2):
        super(CorrScoreModel, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out)
    
    def forward(self, feature):
        batch_size = len(feature)
        feature = torch.tensor(feature).view(batch_size, self.dim_in).cuda().float()
        return F.log_softmax(self.linear(feature), dim=1)


# In[38]:


## Define the function to train
def train_corr_score_model(dataset, dim_in=3, dim_out=2, batch_size=1280, num_epoch=100):
    # Generate train and test sets
    num_data = len(dataset)
    msk = np.random.rand(num_data) < 0.7
    train_set = dataset[msk]
    test_set = dataset[~msk]
    num_train = len(train_set)
    
    # Define the model
    model = CorrScoreModel(dim_in, dim_out)
    model.cuda() # to GPU
    func_loss = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    
    loss_list = []
    for epoch in tqdm(range(num_epoch)):
        loss_total = 0
        num_it = 0
        for i in tqdm(range(0, num_train, batch_size)):
            model.zero_grad()
            
            j = i+batch_size if i+batch_size<num_train else num_train
            X = train_set[i:j, :-1]
            Y = torch.tensor(train_set[i:j, -1]).cuda().long().view(j-i)
            
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
        train_Yp = model(train_set[:,:-1])
        test_Yp  = model(test_set[:, :-1])
        train_Y = torch.tensor(train_set[:,-1]).cuda().long().view(-1)
        test_Y  = torch.tensor(test_set[:, -1]).cuda().long().view(-1)
        
        err_train = func_loss(train_Yp, train_Y)
        err_test  = func_loss(test_Yp , test_Y )
        
        train_Yp = train_Yp.exp().cpu().numpy()
        test_Yp  = test_Yp.exp().cpu().numpy()
        train_Y = np.array(train_set[:,-1], dtype=int)
        test_Y = np.array(test_set[:,-1], dtype=int)
        
        train_Y_label = np.argmax(train_Yp, axis=1)
        test_Y_label  = np.argmax(test_Yp, axis=1)
        num_train_err = np.sum(train_Y_label != train_Y)
        num_test_err  = np.sum(test_Y_label != test_Y)
        train_err_rate = num_train_err / len(train_Y)
        test_err_rate  = num_test_err / len(test_Y)
        
        print('train err : %.2f, train err_rate : %.2f' % (err_train, train_err_rate))
        print('test  err : %.2f, test  err_rate : %.2f' % (err_test , test_err_rate ))
        print('Out put example :', train_Yp[-1], test_Y[-1])
        
    return model
    


# In[10]:


dataset = feature_list[:,1:]
print(dataset.shape)


# In[39]:


corr_score_model = train_corr_score_model(dataset)


# In[40]:


torch.save(corr_score_model.state_dict(), 'corr_score_model_state.pth')


# In[2]:


import numpy as np
a = np.random.rand(12)


# In[4]:


b = a.reshape((-1,3))
print(b.shape)


# In[ ]:




