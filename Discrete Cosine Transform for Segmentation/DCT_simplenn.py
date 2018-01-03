
# coding: utf-8

# In[1]:


from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[2]:


dataset='cifar10'
dataroot='f1'
workers=1
batchSize=5
imageSize=32
nz=400
ngf=64
ndf=64
niter=25
lr=0.0002
beta1=0.5
cuda=True
ngpu=1
netG=''
netD=''
outf='f1'
manualSeed=10
try:
    os.makedirs(outf)
except OSError:
    pass


# In[3]:


import os
I='hundred/'
ls1=os.listdir(I)
M='dct_3/'
ls2=os.listdir(M)

import scipy.misc as sm
import numpy as np
import scipy.io as si
names=[]
xx=400*3
X=np.zeros((len(ls2),xx))
Y=np.zeros((len(ls2),xx))
for ind,i in enumerate(ls2):
    names.append(i)
    li=len(i)
    temp=i[0:li-4]
    #temp=i.split('__')[0]
    tmp1=si.loadmat(M+i)['xx'][0:20,0:20,:]
    X[ind,:]=np.ravel(tmp1)
    temp2=sm.imread(I+temp+'.png')/255.0
    Y[ind,:]=np.ravel(temp2[0:20,0:20,:])
    
    print(ind)
    
    
    
    


# In[4]:


import sklearn.preprocessing as sp
#X=X/np.max(X)
X=sp.normalize(X)    


# In[ ]:





# In[5]:


inps=3*imageSize*imageSize


# In[6]:


def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=xx, kernel_initializer='normal', activation='relu'))
	model.add(Dense(xx, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# In[ ]:





# In[ ]:


model=larger_model()


# In[ ]:


model.fit(X, Y, epochs=150, batch_size=30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




