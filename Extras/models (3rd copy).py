from __future__ import print_function

import torch
import torch.nn as nn
import torch.legacy.nn as nn2
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# For input size input_nc x 256 x 256
class G(nn.Module):
    def __init__(self, input_nc, output_nc, ngf):
        super(G, self).__init__()
        self.conv1 = nn2.SpatialDilatedConvolution(input_nc, ngf, 5, 5,        1 , 1, 2 , 2, 1, 1)
        self.conv2 = nn2.SpatialDilatedConvolution(ngf, ngf*2, 5, 5,        1 , 1, 4 , 4  , 2, 2)
        self.conv3 = nn2.SpatialDilatedConvolution(ngf*2, ngf*4, 5, 5,        1 , 1, 6 , 6, 3, 3)
        self.conv4 = nn2.SpatialDilatedConvolution(ngf*4, 3, 5, 5,        1 , 1, 6 , 6, 3, 3)

        
        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ngf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ngf * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        '''
        e1 = self.conv1.updateOutput(input.data.cpu())
        e1=Variable(e1.cuda())
        #print (e1.size())

        v1=   self.leaky_relu(e1)
        a1=self.conv2.updateOutput(  v1.data.cpu())
        # state size is (ngf) x 128 x 128
        e2 = self.batch_norm2(   Variable(a1.cuda()) )

        #e2=Variable(e2.cuda())
        #print (e2.size())

        
        v1=   self.leaky_relu(e2)
        a1=self.conv3.updateOutput(  v1.data.cpu())
        # state size is (ngf) x 128 x 128
        e3 = self.batch_norm4(   Variable(a1.cuda()) )

        #e2=Variable(e2.cuda())
        #print (e3.size())
        v1=   self.leaky_relu(e3)
        a1=self.conv4.updateOutput(  v1.data.cpu())
        # state size is (ngf) x 128 x 128
        e4 =  Variable(a1.cuda()) 
        #print (e3.size())

        
        

        # state size is (ngf x 2) x 64 x 64
        # state size is (ngf x 4) x 32 x 32
        #print (e8)

        output = self.tanh(e4)
        #print (output.size())
        return output
        '''
        e1 = self.conv1.updateOutput(input.data.cpu())
        e1=Variable(e1)
        #print (e1.size())

        v1=   self.leaky_relu(e1)
        a1=self.conv2.updateOutput(  v1.data.cpu())
        # state size is (ngf) x 128 x 128
        e2 = self.batch_norm2(   Variable(a1  ) )

        #e2=Variable(e2.cuda())
        #print (e2.size())

        
        v1=   self.leaky_relu(e2)
        a1=self.conv3.updateOutput(  v1.data.cpu())
        # state size is (ngf) x 128 x 128
        e3 = self.batch_norm4(   Variable(a1) )

        #e2=Variable(e2.cuda())
        #print (e3.size())
        v1=   self.leaky_relu(e3)
        a1=self.conv4.updateOutput(  v1.data.cpu())
        # state size is (ngf) x 128 x 128
        e4 =  Variable(a1) 
        #print (e3.size())

        
        

        # state size is (ngf x 2) x 64 x 64
        # state size is (ngf x 4) x 32 x 32
        #print (e8)

        output = self.tanh(e4)
        #print (output.size())
        return output
        

class D(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(input_nc + output_nc, ndf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)
        self.conv5 = nn.Conv2d(ndf * 8, 1, 4, 1, 1)

        self.batch_norm2 = nn.BatchNorm2d(ndf * 2)
        self.batch_norm4 = nn.BatchNorm2d(ndf * 4)
        self.batch_norm8 = nn.BatchNorm2d(ndf * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input is (nc x 2) x 256 x 256
        h1 = self.conv1(input)
        # state size is (ndf) x 128 x 128
        h2 = self.batch_norm2(self.conv2(self.leaky_relu(h1)))
        # state size is (ndf x 2) x 64 x 64
        h3 = self.batch_norm4(self.conv3(self.leaky_relu(h2)))
        # state size is (ndf x 4) x 32 x 32
        h4 = self.batch_norm8(self.conv4(self.leaky_relu(h3)))
        # state size is (ndf x 8) x 31 x 31
        h5 = self.conv5(self.leaky_relu(h4))
        # state size is (ndf) x 30 x 30, corresponds to 70 x 70 receptive
        output = self.sigmoid(h5)
        return output
