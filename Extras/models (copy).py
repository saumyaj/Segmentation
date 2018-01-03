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
        self.conv1 = nn2.SpatialDilatedConvolution(input_nc, ngf, 6, 6,        1 , 1, 0 , 0, 2, 2)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 5, 2, 0)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 6, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 6, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 6, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 6, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 6, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 6, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 6, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 7, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 7, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 8, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 11, 1, 0)

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
        e1 = self.conv1.updateOutput(input.data.cpu())
        e1=Variable(e1.cuda())


        # state size is (ngf) x 128 x 128
        e2 = self.batch_norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 64 x 64
        e3 = self.batch_norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 32 x 32
        e4 = self.batch_norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 16 x 16
        # state size is (ngf x 8) x 2 x 2
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e4))
        #print (e8)

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(e8))))
        # state size is (ngf x 8) x 2 x 2
        #print (('d1', d1_.size()))
        #print (('e4', e4.size()))
        d1 = torch.cat((d1_, e4), 1)
        d5_ = self.batch_norm4(self.dconv5(self.relu(d1)))
        #print (('d5', d5_.size()))
        #print (('e3', e3.size()))
        # state size is (ngf x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.batch_norm2(self.dconv6(self.relu(d5)))
        # state size is (ngf x 2) x 64 x 64
        #print (('d6', d6_.size()))
        #print (('e2', e2.size()))
        d6 = torch.cat((d6_, e2), 1)
        d7_ = (self.dconv7(self.relu(d6)))
        #print (('d7', d7_.size()))
        #print (('e1', e1.size()))

        # state size is (ngf) x 128 x 128
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.relu(d7))
        # state size is (nc) x 256 x 256
        output = self.tanh(d8)
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
