import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def call_bn(bn, x):
	return bn(x)

# class model(nn.Module):
# 	def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
# 		self.dropout_rate = dropout_rate
# 		self.top_bn = top_bn
# 		super(model, self).__init__()
# 		self.conv1=nn.Conv2d(input_channel,32,kernel_size=3,stride=1, padding=0)
# 		self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1, padding=0)
# 		self.bn1=nn.BatchNorm2d(32)
# 		self.bn2=nn.BatchNorm2d(64)
# 		flat_size = 144 * 64
# 		self.l_c1=nn.Linear(flat_size,128)
# 		self.l_c2=nn.Linear(128,n_outputs)
# 	def forward(self, x):
# 		h=x
# 		h=self.conv1(h)
# 		# print(h.size())
# 		h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
# 		# print(h.size())
# 		h=self.conv2(h)
# 		# print(h.size())
# 		h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
# 		h=F.max_pool2d(h, kernel_size=2, stride=2)
# 		# print(h.size())
# 		h=F.dropout2d(h, p=self.dropout_rate)
# 		h = h.view(h.size(0), -1)
# 		# print(h.size())
# 		h=self.l_c1(h)
# 		# print(h.size())
# 		h=F.leaky_relu(h, negative_slope=0.01)
# 		h=F.dropout2d(h, p=2*self.dropout_rate)
# 		logit=self.l_c2(h)
# 		# input()

# 		return logit

class model(nn.Module):
    def __init__(self, image_size, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(model, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        
        features=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)

        h=F.avg_pool2d(features, kernel_size=features.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        if self.top_bn:
            logit=call_bn(self.bn_c1, logit)
        return logit, features