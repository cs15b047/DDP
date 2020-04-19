import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class model(nn.Module):
    def __init__(self, image_size, input_channel=3, n_outputs=14, dropout_rate=0.00, top_bn=False):
        self.dropout_rate = dropout_rate
        super(model, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.l_c1=nn.Linear(512,n_outputs)
        ################Freeze backbone###########
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        ##########################################
        self.backbone.fc = self.l_c1
        self.useful_part = nn.Sequential(*list(self.backbone.children())[:-2])
        # print(list(self.backbone.parameters()))
        # input()

    def forward(self, x):
        # h = self.useful_part(x)
        # h = h.view(h.size(0), h.size(1))
        # h=F.dropout(h, p=self.dropout_rate) ##########Hyperparam ---> dropout#############
        # logit = self.l_c1(h)
        # return logit
        features_vis = self.useful_part(x)
        x = self.backbone.avgpool(features_vis)
        features = x.view(x.size(0), -1)
        logits = self.backbone.fc(features)
        return logits, features, features_vis
