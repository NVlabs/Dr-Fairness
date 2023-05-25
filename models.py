# ---------------------------------------------------------------
# Taken from the following link as is from:
# https://github.com/princetonvisualai/gan-debiasing
#
# The original version of this file can be
# found in https://github.com/princetonvisualai/gan-debiasing
# ---------------------------------------------------------------

"""
References:
[1] Ramaswamy et al., Fair Attribute Classification through Latent Space De-biasing, CVPR 2021
[2] https://github.com/princetonvisualai/gan-debiasing
"""

import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
from os import listdir, path, mkdir
from PIL import Image

from ema import EMA

class ResNet50(nn.Module):
    """ResNet50.
       
       We define ResNet50 as in Ramaswamy et al., CVPR 2021. 
       Details are in https://github.com/princetonvisualai/gan-debiasing
       
        Attributes: 
            resnet: A torch model containing ResNet50.

    """
    
    def __init__(self, n_classes=1, pretrained=True, hidden_size=2024, dropout=0.5): 
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, n_classes)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True
            
    def require_last_layer_grads(self):
        for module in self.resnet.modules():
            if module._get_name() != 'Linear':
                for param in module.parameters():
                    param.requires_grad_(False)
            elif module._get_name() == 'Linear':
                for param in module.parameters():
                    param.requires_grad_(True)


    def forward(self, x):
        outputs = self.resnet(x)

        return outputs, outputs
    

class attribute_classifier():
    """Attribute classifier.
       
       We define the attribute classifier as in Ramaswamy et al., CVPR 2021. 
       Details on the argumetns are in https://github.com/princetonvisualai/gan-debiasing

    """
    
    def __init__(self, device, dtype, n_classes=1, pretrained=True, modelpath = None, learning_rate = 1e-4, use_ema = False, ema_decay = 0.99, pretrain = False, domain = False):

        self.model = ResNet50(n_classes=n_classes, pretrained=True)
        self.model.require_all_grads()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if use_ema == True:
            self.optimizer = EMA(self.optimizer, ema_decay=ema_decay)
            
        self.device= device
        self.dtype = dtype
        self.epoch = 0
        self.best_acc = 0.0
        self.print_freq = 100
        
        self.model = self.model.to(device=device, dtype=dtype)
        
        if modelpath!=None:
            A = torch.load(modelpath, map_location='cpu') # map_location=device
            if domain == False:
                self.model.load_state_dict(A['model']) # torch.load(modelpath,map_location=device) )
            else:
                model_dict = self.model.state_dict()
                
                pretrained_dict = {k: v for k, v in A['model'].items() if
                       (k in model_dict) and (model_dict[k].shape == A['model'][k].shape)}
                model_dict.update(pretrained_dict) 
                self.model.load_state_dict(model_dict)
            
            if pretrain == False:
                self.optimizer.load_state_dict(A['optim'])
                self.epoch = A['epoch']
                
            
    def forward(self, x):
        out, feature = self.model(x)
        return out, feature
            
    def save_model(self, path, epoch):
        torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict(), 'epoch':epoch}, path)

    
    
def weights_init_normal(m):
    """Initializes the weight and bias of the model.

    Args:
        m: A torch model to initialize.

    Returns:
        None.
    """
    
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)
