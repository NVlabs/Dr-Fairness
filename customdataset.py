# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for Dr-Fairness. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import sys, os
import numpy as np
import math
import random
import itertools
import copy
from PIL import Image

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch

from torchvision import transforms

import blobfile as bf


class CelebaDataset(Dataset):
    """Custom Dataset for default data loading on CelebA.

    Attributes:
        list_IDs: A list that contains the path of images.
        y: A dictionary for y features (label attributes) of data.
        z: A dictionary for z features (group attributes) of data.
        transform: A function for image transformation.
    """
    def __init__(self, list_IDs, y_tensor, z_tensor, transform=transforms.ToTensor()):
        """Initializes the dataset with torch tensors."""
        
        self.y = y_tensor
        self.z = z_tensor
        
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Returns the selected data based on the index information."""

        ID = self.list_IDs[index]
        img = Image.open(ID)
        X = self.transform(img)
        y = self.y[ID]
        z = self.z[ID]

        return X, y, z

class CelebaDataset_Adaptive(Dataset):
    """Custom Dataset for adaptive data loading (e.g., Dr-Fairness) on CelebA.

    Attributes:
        list_IDs: A list that contains the path of images.
        y: A dictionary for y features (label attributes) of data.
        z: A dictionary for z features (group attributes) of data.
        transform: A function for image transformation.
    """
    def __init__(self, list_IDs, y_tensor, z_tensor, transform=transforms.ToTensor()):
        """Initializes the dataset with torch tensors."""
        
        self.y = y_tensor
        self.z = z_tensor
        
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        index = [int (i) for i in index]
        ID = np.array(self.list_IDs)[index]
        
        X_total = []
        y_total = []
        z_total = []
        for j, id_tmp in enumerate(ID):
            img = Image.open(id_tmp)
            X = self.transform(img)
            y = self.y[id_tmp]
            z = self.z[id_tmp]
            
            X_total.append(X.numpy())
            y_total.append(y.numpy())
            z_total.append(z.numpy())
            
        return torch.Tensor(X_total), torch.Tensor(y_total), torch.Tensor(z_total)
    
    
class CustomDataset(Dataset):
    """Custom Dataset.

    Attributes:
        x: A PyTorch tensor for x features of data.
        y: A PyTorch tensor for y features (label attributes) of data.
        z: A PyTorch tensor for z features (group attributes) of data.
    """
    def __init__(self, x_tensor, y_tensor, z_tensor):
        """Initializes the dataset with torch tensors."""
        
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        
    def __getitem__(self, index):
        """Returns the selected data based on the index information."""
        
        return (self.x[index], self.y[index], self.z[index])

    def __len__(self):
        """Returns the length of data."""
        
        return len(self.x)
    

