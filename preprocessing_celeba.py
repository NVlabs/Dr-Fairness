# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for Dr-Fairness, except for functions img_transform() and create_dataset_real_imgs(). 
# To view a copy of this work's license, see the LICENSE file.
#
# The original version of functions img_transform() 
# and create_dataset_real_imgs() can be found in
# https://github.com/princetonvisualai/gan-debiasing
# ---------------------------------------------------------------

"""
References:
[1] Ramaswamy et al., Fair Attribute Classification through Latent Space De-biasing, CVPR 2021
[2] https://github.com/princetonvisualai/gan-debiasing
"""

import sys, os
import numpy as np
import math
import random
import itertools
import copy
from PIL import Image
import pickle
from copy import deepcopy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch

from torchvision import transforms

def img_transform(augment = False):
    """Defines the image transformation.
       We utilize the image transformation function used in Ramaswamy et al., CVPR 2021. 
       Details are in https://github.com/princetonvisualai/gan-debiasing
       
        Args: 
            augment: A boolean indicating data augmentation.
        
        Returns:
            Image transformation.
    """
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    if augment:
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    return transform


def get_all_attr(gen = False):
    """Gets all attribute names in either real or generated dataset.
        Args: 
            gen: A boolean indicating whether the target dataset is real or generated.
        
        Returns:
            All attribute names.
    """
    
    orig_attr = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',  'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    if gen == False:
        return orig_attr
    else:
        new_attr = orig_attr[:]
        new_attr[39] = 'age'
        new_attr[20] = 'gender'
        new_attr[31] = 'smiling'
        new_attr[15] = 'glasses'
        return new_attr
        
        
def get_attr_index(y_name, z_name):
    """Gets attribute indeces in the real and generated datasets.
        Args: 
            y_name: A string indicating the label attribute.
            z_name: A string indicating the group attribute.
        
        Returns:
            Attribute indeces in the real and generated datasets
    """
    
    # Changed some attr names for matching with generated attr names
    orig_attr = get_all_attr(gen = True)
    gen_attr = ['age', 'gender', 'smiling', 'glasses', 'haircolors']
    
    for idx, tmp_name in enumerate(orig_attr):
        if y_name == tmp_name:
            orig_y_idx = idx
        elif y_name == 'haircolors':
            orig_y_idx = [8, 9]
            
        if z_name == tmp_name:
            orig_z_idx = idx
        elif z_name == 'agegender':
            orig_z_idx = [39, 20]
            
    for idx, tmp_name in enumerate(gen_attr):
        if y_name == tmp_name:
            gen_y_idx = idx
            
        if z_name == tmp_name:
            gen_z_idx = idx
        elif z_name == 'agegender':
            gen_z_idx = [0, 1]
            
    return orig_y_idx, orig_z_idx, gen_y_idx, gen_z_idx

def get_yz_index(list_ids, y_attr, z_attr):
    """Gets (y, z)-class-wise information in the dataset (e.g., (y, z)-index).
        Args: 
            list_ids: A list that contains the path of images.
            y_attr: A dictionary for y features (label attributes) of data.
            z_attr: A dictionary for z features (group attributes) of data.
        
        Returns:
            (y, z)-class-wise information in the dataset.
    """
    
    y_value = []
    z_value = []
    for tmp_id in list_ids:
        y_value.append(y_attr[tmp_id])
        z_value.append(z_attr[tmp_id])

    # y_value = [value[0] == 1 for value in y_attr.values()]
    y_data = np.array(y_value).astype(int)

    # z_value = [value[0] == 1 for value in z_attr.values()]
    z_data = np.array(z_value).astype(int)


    # Takes the unique values of the tensors
    z_item = list(set(z_data.tolist()))
    y_item = list(set(y_data.tolist()))

    yz_tuple = list(itertools.product(y_item, z_item))

    # Makes masks
    z_mask = {}
    y_mask = {}
    yz_mask = {}

    for tmp_z in z_item:
        z_mask[tmp_z] = (z_data == tmp_z)

    for tmp_y in y_item:
        y_mask[tmp_y] = (y_data == tmp_y)

    for tmp_yz in yz_tuple:
        yz_mask[tmp_yz] = (y_data == tmp_yz[0]) & (z_data == tmp_yz[1])

    # Finds the index
    z_index = {}
    y_index = {}
    yz_index = {}

    for tmp_z in z_item:
        z_index[tmp_z] = (z_mask[tmp_z] == 1).nonzero()[0]

    for tmp_y in y_item:
        y_index[tmp_y] = (y_mask[tmp_y] == 1).nonzero()[0]

    for tmp_yz in yz_tuple:
        yz_index[tmp_yz] = (yz_mask[tmp_yz] == 1).nonzero()[0]

    # Length information
    z_len = {}
    y_len = {}
    yz_len = {}

    for tmp_z in z_item:
        z_len[tmp_z] = len(z_index[tmp_z])

    for tmp_y in y_item:
        y_len[tmp_y] = len(y_index[tmp_y])

    for tmp_yz in yz_tuple:
        yz_len[tmp_yz] = len(yz_index[tmp_yz])
    
    other_info = {"yz_tuple": yz_tuple, "y_len": y_len, "num_z": len(z_item)}
    
    return yz_index, yz_len, other_info

def get_gen_info(attribute, protected_attribute):
    """Gets the label and group information, including the number of classes and label and group names.
        Args: 
            attribute: A integer or a list indicating the label attribute.
            protected_attribute: A integer or a list indicating the group attribute.
        
        Returns:
            The label and group information, including the number of classes and label and group names.
    """
    
    all_attr = ['age', 'gender', 'smiling', 'glasses', 'haircolors']
    
    y_name = all_attr[attribute]
    
    if type(protected_attribute) == list:
        z_name = []
        for i in protected_attribute:
            z_name.append(all_attr[i])
    else:    
        z_name = all_attr[protected_attribute]
    
    y_n_class = 0
    z_n_class = 0
    
    y_names = []
    z_names = []
    
    if z_name == 'gender':
        z_n_class = 2
        z_names = ['female', 'male']
        
        if y_name == 'age':
            y_n_class = 2
            y_names = ['old', 'young']
        elif y_name == 'glasses':
            y_n_class = 2
            y_names = ['noglasses', 'glasses']
        elif y_name == 'haircolors':
            y_n_class = 3
            y_names = ['blackhair', 'blondhair', 'otherhair']
    elif ('gender' in z_name) and ('age' in z_name):
        z_n_class = 4
        z_names = [['old', 'female'], ['old', 'male'], ['young', 'female'], ['young', 'male']]
        
        if y_name == 'glasses':
            y_n_class = 2
            y_names = ['noglasses', 'glasses']
        elif y_name == 'haircolors':
            y_n_class = 3
            y_names = ['blackhair', 'blondhair', 'otherhair']
    
    y_item = np.arange(0, y_n_class)
    z_item = np.arange(0, z_n_class)
    yz_tuple = list(itertools.product(y_item, z_item))
    
    
    return yz_tuple, y_names, z_names
    

def create_dataset_real_imgs(path, attribute, protected_attribute, augment=False, number=0, split='train'):
    """Preprocess the real data. 
       We follow the pre-processing used in Ramaswamy et al., CVPR 2021.
       Details are in https://github.com/princetonvisualai/gan-debiasing
        
        Args: 
            path: A string indicating the path of generated data.
            attribute: A integer or a list indicating the label attribute.
            protected_attribute: A integer or a list indicating the group attribute.
            augment: A boolean indicating data augmentation.
            number: An integer indicating the default index.
            split: A string indicating whether the dataset is for training, validation, or testing.
        
        Returns:
            Information for creating a dataset.
    """
    
    img_path = path+'/img_align_celeba/'
    attr_path = path+'/list_attr_celeba.txt'
    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637
    
    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    y_attr = {}
    z_attr = {}
    for i in range(beg+2, beg+number+2):
        temp = label[i].strip().split()
        list_ids.append(img_path+temp[0])
        y_attr[img_path+temp[0]]=torch.Tensor(([(int(temp[attribute+1])+1)/2]))
        z_attr[img_path+temp[0]]=torch.Tensor(([(int(temp[protected_attribute+1])+1)/2]))


    return list_ids, y_attr, z_attr


def create_dataset_real_imgs_haircolors(path, protected_attribute = 20, augment=False, number=0, split='train'):
    """Preprocess the real data. 
       A specialized version of the function 'def create_dataset_real_imgs()' for haircolor classification.
        
        Args: 
            path: A string indicating the path of generated data.
            protected_attribute: A integer or a list indicating the group attribute.
            augment: A boolean indicating data augmentation.
            number: An integer indicating the default index.
            split: A string indicating whether the dataset is for training, validation, or testing.
        
        Returns:
            Information for creating a dataset.
    """
    
    if protected_attribute == [39, 20]:
        return create_dataset_real_imgs_agegender_haircolors(path, augment=augment, number=number, split=split)
        
    img_path = path+'/img_align_celeba/'
    attr_path = path+'/list_attr_celeba.txt'
    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637
    
    black_attribute = 8
    blond_attribute = 9
    
    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    y_attr = {}
    z_attr = {}
    for i in range(beg+2, beg+number+2):
        temp = label[i].strip().split()
        list_ids.append(img_path+temp[0])
        hair_attr_tmp = 2
        if int(temp[black_attribute+1]) == 1:
            hair_attr_tmp = 0
        elif int(temp[blond_attribute+1]) == 1:
            hair_attr_tmp = 1

        y_attr[img_path+temp[0]]=torch.Tensor(([hair_attr_tmp]))
        z_attr[img_path+temp[0]]=torch.Tensor(([(int(temp[protected_attribute+1])+1)/2]))

    return list_ids, y_attr, z_attr


def create_dataset_real_imgs_agegender_haircolors(path, augment=False, number=0, split='train'):
    """Preprocess the real data. 
       A specialized version of the function 'def create_dataset_real_imgs()' 
       for haircolor classification with the (age+gender) group attribute.
        
        Args: 
            path: A string indicating the path of generated data.
            augment: A boolean indicating data augmentation.
            number: An integer indicating the default index.
            split: A string indicating whether the dataset is for training, validation, or testing.
        
        Returns:
            Information for creating a dataset.
    """
    
    
    img_path = path+'/img_align_celeba/'
    attr_path = path+'/list_attr_celeba.txt'
    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637
    
    age_attribute = 39
    gender_attribute = 20
    
    black_attribute = 8
    blond_attribute = 9
    
    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    y_attr = {}
    z_attr = {}
    for i in range(beg+2, beg+number+2):
        temp = label[i].strip().split()
        list_ids.append(img_path+temp[0])
        
        hair_attr_tmp = 2
        if int(temp[black_attribute+1]) == 1:
            hair_attr_tmp = 0
        elif int(temp[blond_attribute+1]) == 1:
            hair_attr_tmp = 1
            
        if (int(temp[age_attribute+1]) == -1) and (int(temp[gender_attribute+1]) == -1):
            z_attr_tmp = 0
        elif (int(temp[age_attribute+1]) == -1) and (int(temp[gender_attribute+1]) == 1):    
            z_attr_tmp = 1
        elif (int(temp[age_attribute+1]) == 1) and (int(temp[gender_attribute+1]) == -1):
            z_attr_tmp = 2
        else:
            z_attr_tmp = 3
            
        y_attr[img_path+temp[0]]=torch.Tensor(([hair_attr_tmp]))
        z_attr[img_path+temp[0]]=torch.Tensor(([z_attr_tmp]))

    return list_ids, y_attr, z_attr



def create_dataset_gen_imgs(path, attribute, protected_attribute = 1, use_val = True):
    """Preprocess the generated data.
        
        Args: 
            path: A string indicating the path of generated data.
            attribute: A integer or a list indicating the label attribute.
            protected_attribute: A integer or a list indicating the group attribute.
            use_val: A boolean indicating whether the dataset is for training or validation.
        
        Returns:
            Information for creating a dataset.
    """
    
    if protected_attribute == [0, 1]:
        return create_dataset_gen_imgs_agegender(path, attribute, protected_attribute, use_val = use_val)
    
    
    folder_path = [x[0] for x in os.walk(path)]
    folder_path = folder_path[1:]

    all_attr = ['age', 'gender', 'smiling', 'glasses', 'haircolors']

    label_attr = attribute
    group_attr = protected_attribute
    
    yz_tuple, y_names, z_names = get_gen_info(label_attr, group_attr)
    
    img_path = {}
    for tmp_yz in yz_tuple:
        img_path[tmp_yz] = []
    
    for tmp_x in folder_path:
        tmp_folder = tmp_x.split('/')[-1]
        
        for tmp_yz in yz_tuple:
            if (tmp_folder.split('_')[label_attr] == y_names[tmp_yz[0]]) and (tmp_folder.split('_')[group_attr] == z_names[tmp_yz[1]]):
                img_path[tmp_yz].append(tmp_x)
                
    list_ids = []
    y_attr = {}
    z_attr = {}
        
    for tmp_yz in yz_tuple:
        for i in range(len(img_path[tmp_yz])):
            tmp_ids = [x[2] for x in os.walk(img_path[tmp_yz][i])][0]
            tmp_ids = [img_path[tmp_yz][i] + '/' + x for x in tmp_ids]

            list_ids.extend(tmp_ids)
            for tmp_id in tmp_ids:
                y_attr[tmp_id] = torch.Tensor(([tmp_yz[0]]))
                z_attr[tmp_id] = torch.Tensor(([tmp_yz[1]]))
    
    
    if use_val == True:
        np.random.seed(0)
        list_ids_val = np.random.choice(list_ids, 20000, replace=False)
        list_ids_train = list(set(list_ids) - set(list_ids_val))

        return list_ids_train, list_ids_val, y_attr, z_attr
    
    else:
        return list_ids, y_attr, z_attr
    

    
def create_dataset_gen_imgs_agegender(path, attribute, protected_attribute = [0, 1], use_val = True):
    """Preprocess the generated data. 
       A specialized version of the function 'def create_dataset_gen_imgs()' for the (age+gender) group attribute.
        
        Args: 
            path: A string indicating the path of generated data.
            attribute: A integer or a list indicating the label attribute.
            protected_attribute: A integer or a list indicating the group attribute.
            use_val: A boolean indicating whether the dataset is for training or validation.
        
        Returns:
            Information for creating a dataset.
    """
    
    folder_path = [x[0] for x in os.walk(path)]
    folder_path = folder_path[1:]

    all_attr = ['age', 'gender', 'smiling', 'glasses', 'haircolors']

    label_attr = attribute
    group_attr = protected_attribute
    
    yz_tuple, y_names, z_names = get_gen_info(label_attr, group_attr)
    
    img_path = {}
    for tmp_yz in yz_tuple:
        img_path[tmp_yz] = []
    
    for tmp_x in folder_path:
        tmp_folder = tmp_x.split('/')[-1]
        for tmp_yz in yz_tuple:
            if (tmp_folder.split('_')[label_attr] == y_names[tmp_yz[0]]) and (tmp_folder.split('_')[group_attr[0]] == z_names[tmp_yz[1]][0]) and (tmp_folder.split('_')[group_attr[1]] == z_names[tmp_yz[1]][1]):                
                img_path[tmp_yz].append(tmp_x)
                
    list_ids = []
    y_attr = {}
    z_attr = {}
        
    for tmp_yz in yz_tuple:
        for i in range(len(img_path[tmp_yz])):
            tmp_ids = [x[2] for x in os.walk(img_path[tmp_yz][i])][0]
            tmp_ids = [img_path[tmp_yz][i] + '/' + x for x in tmp_ids]

            list_ids.extend(tmp_ids)
            for tmp_id in tmp_ids:
                y_attr[tmp_id] = torch.Tensor(([tmp_yz[0]]))
                z_attr[tmp_id] = torch.Tensor(([tmp_yz[1]]))
    
    
    if use_val == True:
        np.random.seed(0)
        list_ids_val = np.random.choice(list_ids, 20000, replace=False)
        list_ids_train = list(set(list_ids) - set(list_ids_val))

        return list_ids_train, list_ids_val, y_attr, z_attr
    
    else:
        return list_ids, y_attr, z_attr

