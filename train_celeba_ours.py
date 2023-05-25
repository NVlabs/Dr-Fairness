# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for Dr-Fairness. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import sys, os
import numpy as np
import math
import time
import random
import copy
import itertools
from argparse import Namespace
import argparse
from PIL import Image

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch

from models import attribute_classifier
from customdataset import CelebaDataset, CelebaDataset_Adaptive
import preprocessing_celeba as pre
from DrFairness import FairAccurateSampler
from ema import EMA

import warnings
warnings.filterwarnings("ignore")


def train(model_class, dset_real, dset_real_val, dset_gen, dset_gen_val, device, dtype, k, target_fairness, n_classes = 1, save_path = 'intermediate_models/', batch_size = 64, num_workers = 10, data_lr = 0.01, update_iter = 1, seed = 0, start_epoch = 0):
    """Trains the model.
        
        Args: 
            model_class: A model class containing model and optimizer.
            dset_real, dset_real_val: Training and validation datasets with real data.
            dset_gen, dset_gen_val: Training and validation datasets with generated data.
            device: A string indicating the device (e.g., cpu, cuda:0)
            dtype: A data type for the torch variables.
            k: A float indicating the tuning knob k used in the outer objective.
            target_fairness: A string indicating the target fairness type. Default: equalized odds (eqodds).
            n_classes: An integer indicating the number of label classes.
            save_path: A string indicating the directory for model saving.
            batch_size: An integer for the size of a batch.
            num_workers: An integer for the number of workers.
            data_lr: A float indicating the learning rate of the data ratios.
            update_iter: An integer indicating how frequently updates the data ratios.
            seed: An integer indicating the random seed.
            start_epoch: An integer indicating the starting point of the model training, which has a positive value when model training is continued with the pre-trained model.
        
    """        
    torch.set_flush_denormal(True)
    
    torch.manual_seed(seed)
    model = model_class.model.to(device=device, dtype=dtype)
    
    # ---------------------
    #  Define Sampler and DataLoader
    # ---------------------
    sampler = FairAccurateSampler (model, dset_real, dset_real_val, dset_gen, dset_gen_val, k = k, target_fairness = target_fairness, n_classes = n_classes, device = device, dtype = dtype, batch_size = batch_size, num_workers = num_workers, replacement = False, seed = seed, text_directory = save_path+"/data_ratios.txt", data_lr = data_lr, update_iter = update_iter)
    
    all_list_IDs = []
    all_list_IDs.extend(dset_real.list_IDs)
    all_list_IDs.extend(dset_gen.list_IDs)
        
    all_y_attr = dset_real.y.copy()
    all_y_attr.update(dset_gen.y.copy())
    all_z_attr = dset_real.z.copy()
    all_z_attr.update(dset_gen.z.copy())
    
    transform = pre.img_transform()
    dset_all = CelebaDataset_Adaptive(all_list_IDs, all_y_attr, all_z_attr, transform)
    
    train_loader = DataLoader (dset_all, sampler=sampler, num_workers = num_workers)
    
    model_class.model.train()
    model_class.optimizer.zero_grad()
    
    
    if n_classes == 1:
        lossCE = torch.nn.BCEWithLogitsLoss().to(device=device, dtype=dtype)
    else:
        lossCE = torch.nn.CrossEntropyLoss().to(device=device, dtype=dtype)
    
    train_loss = 0
    
    # ---------------------
    #  Train the model
    # ---------------------
    for epoch in range(total_epochs):
        print("Epoch: {}".format(epoch + start_epoch))
        y_all = []
        scores_all = []
        for i, (images, targets, z) in enumerate(train_loader):
            
            images = images[0]
            targets = targets[0]
            images, targets = images.to(device=device, dtype=dtype), targets.to(device=device, dtype=dtype)
            
            outputs, _ = model_class.forward(images)
            
            if n_classes != 1:
                targets = targets.long()
                
            loss = lossCE(outputs.squeeze(), targets.squeeze())
            
            loss = loss / update_iter
            loss.backward()
            
            train_loss += loss.item()
                
            if (i+1) == update_iter:
                model_class.optimizer.step()                
                model_class.optimizer.zero_grad()           
                print("...After {} iters. Loss: {}".format(update_iter, train_loss), flush=True)
                train_loss = 0
                break

        # ---------------------
        #  Save the intermediate model
        # ---------------------
        if (epoch+1)%100 == 0:
            path = save_path + '/'+str(epoch + start_epoch)+'.pth'
            model_class.save_model(path, epoch + start_epoch)

            
def dataset_creation_real(path, label_attribute, group_attribute):
    """Creates training and validation datasets for real data.
        
        Args: 
            path: A string indicating data directory.
            label_attribute: Integers indicating label attributes.
            group_attribute: Integers indicating group attributes.
            
        Returns:
            Training and validation datasets.
            
        """
    
    if label_attribute != [8, 9]:
        list_ids_real, y_attr_real, z_attr_real = pre.create_dataset_real_imgs(path, attribute=label_attribute, protected_attribute=group_attribute, split='train')
        list_ids_real_val, y_attr_real_val, z_attr_real_val = pre.create_dataset_real_imgs(path, attribute=label_attribute, protected_attribute=group_attribute, split='valid')
    else:
        list_ids_real, y_attr_real, z_attr_real = pre.create_dataset_real_imgs_haircolors(path, protected_attribute = group_attribute, split='train')
        list_ids_real_val, y_attr_real_val, z_attr_real_val = pre.create_dataset_real_imgs_haircolors(path, protected_attribute = group_attribute, split='valid')

    transform = pre.img_transform()
    dset_real = CelebaDataset(list_ids_real, y_attr_real, z_attr_real, transform)
    dset_real_val = CelebaDataset(list_ids_real_val, y_attr_real_val, z_attr_real_val, transform)
    
    return dset_real, dset_real_val


def dataset_creation_gen(path, label_attribute, group_attribute):
    """Creates training and validation datasets for generated data.
        
        Args: 
            path: A string indicating data directory.
            label_attribute: Integers indicating label attributes.
            group_attribute: Integers indicating group attributes.
            
        Returns:
            Training and validation datasets.
    """
    
    list_ids_gen, list_ids_gen_val, y_attr_gen, z_attr_gen = pre.create_dataset_gen_imgs(path, label_attribute, group_attribute)

    transform = pre.img_transform()
    dset_gen = CelebaDataset(list_ids_gen, y_attr_gen, z_attr_gen, transform)
    dset_gen_val = CelebaDataset(list_ids_gen_val, y_attr_gen, z_attr_gen, transform)
    
    return dset_gen, dset_gen_val


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fairness', type=str, default='original')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='data/celeba')
    parser.add_argument('--gen_path', type=str, default='data/celeba_gen/')
    parser.add_argument('--save_path', type=str, default='intermediate_models/')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--total_epochs', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--n_classes', type=int, default=1)
    
    parser.add_argument('--model_lr', type=float, default=0.0001)
    parser.add_argument('--data_lr', type=float, default=0.005)
    parser.add_argument('--k', type=float, default=20)
    parser.add_argument('--update_iter', type=int, default=1)
    parser.add_argument('--use_ema', type=int, default=1)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--y', type=str, default='age')
    parser.add_argument('--z', type=str, default='gender')
    
    opt = vars(parser.parse_args())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.cuda.manual_seed_all(opt['seed'])
        device = "cuda:"+str(opt['cuda_device'])    
    dtype = torch.float32
    num_device = 1
    num_workers = opt['num_workers']
    batch_size = opt['batch_size']
    n_classes = opt['n_classes']
    total_epochs = opt['total_epochs']
    model_lr = opt['model_lr']
    data_path = opt['data_path']
    gen_path = opt['gen_path']
    target_fairness = opt['fairness']
    if opt['use_ema'] == 1:
        use_ema = True
    else:
        use_ema = False
        
    
    seeds_cand = [0,1,2]
    
    for seed in seeds_cand:
        
        # ---------------------
        #  Create the saved path
        # ---------------------
        save_path = opt['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        save_path = save_path + '/seed_' + str(seed)        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        open(save_path+'/data_ratios.txt', 'w').close()

        print("----------------------------------")
        print("Using Both Real & Generated Data")
        print("Target fairness: ", target_fairness)
        print("Total epochs: ", total_epochs)
        print("Num classes: ", n_classes)
        print("Batch size: ", batch_size)
        print("Data path: ", data_path)
        print("Gen path: ", gen_path)
        print("Save path: ", save_path)
        print("Device: ", device)
        print("Num workers: ", num_workers)
        print("----------------------------------")
        print("Model lr: ", opt['model_lr'])
        print("Data Ratio lr: ", opt['data_lr'])
        print("Tuning knob k: ", opt['k'])
        print("Update iter freq: ", opt['update_iter'])
        if use_ema == True:
            print("Use EMA: True")
            print("EMA decay: ", opt['ema_decay'])
        else:
            print("Use EMA: False")
        print("----------------------------------")
        print("Label attr: ", opt['y'])
        print("Group attr: ", opt['z'])
        print("----------------------------------")
        
        # ---------------------
        #  Load real and generated datasets
        # ---------------------
        orig_y_idx, orig_z_idx, gen_y_idx, gen_z_idx = pre.get_attr_index(opt['y'], opt['z'])
        dset_real, dset_real_val = dataset_creation_real(data_path, label_attribute=orig_y_idx, group_attribute=orig_z_idx)
        dset_gen, dset_gen_val = dataset_creation_gen(gen_path, label_attribute=gen_y_idx, group_attribute=gen_z_idx)
        
        # ---------------------
        #  Initialize the classifier
        # ---------------------
        model_class = attribute_classifier(device, dtype, n_classes=n_classes, learning_rate = model_lr, use_ema = use_ema, ema_decay = opt['ema_decay'])
        
        # ---------------------
        #  Run the training function
        # ---------------------
        train(model_class, dset_real, dset_real_val, dset_gen, dset_gen_val, device, dtype, k = opt['k'], target_fairness = target_fairness, n_classes = n_classes, save_path = save_path, batch_size = batch_size, num_workers = num_workers, data_lr = opt['data_lr'], update_iter = opt['update_iter'], seed = seed)
