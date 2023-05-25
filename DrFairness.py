# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NSCL license
# for Dr-Fairness, except for functions select_batch_replacement() and __iter__(). 
# To view a copy of this work's license, see the LICENSE file.
#
# The original version of functions 
# select_batch_replacement() and __iter__() can be found in
# https://github.com/yuji-roh/fairbatch
# ---------------------------------------------------------------

import sys, os
import numpy as np
import math
import random
import itertools
import copy

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.optim as optim
import torch

import time

from torch.autograd import grad

from customdataset import CelebaDataset
    
    
class FairAccurateSampler(Sampler):
    """FairAccurateSampler (Sampler in DataLoader).
    
    This class is for implementing the lambda and mu adjustment and batch selection of DrFairness.

    Attributes:
        model: A model containing the intermediate states of the training.
        dset_real, dset_real_val: Training and validation datasets with real data.
        dset_gen, dset_gen_val: Training and validation datasets with generated data.
        k: A float indicating the tuning knob k used in the outer objective.
        target_fairness: A string indicating the target fairness type. Default: equalized odds (eqodds).
        n_classes: An integer indicating the number of label classes.
        device: A string indicating the device (e.g., cpu, cuda:0)
        dtype: A data type for the torch variables.
        batch_size: An integer for the size of a batch.
        num_workers: An integer for the number of workers.
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        seed: An integer indicating the random seed.
        text_directory: A string indicating the location for saving the intermediate data ratios.
        data_lr: A float indicating the learning rate of the data ratios.
        update_iter: An integer indicating how frequently updates the data ratios.
        
        N: An integer counting the size of data.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the index of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        S: A dictionary containing the default size of each class in a batch.
        lbs, mus: Real numbers indicating the lambda and mu values in DrFairness.
    """
    def __init__(self, model, dset_real, dset_real_val, dset_gen, dset_gen_val, k, target_fairness, n_classes, device = 'cpu', dtype = torch.float32, batch_size = 64, num_workers = 10, replacement = False, seed = 0, text_directory = None, data_lr = 0.005, update_iter = 1):
        """Initializes DrFairness."""
        
        torch.set_flush_denormal(True)
        
        np.random.seed(seed)
        random.seed(seed)
#         torch.random.seed(seed)
        
        self.text_directory = text_directory
    
        self.device = device
        self.dtype = dtype
        self.num_workers = num_workers
        
        self.model = model
        
        self.dset_real = dset_real
        self.dset_real_val = dset_real_val
        self.dset_gen = dset_gen
        self.dset_gen_val = dset_gen_val
        
        self.x_real = dset_real.list_IDs
        self.x_gen = dset_gen.list_IDs
        
        if type(dset_real.y) == dict:
            self.y_real = []
            self.z_real = []
            for real_id in self.x_real:
                self.y_real.append(dset_real.y[real_id])
                self.z_real.append(dset_real.z[real_id])
            self.y_real = torch.Tensor(self.y_real)
            self.z_real = torch.Tensor(self.z_real)
        else:
            self.y_real = dset_real.y
            self.z_real = dset_real.z
        
        if type(dset_gen.y) == dict:
            self.y_gen = []
            self.z_gen = []
            for gen_id in self.x_gen:
                self.y_gen.append(dset_gen.y[gen_id])
                self.z_gen.append(dset_gen.z[gen_id])
            self.y_gen = torch.Tensor(self.y_gen)
            self.z_gen = torch.Tensor(self.z_gen)
        else:
            self.y_gen = dset_gen.y
            self.z_gen = dset_gen.z
        
        
        self.k = k
        self.fairness_type = target_fairness
        self.n_classes = n_classes
        self.replacement = replacement
        self.update_iter = update_iter
        
        self.N_real = len(self.z_real)
        self.N_gen = len(self.z_gen)
        
        self.batch_size = batch_size
        self.batch_num = int( self.N_real / self.batch_size)
        
        # Takes the unique values of the tensors
        self.z_item = list(set(self.z_real.tolist()))
        self.y_item = list(set(self.y_real.tolist()))
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))
        
        
        # Makes masks for real data
        self.z_mask_real = {}
        self.y_mask_real = {}
        self.yz_mask_real = {}
        for tmp_z in self.z_item:
            self.z_mask_real[tmp_z] = (self.z_real == tmp_z)
        for tmp_y in self.y_item:
            self.y_mask_real[tmp_y] = (self.y_real == tmp_y)
        for tmp_yz in self.yz_tuple:
            self.yz_mask_real[tmp_yz] = (self.y_real == tmp_yz[0]) & (self.z_real == tmp_yz[1])

        # Finds the index
        self.z_index_real = {}
        self.y_index_real = {}
        self.yz_index_real = {}
        for tmp_z in self.z_item:
            self.z_index_real[tmp_z] = (self.z_mask_real[tmp_z] == 1).nonzero().squeeze().reshape(-1)
        for tmp_y in self.y_item:
            self.y_index_real[tmp_y] = (self.y_mask_real[tmp_y] == 1).nonzero().squeeze().reshape(-1)
        for tmp_yz in self.yz_tuple:
            self.yz_index_real[tmp_yz] = (self.yz_mask_real[tmp_yz] == 1).nonzero().squeeze().reshape(-1)
            
        # Length information
        self.z_len_real = {}
        self.y_len_real = {}
        self.yz_len_real = {}
        for tmp_z in self.z_item:
            self.z_len_real[tmp_z] = len(self.z_index_real[tmp_z])
        for tmp_y in self.y_item:
            self.y_len_real[tmp_y] = len(self.y_index_real[tmp_y])
        for tmp_yz in self.yz_tuple:
            self.yz_len_real[tmp_yz] = len(self.yz_index_real[tmp_yz])
        
        
        # Makes masks for generated data
        self.z_mask_gen = {}
        self.y_mask_gen = {}
        self.yz_mask_gen = {}
        for tmp_z in self.z_item:
            self.z_mask_gen[tmp_z] = (self.z_gen == tmp_z)
        for tmp_y in self.y_item:
            self.y_mask_gen[tmp_y] = (self.y_gen == tmp_y)
        for tmp_yz in self.yz_tuple:
            self.yz_mask_gen[tmp_yz] = (self.y_gen == tmp_yz[0]) & (self.z_gen == tmp_yz[1])

        # Finds the index
        self.z_index_gen = {}
        self.y_index_gen = {}
        self.yz_index_gen = {}
        for tmp_z in self.z_item:
            self.z_index_gen[tmp_z] = (self.z_mask_gen[tmp_z] == 1).nonzero().squeeze().reshape(-1)
        for tmp_y in self.y_item:
            self.y_index_gen[tmp_y] = (self.y_mask_gen[tmp_y] == 1).nonzero().squeeze().reshape(-1)
        for tmp_yz in self.yz_tuple:
            self.yz_index_gen[tmp_yz] = (self.yz_mask_gen[tmp_yz] == 1).nonzero().squeeze().reshape(-1)
            
        # Length information
        self.z_len_gen = {}
        self.y_len_gen = {}
        self.yz_len_gen = {}
        for tmp_z in self.z_item:
            self.z_len_gen[tmp_z] = len(self.z_index_gen[tmp_z])
        for tmp_y in self.y_item:
            self.y_len_gen[tmp_y] = len(self.y_index_gen[tmp_y])
        for tmp_yz in self.yz_tuple:
            self.yz_len_gen[tmp_yz] = len(self.yz_index_gen[tmp_yz])
        
        
        # Default batch size
        self.S = {}        
        self.S_y = {}

        for tmp_yz in self.yz_tuple:
            self.S[tmp_yz] = self.batch_size * self.yz_len_real[tmp_yz] / self.N_real
        for tmp_y in self.y_item:
            self.S_y[tmp_y] = self.batch_size * self.y_len_real[tmp_y] / self.N_real

        
        self.lbs = []
        self.mus = []
        self.lbs_logit = []
        self.mus_logit = []
        
        lb_y_tmp = {}
        self.lb_y_index = {}
        for tmp_y in self.y_item:
            lb_y_tmp[tmp_y] = []
            self.lb_y_index[tmp_y] = []
            
        for idx, tmp_yz in enumerate(self.yz_tuple):
            lb_tmp = self.yz_len_real[tmp_yz]/self.y_len_real[tmp_yz[0]]
            self.lbs.append(lb_tmp)
            lb_y_tmp[tmp_yz[0]].append(lb_tmp)
            self.lb_y_index[tmp_yz[0]].append(idx)
        
        for idx, tmp_yz in enumerate(self.yz_tuple):
            sum_lb_y = np.sum(np.exp(lb_y_tmp[tmp_yz[0]]))
            lb_logit_tmp = np.log(self.lbs[idx]) + np.log(sum_lb_y)
            self.lbs_logit.append(lb_logit_tmp)
        
        for idx, tmp_yz in enumerate(self.yz_tuple):
            self.mus.append(0.5)
            self.mus_logit.append(0.01)

            
        self.lbs = torch.tensor(self.lbs, requires_grad=False)
        self.lbs_logit = torch.tensor(self.lbs_logit, requires_grad=True)
        self.optimizer_lb = optim.Adam([self.lbs_logit], lr=data_lr)
        
        self.mus = torch.tensor(self.mus, requires_grad=False)
        self.mus_logit = torch.tensor(self.mus_logit, requires_grad=True)
        self.optimizer_mu = optim.Adam([self.mus_logit], lr=data_lr)
        
        if self.text_directory != None:
            with open(self.text_directory, "a") as myfile:
                for idx, tmp_yz in enumerate(self.yz_tuple):
                    myfile.write(str(tmp_yz)+'\t'+str(self.lbs[idx].item())+'\t'+str(self.mus[idx].item())+'/')
                myfile.write('\n')
        
        self.epoch = 0
        
    def model_eval(self, dset):
        """Calculates the losses of the intermediate model.
        
        Args: 
            dset: A dataset.
        
        Returns:
            A dictionary that contains the model loss of each (y,z)-class.
            
        """
        
        self.model.eval()
        test_loader = torch.utils.data.DataLoader (dset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)
        
        logit_all = []
        test_tmp = []
        yhat_yz_eval = {}
        for tmp_yz in self.yz_tuple:
            yhat_yz_eval[tmp_yz] = 0
        
        for i, (images, targets, z) in enumerate(test_loader):  
            print('   Evaluation in Sampler... [{}|{}]'.format(i+1, len(test_loader)), end="\r")
            images, targets, z = images.to(device=self.device, dtype=self.dtype), targets.to(device=self.device, dtype=self.dtype), z.to(device=self.device, dtype=self.dtype)    
            outputs, _ = self.model(images)
            
            if self.n_classes == 1:
                criterion = torch.nn.BCEWithLogitsLoss(reduction = 'none').to(device=self.device, dtype=self.dtype)
            else:
                criterion = torch.nn.CrossEntropyLoss(reduction = 'none').to(device=self.device, dtype=self.dtype)
                targets = targets.long()
            
            loss_tmp = criterion (outputs.squeeze(), targets.squeeze()) * torch.sum(self.lbs_logit) / torch.sum(self.lbs_logit) * torch.sum(self.mus_logit) / torch.sum(self.mus_logit)
            
            yz_mask_tmp = {}
            for tmp_yz in self.yz_tuple:
                yz_mask_tmp[tmp_yz] = (targets.squeeze() == tmp_yz[0]) & (z.squeeze() == tmp_yz[1])
            
            yz_index_tmp = {}
            for tmp_yz in self.yz_tuple:
                yz_index_tmp[tmp_yz] = (yz_mask_tmp[tmp_yz] == 1).nonzero().squeeze()
            
            
            for tmp_yz in self.yz_tuple:
                loss_sum = torch.sum(loss_tmp[yz_index_tmp[tmp_yz]])
                if yhat_yz_eval[tmp_yz] == 0:
                    yhat_yz_eval[tmp_yz] = loss_sum
                else:
                    yhat_yz_eval[tmp_yz] = yhat_yz_eval[tmp_yz] + loss_sum
            
        return yhat_yz_eval
    
    
    def get_val_data(self, dset, val_size):
        """Selects a certain number of samples for validation.
        
        Args: 
            dset: A dataset.
            val_size: An integer indicating the size of validation set used in DrFairness.
        
        Returns:
            A validation dataset and the related information.
            
        """
        
        all_class_pos = False # A flag to check whether the new validation set contains at least one sample for all (y,z)-classes
        
        while all_class_pos == False:
            random_index = np.random.choice(len(dset.list_IDs), val_size, replace=False)
            random_index = np.sort(random_index)

            y_attr_val = {}
            z_attr_val = {}
            list_IDs_val = []
            for tmp_idx in random_index:
                tmp_ID = dset.list_IDs[tmp_idx]
                list_IDs_val.append(tmp_ID)
                y_attr_val[tmp_ID] = dset.y[tmp_ID]
                z_attr_val[tmp_ID] = dset.z[tmp_ID]

            dset_val = CelebaDataset(list_IDs_val, y_attr_val, z_attr_val, dset.transform)

            x_val = dset_val.list_IDs

            y_val = []
            z_val = []
            for val_id in x_val:
                y_val.append(dset_val.y[val_id])
                z_val.append(dset_val.z[val_id])
            y_val = torch.Tensor(y_val)
            z_val = torch.Tensor(z_val)


            # Makes masks for val data
            z_mask_val = {}
            y_mask_val = {}
            yz_mask_val = {}
            for tmp_z in self.z_item:
                z_mask_val[tmp_z] = (z_val == tmp_z)
            for tmp_y in self.y_item:
                y_mask_val[tmp_y] = (y_val == tmp_y)
            for tmp_yz in self.yz_tuple:
                yz_mask_val[tmp_yz] = (y_val == tmp_yz[0]) & (z_val == tmp_yz[1])
            
            # Finds the index
            yz_index_val = {}
            for tmp_yz in self.yz_tuple:
                yz_index_val[tmp_yz] = (yz_mask_val[tmp_yz] == 1).nonzero().squeeze().reshape(-1)
                    
            z_index_val = {}
            for tmp_z in self.z_item:
                z_index_val[tmp_z] = (z_mask_val[tmp_z] == 1).nonzero().squeeze().reshape(-1)
            
            zero_flag = 0
            for tmp_yz in self.yz_tuple:
                if len(yz_index_val[tmp_yz]) > 0:
                    zero_flag += 1
            
            if zero_flag == len(self.yz_tuple):
                all_class_pos = True
        
        
        return dset_val, y_val, z_val, yz_index_val, z_index_val
    
    

    def adjust_lambda(self):
        """Updates the data ratio values for DrFairness algorithm.
        
        The detailed algorithms are decribed in the paper.

        """
        
        self.optimizer_lb.zero_grad()
        self.optimizer_mu.zero_grad()
        
        
        dset_real_val, y_real_val, z_real_val, yz_index_real_val, z_index_real_val = self.get_val_data(self.dset_real_val, val_size = 128)
        dset_gen_val, y_gen_val, z_gen_val, yz_index_gen_val, z_index_gen_val = self.get_val_data(self.dset_gen_val, val_size = 128)
        
        time1 = time.time()
        yhat_yz_real = self.model_eval(dset_real_val)
        yhat_yz_gen = self.model_eval(dset_gen_val)
        time2 = time.time()
        
        for tmp_yz in self.yz_tuple:
            yhat_yz_real[tmp_yz] = yhat_yz_real[tmp_yz]/len(yz_index_real_val[tmp_yz])
            yhat_yz_gen[tmp_yz] = yhat_yz_gen[tmp_yz]/len(yz_index_gen_val[tmp_yz])
            
        # Calculate the fairness loss in the outer objective
        for idx_y, tmp_y in enumerate(self.y_item):
            z_pairs = [(a, b) for idx_z, a in enumerate(self.z_item) for b in self.z_item[idx_z + 1:]]
            for idx_z, z_pair in enumerate(z_pairs):
                if idx_y == 0 and idx_z == 0:
                    outer_fair = torch.abs(yhat_yz_real[(tmp_y,z_pair[0])] - yhat_yz_real[(tmp_y,z_pair[1])]).view(1)
                else:
                    outer_fair = torch.cat([outer_fair, torch.abs(yhat_yz_real[(tmp_y,z_pair[0])] - yhat_yz_real[(tmp_y,z_pair[1])]).view(1)])

        outer_fair = torch.max(outer_fair)
        
        # Calculate the accuracy loss in the outer objective
        for idx, tmp_yz in enumerate(self.yz_tuple):
            if idx == 0:
                outer_acc = yhat_yz_real[tmp_yz] * (len(yz_index_real_val[tmp_yz])/len(y_real_val))
            else:
                outer_acc = outer_acc + yhat_yz_real[tmp_yz] * (len(yz_index_real_val[tmp_yz])/len(y_real_val))
        
        # Calculate the outer objective
        outer = outer_fair + self.k * outer_acc
        
        
        # Calculate the inner objective
        weighted_S_real = {}
        weighted_S_gen = {}
        for idx, tmp_yz in enumerate(self.yz_tuple):
            weighted_S_real[tmp_yz] = self.lbs[idx] * self.mus[idx] * sum((y_real_val == tmp_yz[0]).squeeze())
            weighted_S_gen[tmp_yz] = self.lbs[idx] * (1-self.mus[idx]) * sum((y_real_val == tmp_yz[0]).squeeze())
            
        for idx, tmp_yz in enumerate(self.yz_tuple):
            if idx == 0:
                inner = (weighted_S_real[tmp_yz] * yhat_yz_real[tmp_yz]) + (weighted_S_gen[tmp_yz] * yhat_yz_gen[tmp_yz])
            else:
                inner = inner + (weighted_S_real[tmp_yz] * yhat_yz_real[tmp_yz]) + (weighted_S_gen[tmp_yz] * yhat_yz_gen[tmp_yz])
        inner = inner/len(y_real_val)
        
        
        # Get gradients for lbs_logit and mus_logit
        def flatten_grad(loss_grad):
            return torch.cat([p.view(-1) for p in loss_grad])
        
        d_outer_d_lbs = flatten_grad(grad(outer, self.lbs_logit, create_graph=True, allow_unused=True))
        d_outer_d_w = flatten_grad(grad(outer, self.model.resnet.fc.parameters(), create_graph=True, allow_unused=True))
        d_inner_d_w = flatten_grad(grad(inner, self.model.resnet.fc.parameters(), create_graph=True, allow_unused=True))
        d_inner_d_wlb = flatten_grad(grad(d_inner_d_w, self.lbs_logit, grad_outputs= d_outer_d_w, retain_graph=True, allow_unused=True))
        final_grad_lbs = d_outer_d_lbs - d_inner_d_wlb
        
        
        self.lbs_logit.grad = final_grad_lbs.detach()
        self.optimizer_lb.step()
        
        d_outer_d_mus = flatten_grad(grad(outer, self.mus_logit, create_graph=True, allow_unused=True))
        d_outer_d_w_2 = flatten_grad(grad(outer, self.model.resnet.fc.parameters(), create_graph=True, allow_unused=True))
        d_inner_d_w_2 = flatten_grad(grad(inner, self.model.resnet.fc.parameters(), create_graph=True, allow_unused=True))
        d_inner_d_wmu = flatten_grad(grad(d_inner_d_w_2, self.mus_logit, grad_outputs = d_outer_d_w_2, allow_unused=True))
        final_grad_mus = d_outer_d_mus - d_inner_d_wmu
        
        self.mus_logit.grad = final_grad_mus.detach()
        self.optimizer_mu.step()
        
        # Get the updated lambda and mu values
        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim = 0)

        for tmp_y in self.y_item:
            self.lbs[self.lb_y_index[tmp_y]] = softmax(self.lbs_logit[self.lb_y_index[tmp_y]]).float()
        self.mus = sigmoid(self.mus_logit)
        
        # Save the updated lambda and mu values
        if self.text_directory != None:
            with open(self.text_directory, "a") as myfile:
                for idx, tmp_yz in enumerate(self.yz_tuple):
                    myfile.write(str(tmp_yz)+'\t'+str(self.lbs[idx].item())+'\t'+str(self.mus[idx].item())+'/')
                myfile.write('\n')

    
    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indices that indicate the data.
            
        """
        
        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                np_index = np.random.choice(full_index, batch_size, replace = False)
                select_index.append(np_index)
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    np_index = np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)]))
                    select_index.append(np_index)
                    
                    start_idx = len(full_index)-start_idx
                else:
                    np_index = tmp_index[start_idx:start_idx + batch_size]
                    select_index.append(np_index)
                    start_idx += batch_size
    
        return select_index

    
    def __iter__(self):
        """Serves the batches to training.
        
        Returns:
            Indices that indicate the data in each batch.
            
        """
            
        self.batch_num = self.update_iter # Set the number of mini-batches to the data ratio update frequency

        self.adjust_lambda() # Adjust the lambda and mu values

        each_size_real = {}
        each_size_gen = {}

        sort_index_real = {}
        sort_index_gen = {}

        for idx, tmp_yz in enumerate(self.yz_tuple):
            if torch.isnan(self.lbs[idx]) or torch.isnan(self.mus[idx]):
                print("Some of the weights are nan!")
                break
            each_size_real[tmp_yz] = round(self.lbs[idx].item() * self.mus[idx].item() * self.S_y[tmp_yz[0]])
            each_size_gen[tmp_yz] = round(self.lbs[idx].item() * (1-self.mus[idx].item()) * self.S_y[tmp_yz[0]])

        for idx, tmp_yz in enumerate(self.yz_tuple):
            # Get the indices for each class
            sort_index_real[tmp_yz] = self.select_batch_replacement(each_size_real[tmp_yz], self.yz_index_real[tmp_yz], self.batch_num, self.replacement)
            sort_index_gen[tmp_yz] = self.select_batch_replacement(each_size_gen[tmp_yz], self.yz_index_gen[tmp_yz], self.batch_num, self.replacement)

        for batch_i in range(self.batch_num):
            for idx, tmp_yz in enumerate(self.yz_tuple):
                if idx == 0:
                    key_in_fairbatch = sort_index_real[tmp_yz][batch_i].copy()
                    key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_gen[tmp_yz][batch_i].copy()+self.N_real))
                else:
                    key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_real[tmp_yz][batch_i].copy()))
                    key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_gen[tmp_yz][batch_i].copy()+self.N_real))

            random.shuffle(key_in_fairbatch)

            yield key_in_fairbatch
    
    def __len__(self):
        """Returns the number of batches."""
        return self.batch_num

