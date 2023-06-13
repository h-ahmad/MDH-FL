#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:54:55 2023

@author: hussain
"""

import random
import numpy as np
import torch
import torch.nn as nn
from util import init_nets, SCELoss, update_model_with_private_data, evaluate_network
from data_loader import get_public_data_indexs, get_data_loader
import torchvision.transforms as transforms
import statistics
import torch.nn.functional as F
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description = 'Main Script')
parser.add_argument('--data_path', type = str, default = './data', help = 'Path to the main directory')
parser.add_argument('--public_dataset_name', type = str, default = 'int20', help = 'Name of public dataset')
parser.add_argument('--public_dataset_path', type = str, default = 'int20', help = 'Directory of public dataset')
parser.add_argument('--public_dataset_length', type = int, default = 5000, help = 'Length of public dataset')
parser.add_argument('--private_dataset_name', type = str, default = 'matek19', help = 'Name of private dataset')
parser.add_argument('--private_dataset_path', type = str, default = 'matek19', help = 'Directory of private dataset')
parser.add_argument('--private_dataset_length', type = int, default = 3000, help = 'Length of private dataset')
parser.add_argument('--number_of_clients', type = int, default = 3, help = 'Total nodes to which dataset is divided')
parser.add_argument('--batch_size', type = int, default = 16, help = 'Training batch size')
parser.add_argument('--communication_epochs', type = int, default = 40, help = 'Communication rounds for collaborative learning')
parser.add_argument('--noise_type', type = str, default = 'No', choices = ['pair', 'symmetric', 'No'], help = 'Noise type to be added')
parser.add_argument('--noise_rate', type = float, default = 0, help = 'Noise rate', choices = [0, 0.1, 0.2])
parser.add_argument('--total_private_samples', type = int, default = 13000, help = 'Total private samples')
parser.add_argument('--fl_type', type = str, default = 'homogeneous', choices = ['heterogeneous', 'homogeneous'], help = 'Type of FL, either heterogeneous or homogeneous')
parser.add_argument('--client_confidence_reweight_loss', type = str, default = 'SCE', choices=['SCE', 'CE'])
parser.add_argument('--number_of_classes', type = int, default = 13, help = 'Total number of classes in dataset')
parser.add_argument('--optimizer_name', type=str, default = 'Adam', help = 'Optimizer method')
parser.add_argument('--learning_rate', type=float, default = 0.001, help = 'Learning rate for optimizer')

args = parser.parse_args() 

if __name__ =='__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    Client_Confidence_Reweight = True
    if Client_Confidence_Reweight == False:
        beta = 0
    else:
        beta = 0.5
    
    transform = transforms.Compose([transforms.ToTensor()])
    device_ids = [0]
    
    if args.fl_type == 'heterogeneous':
        private_network_list = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2']
    else:
        private_network_list = ['ResNet12','ResNet12','ResNet12','ResNet12']
    net_list = init_nets(n_parties=args.number_of_clients,nets_name_list=private_network_list)
    
    net_dataidx_map = {}
    for index in range(args.number_of_clients):
        idxes = np.random.permutation(args.total_private_samples)
        idxes = idxes[0:args.private_dataset_length]
        net_dataidx_map[index]= idxes
        
    # public_data_indexs = get_public_data_indexs(data_path = args.data_path, dataset_name = args.public_dataset_name, 
    #                                             size=args.public_dataset_length, noise_type=args.noise_type, noise_rate=args.noise_rate) # <class 'numpy.ndarray'>

    public_train_dl = get_data_loader(args.data_path, args.public_dataset_name, train = True, noise_type = args.noise_type, noise_rate = args.noise_rate, data_indices = None)
    
    current_mean_loss_list = []
    col_loss_list = []
    local_loss_list = []
    acc_list = []
    for epoch_index in range(args.communication_epochs):        
        for participant_index in range(args.number_of_clients):
            network = net_list[participant_index]
            # network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network = network.to(device)
        
        # Calculate Client Confidence with label quality and model performance
        amount_with_quality = [1 / (args.number_of_clients - 1) for i in range(args.number_of_clients)]
        quality_list = []
        weight_with_quality = []
        amount_with_quality_exp = []
        last_mean_loss_list = current_mean_loss_list
        current_mean_loss_list = []
        for participant_index in range(args.number_of_clients):
            network = net_list[participant_index]
            # network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network = network.to(device)
            network.train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local = get_data_loader(args.data_path, args.private_dataset_name, train = True, noise_type = args.noise_type, noise_rate = args.noise_rate, data_indices = private_dataidx)
            if args.client_confidence_reweight_loss == 'CE':
                criterion = nn.CrossEntropyLoss()
            if args.client_confidence_reweight_loss == 'SCE':
                criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=args.number_of_classes)
            criterion.to(device)
            participant_loss_list = []
            for batch_idx, (images, labels) in enumerate(train_dl_local):
                # print(' loop 3, client : ', participant_index, ', epoch: ', batch_idx+1)
                images = images.to(device)
                labels = labels.to(device)
                private_linear_output, _ = network(images)
                private_loss = criterion(private_linear_output, labels)
                participant_loss_list.append(private_loss.item())
            mean_participant_loss = statistics.mean(participant_loss_list)
            current_mean_loss_list.append(mean_participant_loss)
        #EXP Normalization Process
        if epoch_index > 0 :
            for participant_index in range(args.number_of_clients):
                delta_loss = last_mean_loss_list[participant_index] - current_mean_loss_list[participant_index]
                quality_list.append(delta_loss / current_mean_loss_list[participant_index])
            quality_sum = sum(quality_list)
            for participant_index in range(args.number_of_clients):
                amount_with_quality[participant_index] += beta * quality_list[participant_index] / quality_sum
                amount_with_quality_exp.append(np.exp(amount_with_quality[participant_index]))
            amount_with_quality_sum = sum(amount_with_quality_exp)
            for participant_index in range(args.number_of_clients):
                weight_with_quality.append(amount_with_quality_exp[participant_index] / amount_with_quality_sum)
        else:
            weight_with_quality = [1 / (args.number_of_clients - 1) for i in range(args.number_of_clients)]
        weight_with_quality = torch.tensor(weight_with_quality)
        
        # training on public dataset
        for batch_idx, (images, _) in enumerate(public_train_dl):            
            linear_output_list = []
            linear_output_target_list = []
            kl_loss_batch_list = []
            # Calculate Linear Output
            for participant_index in range(args.number_of_clients):
                print('public data, client: ', participant_index+1, ', epoch: ', batch_idx+1)
                network = net_list[participant_index]
                # network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network = network.to(device)
                network.train()
                images = images.to(device)
                linear_output,_ = network(x=images)
                linear_output_softmax = F.softmax(linear_output,dim =1)
                linear_output_target_list.append(linear_output_softmax.clone().detach())
                linear_output_logsoft = F.log_softmax(linear_output,dim=1)
                linear_output_list.append(linear_output_logsoft)
            
            # Update Participants' Models via KL Loss and Data Quality
            local_loss_batch_list = []            
            for participant_index in range(args.number_of_clients):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                criterion = nn.KLDivLoss(reduction='batchmean')
                criterion.to(device)
                optimizer = optim.Adam(network.parameters(), lr = args.learning_rate)
                optimizer.zero_grad()
                loss = torch.tensor(0)
                for i in range(args.number_of_clients):
                    if i != participant_index:
                        weight_index = weight_with_quality[i]
                        loss_batch_sample = criterion(linear_output_list[participant_index], linear_output_target_list[i])
                        temp = weight_index * loss_batch_sample
                        loss = loss + temp
                kl_loss_batch_list.append(loss.item())
                loss.backward()
                optimizer.step()
            col_loss_list.append(kl_loss_batch_list)
        # Update Participants' Models via Private Data
        local_loss_batch_list = []
        for participant_index in range(args.number_of_clients):
            network = net_list[participant_index]
            # network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network = network.to(device)
            network.train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local = get_data_loader(args.data_path, args.private_dataset_name, train = True, noise_type = args.noise_type, noise_rate = args.noise_rate, data_indices = private_dataidx)
            private_epoch = max(int(args.public_dataset_length/args.private_dataset_length),1)
            network, private_loss_batch_list = update_model_with_private_data(network, private_epoch, train_dl_local,  args.client_confidence_reweight_loss, args.optimizer_name, args.learning_rate, device)            
            mean_privat_loss_batch = statistics.mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_privat_loss_batch)
        local_loss_list.append(local_loss_batch_list)
        
        # Evaluate models in the final round
        if epoch_index == args.communication_epochs - 1:
            acc_epoch_list = []
            for participant_index in range(args.number_of_clients):
                test_dl = get_data_loader(args.data_path, args.private_dataset_name, train = True, noise_type = args.noise_type, noise_rate = args.noise_rate, data_indices = private_dataidx)
                network = net_list[participant_index]
                # network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network = network.to(device)
                accuracy = evaluate_network(network, test_dl, device)
                acc_epoch_list.append(accuracy)
            acc_list.append(acc_epoch_list)
            accuracy_avg = sum(acc_epoch_list) / args.number_of_clients
            print('accuracy_avg: ', accuracy_avg)
            acc_array = np.array(acc_list)
            np.save(args.fl_type+'_'+args.noise_type+'_'+args.noise_rate+'_accuracy.npy', acc_array)
            
            
            
            
            