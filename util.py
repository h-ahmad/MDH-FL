#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:39:36 2023

@author: hussain
"""

from network.resnet import ResNet10,ResNet12
from network.shufflenet import ShuffleNetG2
from network.mobilnet_v2 import MobileNetV2
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

def init_nets(n_parties,nets_name_list):
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name=='ResNet10':
            net = ResNet10()
        elif net_name =='ResNet12':
            net = ResNet12()
        elif net_name =='ShuffleNet':
            net = ShuffleNetG2()
        elif net_name =='Mobilenetv2':
            net = MobileNetV2()
        nets_list[net_i] = net
    return nets_list

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    
def update_model_with_private_data(network, private_epoch, private_dataloader, loss_function, optimizer_method, learing_rate, device):
    if loss_function =='CE':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='SCE':
        criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=10)

    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    if optimizer_method =='SGD':
        optimizer = optim.SGD(network.parameters(), lr=learing_rate, momentum=0.9, weight_decay=1e-4)
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = network(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            # if epoch_index % 5 ==0:
            #     logger.info('Private Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         batch_idx * len(images), len(private_dataloader.dataset),
            #         100. * batch_idx / len(private_dataloader), loss.item()))
    return network,participant_local_loss_batch_list

def evaluate_network(network, dataloader, device):
    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
    return acc