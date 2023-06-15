#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 10:55:46 2023

@author: hussain
"""

from torch.utils.data import Dataset
import os
import pickle
import sys
import pandas as pd
from glob import glob
from sklearn.model_selection import StratifiedKFold
import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image
from skimage import io
import torchvision.transforms as transforms
import torch

def get_data_loader(data_path, public_dataset_name, train, noise_type, noise_rate, data_indices = None):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Int20Dataset(transform, data_path, public_dataset_name, train = train, noise_type=noise_type, noise_rate=noise_rate, data_indices = data_indices)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers= 4, drop_last=True)
    return dataLoader

def load_data(data_path, dataset_name, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = Int20Dataset(transform, data_path, dataset_name, train = True, noise_type=None, noise_rate=0, data_indices = None)
    trainLoader = torch.utils.data.DataLoader(trainset,batch_size=3000, num_workers= 4)
    for index, (data, target) in enumerate(trainLoader):
        if index == 0:
            X_train = data
            y_train = target
        X_train = torch.cat((X_train, data), 0)
        y_train = torch.cat((y_train, target), 0)
        print('this is trainX: ', X_train.shape)
        
    
    # X_train, y_train = torch.stack(image, dim = 0), torch.stack(label, dim = 0)
    # test_ds = Int20Dataset(transform, data_path, dataset_name, noise_type=None, noise_rate=0)
    # X_test, y_test = test_ds.data, test_ds.target
    return (X_train, y_train)

def get_public_data_indexs(data_path, dataset_name, size, noise_type, noise_rate):
    X_train, y_train = load_data(data_path, dataset_name, noise_type=noise_type, noise_rate=noise_rate)
    n_train = y_train.shape[0]
    idxs = np.random.permutation(n_train)
    idxs = idxs[0:size]
    return idxs

def generate_data_labels(data_path, dataset_name):
    equivalent_classes = {
            #MLL dataset
            '01-NORMO': 'erythroblast',
            '04-LGL': "unknown", #atypical
            '05-MONO': 'monocyte',
            '08-LYMPH-neo': 'lymphocyte_atypical',
            '09-BASO': 'basophil',
            '10-EOS': 'eosinophil',
            '11-STAB': 'neutrophil_banded',
            '12-LYMPH-reaktiv': 'lymphocyte_atypical',
            '13-MYBL': 'myeloblast',
            '14-LYMPH-typ': 'lymphocyte_typical',
            '15-SEG': 'neutrophil_segmented',
            '16-PLZ': "unknown",
            '17-Kernschatten': 'smudge_cell',
            '18-PMYEL': 'promyelocyte',
            '19-MYEL': 'myelocyte',
            '20-Meta': 'metamyelocyte',
            '21-Haarzelle': "unknown",
            '22-Atyp-PMYEL': "unknown",
            
            'BAS': 'basophil',
            'EBO': 'erythroblast',
            'EOS': 'eosinophil',
            'KSC': 'smudge_cell',
            'LYA': 'lymphocyte_atypical',
            'LYT': 'lymphocyte_typical',
            'MMZ': 'metamyelocyte',
            'MOB': 'monocyte', #monoblast
            'MON': 'monocyte',
            'MYB': 'myelocyte',
            'MYO': 'myeloblast',
            'NGB': 'neutrophil_banded',
            'NGS': 'neutrophil_segmented',
            'PMB': "unknown",
            'PMO': 'promyelocyte',
        }
    label_map_all = {
            'basophil': 0,
            'eosinophil': 1,
            'erythroblast': 2,
            'myeloblast' : 3,
            'promyelocyte': 4,
            'myelocyte': 5,
            'metamyelocyte': 6,
            'neutrophil_banded': 7,
            'neutrophil_segmented': 8,
            'monocyte': 9,
            'lymphocyte_typical': 10,
            'lymphocyte_atypical': 11,
            'smudge_cell': 12,
            'unknown': 13
        }
    image_path = os.path.join(data_path, dataset_name)
    labels, img_paths = [], []
    for env in sorted(entry.name for entry in os.scandir(image_path) if entry.is_dir()):
        for label, image_name in enumerate(sorted(os.listdir(os.path.join(image_path, env)))):
            if(label_map_all[equivalent_classes[env]] == 13):
                {}
            else:
                image_full_path = os.path.join(image_path, env)
                labels.append(label_map_all[equivalent_classes[env]])
                img_paths.append(os.path.join(image_full_path, image_name))
    outputs = dict(path=img_paths, label=labels)
    os.makedirs(os.path.join(data_path, 'data_label'), exist_ok=True)
    output_path = os.path.join(data_path, 'data_label', dataset_name+".csv")
    df = pd.DataFrame(data=outputs)
    df.to_csv(str(output_path), index=False)
    
    df = pd.read_csv(os.path.join(data_path, 'data_label', dataset_name+".csv"))
    df['split'] = np.random.randn(df.shape[0], 1)    
    msk = np.random.rand(len(df)) <= 0.8    
    train_df = df[msk]
    train_df = pd.DataFrame(data=train_df)
    train_df.to_csv(str(os.path.join(data_path, 'data_label', dataset_name+"_train.csv")), index=False)
    test_df = df[~msk]
    test_df = pd.DataFrame(data=test_df)
    test_df.to_csv(str(os.path.join(data_path, 'data_label', dataset_name+"_test.csv")), index=False)

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()
    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state) # RandomState(MT19937)
    for idx in np.arange(m):
        i = y[idx]
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y

def noisify_pairflip(y_train, noise, random_state=None, nb_classes=13):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise
    if n > 0.0:
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy
    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=13):
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P
    if n > 0.0:
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy
    return y_train, actual_noise

def noisify(nb_classes=13, train_labels=None, noise_type=None, noise_rate=0):
    if noise_type == 'pairflip':
        train_labels = np.array(train_labels)
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_labels = np.array(train_labels)
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'No':
        train_noisy_labels = train_labels
        actual_noise_rate = 0
    return train_noisy_labels, actual_noise_rate

class Int20Dataset(Dataset):
    
    def __init__(self, transform, data_path, dataset_name, train = True, noise_type=None, noise_rate=0, data_indices = None):
        self.transform = transform
        self.data_path = data_path
        self.train = train
        
        if self.train == True:
            if data_indices is not None:
                self.data = pd.read_csv(os.path.join(os.path.join(data_path, 'data_label'), dataset_name+'_train.csv')).iloc[data_indices]
            else:
                self.data = pd.read_csv(os.path.join(os.path.join(data_path, 'data_label'), dataset_name+'_train.csv'))
        else:
            self.data = pd.read_csv(os.path.join(os.path.join(data_path, 'data_label'), dataset_name+'_test.csv'))
        
        self.train_labels = self.data.iloc[:, 1].tolist()
        self.train_noisy_labels, self.actual_noise_rate = noisify(train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate)
        # self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = self.data.iloc[index, 0]
         
        
        img = self.read_img(img_name)
        # img = io.imread(img)
        img = np.array(img)
        crop_size = 288
        h1 = (img.shape[0] - crop_size) /2
        h1 = int(h1)
        h2 = (img.shape[0] + crop_size) /2
        h2 = int(h2)
        
        w1 = (img.shape[1] - crop_size) /2
        w1 = int(w1)
        w2 = (img.shape[1] + crop_size) /2
        w2 = int(w2)
        img = img[h1:h2,w1:w2, :]
        
        
        # label = self.data.iloc[index, 1]
        label = self.train_noisy_labels[index]
        label = torch.tensor(label)
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.data)    
    def read_img(self, path):
        img = Image.open(path)
        # print('image mode:', img.mode)
        if img.mode == 'CMYK':
            img = img.convert('RGB')    
        if img.mode == 'RGBA':
            img = img.convert('RGB')    
        #img = np.array(img)
        #print('RGB img:', img.shape, img.min(), img.max())
        return img
        

if __name__=='__main__':
    data_path = './data'
    public_dataset_name = 'int20'
    private_dataset_name = 'matek19'
    generate_data_labels(data_path = data_path, dataset_name = private_dataset_name)
    noise_type = 'No' #[pairflip, symmetric, No]
    noise_rate = 0
    transform = transforms.Compose([transforms.ToTensor()])
    int20PublicData = Int20Dataset(transform, data_path=data_path, dataset_name = public_dataset_name , train = True, noise_type=noise_type, noise_rate=noise_rate, data_indices = None)
    print(len(int20PublicData))