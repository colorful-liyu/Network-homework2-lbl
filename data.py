import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

import random
import os
import csv


class SeedTrainSet(Dataset):
    def __init__(self, tgt_mode=0, data_path='data.pkl'):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

            src_data_list = []
            src_label_list = []
            for item in list(data.keys()):

                if str(tgt_mode) in item:
                    tgt_data = data[item]['data']
                    tgt_label = data[item]['label']+1
                else:
                    src_data_list.append(data[item]['data'])
                    src_label_list.append(data[item]['label']+1)
            
            self.src_data_list = torch.tensor(src_data_list).permute(1,0,2)[:2800]
            self.src_label_list = torch.tensor(src_label_list).permute(1,0)[:2800]

            self.tgt_data_list = torch.tensor(tgt_data)[:2500]
            self.tgt_label_list = torch.tensor(tgt_label)[:2500]

            self.data_dim = self.tgt_data_list.shape[-1]

    def __getitem__(self, index: int):
        
        src_data = self.src_data_list[index].reshape(-1, self.data_dim)
        src_label = self.src_label_list[index].reshape(-1, 1)
        src_domain_label = torch.zeros_like(src_label)
        tgt_data = self.tgt_data_list[index]
        tgt_label = self.tgt_label_list[index]
        tgt_domain_label = torch.ones_like(tgt_label)

        sample = {
            'src_data' : src_data.to(torch.float32),
            'src_label' : src_label.long(),
            'src_domain_label' : src_domain_label.long(),
            'tgt_data' : tgt_data.to(torch.float32),
            'tgt_label' : tgt_label.long(),
            'tgt_domain_label' : tgt_domain_label.long()
        }
        
        return sample

    def __len__(self):
        return self.tgt_label_list.shape[0]


class SeedTestSet(Dataset):
    def __init__(self, tgt_mode=0, data_path='data.pkl'):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

            src_data_list = []
            src_label_list = []
            for item in list(data.keys()):

                if str(tgt_mode) in item:
                    tgt_data = data[item]['data']
                    tgt_label = data[item]['label']+1
                else:
                    src_data_list.append(data[item]['data'])
                    src_label_list.append(data[item]['label']+1)
            
            self.src_data_list = torch.tensor(src_data_list).permute(1,0,2)[:-500]
            self.src_label_list = torch.tensor(src_label_list).permute(1,0)[:-500]

            self.tgt_data_list = torch.tensor(tgt_data)[:-500]
            self.tgt_label_list = torch.tensor(tgt_label)[:-500]

            self.data_dim = self.tgt_data_list.shape[-1]

    def __getitem__(self, index: int):
        
        src_data = self.src_data_list[index].reshape(-1, self.data_dim)
        src_label = self.src_label_list[index].reshape(-1, 1)
        src_domain_label = torch.zeros_like(src_label)
        tgt_data = self.tgt_data_list[index]
        tgt_label = self.tgt_label_list[index]
        tgt_domain_label = torch.ones_like(tgt_label)

        sample = {
            'src_data' : src_data.to(torch.float32),
            'src_label' : src_label.long(),
            'src_domain_label' : src_domain_label.long(),
            'tgt_data' : tgt_data.to(torch.float32),
            'tgt_label' : tgt_label.long(),
            'tgt_domain_label' : tgt_domain_label.long()
        }
        
        return sample

    def __len__(self):
        return self.tgt_label_list.shape[0]


def get_loader(args):

    train_data = SeedTrainSet(tgt_mode = args.mode)
    test_data = SeedTestSet(tgt_mode = args.mode)
    
    train_data_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    return train_data_loader, test_data_loader, len(train_data), len(test_data)
    