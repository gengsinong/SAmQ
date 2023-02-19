## A class for giving data in a batch mode. 

# To train deep learning models, we need to be able to proivde data in batches efficiently. 
# To this end, we build this data class. An example can be found at https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

from torch.utils.data import Dataset, DataLoader 
import random
import numpy as np
import torch 
import os
import pickle
import joblib

class rlDataClass():
    def __init__(self, filePath):
        with open(filePath+'data/s.pkl', 'rb') as file:
            self.s = joblib.load(file)
        with open(filePath+'/data/a.pkl', 'rb') as file:
            self.a = joblib.load(file)
        with open(filePath+'/data/r.pkl', 'rb') as file:
            self.r = joblib.load(file)  
    def __len__(self):
        return len(self.a)
    def __getitem__(self,idx):
        r = torch.FloatTensor([self.r[idx]])
        s = torch.FloatTensor([self.s[idx]])
        a = torch.LongTensor([self.a[idx]])
        sNext = torch.FloatTensor([self.s[idx+1]])
        return r, s, a, sNext


class irlDataClass():
    def __init__(self, filePath):
        with open(filePath+'data/s.pkl', 'rb') as file:
            self.s = joblib.load(file)
        with open(filePath+'/data/a.pkl', 'rb') as file:
            self.a = joblib.load(file)
        with open(filePath+'/data/r.pkl', 'rb') as file:
            self.r = joblib.load(file)  
    def __len__(self):
        return len(self.a)
    def __getitem__(self,idx):
        r = torch.FloatTensor([self.r[idx]])
        s = torch.FloatTensor([self.s[idx]])
        a = torch.LongTensor([self.a[idx]])
        sNext = torch.FloatTensor([self.s[idx+1]])
        return r, s, a, sNext
        
