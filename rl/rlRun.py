"""
Main playground module for RL
"""
import numpy as np
import os
import torch
import torch.nn
from torch.nn import functional as F  # NOQA
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

import datetime
import sys
sys.path.append("/home/ubuntu/github/cS")
#from dataClass.dataClass import dataClass
#from torch.utils.data import Dataset, DataLoader
import pickle
from helper.functions import _Default, make_optimizer
from estimation.qEstimation import qEstimation

from multiprocessing import Pool
from scipy import stats

from dataClass.dataClass import rlDataClass
from torch.utils.data import Dataset, DataLoader

from env.simpleEnv import sEnv
from env.utils import dataGen

import pickle
import joblib
import pandas as pd
class rlRunner():
    def __init__(self, nS, nA, filePath):
        self.nS = nS
        self.nA = nA
        self.filePath = filePath
        #self.re = None
        #self.s = None
        #self.a = None
        self.data = None
        self.b = 0.5
        self.c = 0.5
        #self.qM = np.ones(nS, nA)
    def getData(self):
        #self.re, self.s, self.a = dataGen(self.env, 100)
        self.data = rlDataClass(self.filePath)
        self.qEstimation = qEstimation(1,self.nA, hidden_sizes=[10,10], hidden_nonlinearity=torch.sigmoid)
        self.fEstimation = qEstimation(2,self.nS, hidden_sizes=[10,10], hidden_nonlinearity=torch.sigmoid)
        # Estimate the transition. 
        # Define dq for q function. 

    def setUp(self, discount, qOptimizer, qLr):
        # Setting up the training parameters for deep models. 
        self.qOptimizer = qOptimizer
        self.fOptimizer = qOptimizer
        self.qLr = qLr
        self.discount = discount        
        self.qOptimizer = make_optimizer(self.qOptimizer, self.qEstimation, lr=self.qLr, amsgrad = True)

    def trainQEstimation(self, nEpochs, batch_size,num_workers = 0):
        # Algorithm 2 in the PQR.pdf
        # According to the doc, Q^* solves a fixed point. 
        # DQN 
        dataloaders = DataLoader(self.data, batch_size=batch_size,shuffle=True, num_workers=num_workers)
        
        lossF = torch.nn.MSELoss()
        for epoch in range(nEpochs):
            epochLoss = 0

            for _, s, a, sNext in dataloaders:
                #s[s==3]=2
                r = s*self.b+ self.c*a
                y = r+self.discount*torch.log(torch.exp(self.qEstimation(sNext)).sum())
                est = self.qEstimation.forward(s)[a.reshape(len(a))]
                lossV = lossF(est,y)
                self.qOptimizer.zero_grad()
                lossV.backward()
                self.qOptimizer.step()
                epochLoss = epochLoss + lossV

            print(f"EpochLoss for Q={epochLoss}")

    

    def getFreq(self):
        with open(filePath+'data/s.pkl', 'rb') as file:
            sR = joblib.load(file)
        with open(filePath+'/data/a.pkl', 'rb') as file:
            a = joblib.load(file)

        s = np.array(sR[0:(len(sR)-1)])
        sNext = np.array(sR[1:len(sR)])
        a = np.array(a)
        pdData = pd.DataFrame(np.vstack([a,s, sNext]).T)
        count = pdData.groupby([0,1,2], as_index=False).size()#.reset_index()
        print(count)
        freqM = np.zeros([self.nA, self.nS, self.nS])
        print(freqM.shape)
        print(freqM[0,0,2])
        freqV = count.to_numpy()
        for i in range(len(freqV)):
            print(freqV[i, [0,1,2]])
            freqM[freqV[i, [0,1,2]][0],
            freqV[i, [0,1,2]][1],
            freqV[i, [0,1,2]][2]]= freqV[i,3]
        
        print(freqM)
        print(freqM[0,:,:].sum(1))
        for i in range(freqM.shape[0]):
            freqM[i,:,:] = freqM[i,:,:]/freqM[i,:,:].sum(1).reshape(-1,1)
        
        self.freqM = freqM
        #self.freqM = self.freqM/self.freqM.sum(1).reshape([4,1])

    
    def valueIteration(self):
        self.r = np.array([[self.b*i+self.c*j  for j in range(self.nA)] for i in range(self.nS)]  )
        Q = self.Q
        for _ in range(nI):
            qSoftMax =np.log(np.exp(Q).sum(1))   # We have epsilon here!!!
            nLarge = np.zeros((self.nS, self.nA))
            for a in range(self.nA): 
                nLarge[:,a] = np.dot(np.array(self.tMatrix[a,:,:]), qSoftMax.T)
            nQ = self.r + self.discount*nLarge
            # Q = r + discount E(Q|s)
            print(((nQ - Q)**2).sum())
            Q = nQ
        self.Q = Q
        print(self.tMatrix[1,:,:])
        return 

    def getLikelihood(self):
        with open(filePath+'data/s.pkl', 'rb') as file:
            s = joblib.load(file)
        with open(filePath+'/data/a.pkl', 'rb') as file:
            a = joblib.load(file)

        s = np.array(s[0:(len(s)-1)])
        s[s==3]=2
        s[s==1] = 0
        a = np.array(a)
        pdData = pd.DataFrame(np.vstack([s,a]).T)
        qEst = self.qEstimation(torch.FloatTensor(s.reshape(-1,1))).detach().numpy()
        qS =np.choose(a.astype(int), qEst.T)
        v = np.log(np.exp(qEst).sum(1))
        print(v)
        return qS.sum()-v.sum()

    



    def girdSearch(self,bList,cList):
        re = [0]*len(bList)
        for i in range(len(bList)):
            self.b = bList[i]
            self.c = cList[i]
            testEnv.trainQEstimation(100,400)
            re[i] = testEnv.getLikelihood()
        return re

if __name__ == "__main__":
    filePath =  '/home/ubuntu/github/cS/'
    testEnv = rlRunner(4,2, filePath)
    testEnv.getData()
    testEnv.getFreq()
    testEnv.setUp(discount = 0.8, qOptimizer = torch.optim.Adam,  qLr = 0.005,)
    #testEnv.trainQEstimation(100,400)
    print(testEnv.girdSearch([0,0.5,1],[0,0.5,1]))
    #testEnv.trainQEstimation(100,100)
    #testEnv.estimateF(100,100)
