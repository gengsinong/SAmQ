import sys
sys.path.append("/home/ubuntu/github/cS")
from irl.pq import mainRunner
import torch 
import numpy as np



deltaSAL = np.array([0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5])*5
deltaNL = deltaSAL*15

torch.manual_seed(1)

testEnv = mainRunner(dataId = None )
testEnv.getData(expId = 'Saved')
testEnv.setUp( qOptimizer = torch.optim.Adam,  qLr = 0.0005, qStarOptimizer = torch.optim.Adam, qStarLr = 0.003, discount = 0.8)
# Running
testEnv.trainQEstimation(nEpochs = 50,batch_size= 30)
testEnv.trainQStarEstimation(100,20)
testEnv.squashAll(deltaSAList =deltaSAL, deltaNaiveList = deltaNL)# squashing.
    
