# Get bus data. 

import torch

import sys
sys.path.append("/home/ubuntu/github/cS")
from rl.softQ import rlRunner

dS = int(sys.argv[1])
testEnv = rlRunner(N = 50000, dS = dS, nA=2)
testEnv.getData()
testEnv.setUp(discount = 0.8, qOptimizer = torch.optim.Adam,  qLr = 0.003,)
testEnv.trainQEstimation(40,128)
testEnv.generateData(5000)