import sys
sys.path.append("/home/ubuntu/github/cS")
from rl.softQ import rlRunner
import torch 
import numpy as np

dS = int(sys.argv[1]) # Dimension of the state variable. Useful ones are only 2. 
nSampes = int(sys.argv[2]) # Number of samples generated  
discount = float(sys.argv[3])  # Discount factor 
b = float(sys.argv[4])   # 

np.random.seed(1)   # This one not working somehow
torch.manual_seed(1)
testEnv = rlRunner(N = 50000, dS = dS, nA = 2, b= b, theta = 2)
testEnv.getData()
testEnv.setUp(discount = discount, qOptimizer = torch.optim.Adam,  qLr = 0.003,)
testEnv.trainQEstimation(100,100)
testEnv.generateData(nSampes)