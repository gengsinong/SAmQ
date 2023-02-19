import sys
sys.path.append("/home/ubuntu/github/cS")
from rl.iterationDDM import iDDMRunner
import torch 
import numpy as np



nIterations = 5000
thetaL = 1.8+np.arange(60)*0.01
deltaSAL = np.array([0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5])*5
deltaNL = deltaSAL*15

testEnv = iDDMRunner(discount=0.8, thetaList = thetaL, dataId = None)
testEnv.runAll(squashMethod='', deltaList = deltaSAL, nIterations = nIterations)
testEnv.runAll(squashMethod='n', deltaList = deltaNL, nIterations = nIterations)