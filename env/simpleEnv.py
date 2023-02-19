import gym
from gym import spaces
import numpy as np
import random
from math import floor
from math import exp
import sys
sys.path.append("/home/ubuntu/github/cS")

import pickle
import os

class sEnv():

	def __init__(self,  nA,  x0, A, B):

		self.nA = nA
		self.t = 0
		self.x = x0
		self.x0 = x0
		self.A = A
		self.B = B
		self.dS = len(A)


	def rF(self, s, a):

		if a==0:
			return 0
		return (np.dot(self.A,s)+0.3*a).item()

	def tF(self,s,a):
		return np.dot(self.B, s)+a+np.random.normal(0.1)


	def reset(self):
		self.t = 0
		self.x = self.x0
		return 


	def step(self, a):
		reward = self.rF(self.x, a)
		self.x = self.tF(self.x,a)
		return reward



if __name__ == "__main__":
	C=0.1
	testEnv = sEnv( 2, np.array([1,2,3]),np.array([1,2,3]), np.identity(3))
	re = []
	s = [testEnv.x]
	a = []
	for _ in range(100):
		tempA = random.randint(0, testEnv.nA)
		re.append(testEnv.step(tempA))
		s.append(testEnv.x)
		a.append(tempA)


	
