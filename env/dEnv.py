# discrete state discrete action

import gym
from gym import spaces
import numpy as np
import random
from math import floor
from math import exp
import sys
sys.path.append("/home/ubuntu/github/cS")
import mdptoolbox.example
import pickle
import os

class dEnv():

	def __init__(self,  tMatrix, x0, b,c):

		self.nA, _, self.nS = tMatrix.shape
		self.t = 0
		self.x = x0
		self.x0 = x0
		self.c = c
		self.b = b
		self.tMatrix = tMatrix

	def rF(self, s, a):

		if a==0:
			return 0
		# Squashing!!!!
		if s>1:
			return a#return a
		return self.b*s + self.c*a

	def tF(self,s,a):
		#print(a)

		
		p = self.tMatrix[a,s]
		return np.random.choice(a=self.nS, size=1, p=p)[0]		

	def reset(self):
		self.t = 0
		self.x = self.x0
		return 


	def step(self, a):
		
		reward = self.rF(self.x, a)
		self.x = self.tF(self.x,a)
		return reward



if __name__ == "__main__":
	tMatrix, _ = mdptoolbox.example.rand(4,2)
	testEnv = dEnv(  tMatrix, 0, 1, -1)
	re = []
	sA = [testEnv.x]
	aA = []
	for _ in range(100):
		tempA = random.randint(0, testEnv.nA-1)
		re.append(testEnv.step(tempA))
		sA.append(testEnv.x)
		aA.append(tempA)


	
