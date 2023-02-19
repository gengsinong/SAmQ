# Utility functions for environment. 
import numpy as np
import random

# Get data

def dataGen(env, nS):
	re = []
	s = [env.x]
	a = []
	for _ in range(nS):
		tempA = random.randint(0, env.nA)
		re.append(env.step(tempA))
		s.append(env.x)
		a.append(tempA)

	return re, s, a