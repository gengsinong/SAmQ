#!/bin/bash

b=5 # Intercept of the true reward function
nSamples=5000
dS=5
discount=0.8

# Get the data
python getData.py $dS $nSamples $discount $b



# squash
python squash.py

# get Theta

#python getTheta.py