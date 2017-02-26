# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:57:42 2017
Monte Carlo Integration in a regression problem
Koop Poirier Tobias - Bayesian Econometric Methods
Chapter 11 Question 3
Code adapted from MATLAB, then attempted to create same procedure in python 

@author: Francisco Ilabaca

First, generate an artificial data set of size N = 100 from a normal linear 
regression model
$ y = X \beta + \epsilon $
set $\beta_1$ to 0
slope $ \beta_2$ to 1
error precision h = 1

"""
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels as sm
import matplotlib.pyplot as plt
import seaborn as sn


#%% Generate artificial Data set
np.random.seed(12345) # set the seed so that we can replicate results

#Initial parameters from the question
N = 100
beta1 = 0
beta2 = 1 
h = 1 
sigma = np.sqrt(1/h)

# Initial vectors
x = np.ones((N,2), dtype=np.float64)
y = np.zeros((N,1), dtype=np.float64)

# loop to generate data
for i in range(N):
    x[i,1] = np.random.uniform(low=0,high=1)
    e = sigma* np.random.normal(0,1)
    y[i,0] = beta1 + beta2*x[i,1] + e

data = np.concatenate((y,x), axis = 1)

#%% Posterior mean
"""
Now, we calculate the posterior meand and standard deviation for the slope 
coefficient beta2 for this data set.
We do this using a normal-gamma prior with beta_ = [0 1]'
V_ = I2, s^-2 =1 and nu = 1
"""
k = x.ndim # set the number of dimensions for x
# Initialize our hyper parameters, using 0 to denote hyperparameters for prior
v_0 = 1
s2inv_0 = 1
s02 = 1/s2inv_0
b0 = np.zeros((k,1), dtype=np.float64)
b0[k-1,0] = 1 # set value to 1
c = 1
capv0 = c*np.identity(k, dtype=float)


# Parameters for Normal Gamma Posterior
# first, do some OLS since some posterior results are written in terms of OLS
xsquare = np.dot(x.transpose(), x)
xsquare_inv = np.linalg.inv(xsquare)

bols = (xsquare_inv @ x.transpose()) @ y # simple OLS
s2 = np.transpose(y - x@bols)@(y-x@bols)/(N-k) # SSE / (N - k)
v= N - k
