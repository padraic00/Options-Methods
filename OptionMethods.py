import numpy as np
import math as m
import timeit
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt

m=5 #time steps
K=100 #strike price
r=0.05 #risk free rate
T=20/36 #strike time
sigma=0.30 #implied volatiity

time = scipy.linspace (0.0, T , m ) #time series
S = 100 # stock price

logSoverK = scipy . log ( S / K )
n12 = (( r + sigma **2/2) *( T - time ) )
n22 = (( r - sigma **2/2) *( T - time ) )
numerd1 = logSoverK + n12 
numerd2 = logSoverK  + n22
d1 = numerd1 /( sigma * scipy . sqrt (T - time )) 
d2 = numerd2 /( sigma * scipy . sqrt (T - time ))

part1 = S * norm . cdf ( d1 )
part2 = norm.cdf(d2) * K * scipy.exp( - r *( T - time ) ) 
VC=part1-part2