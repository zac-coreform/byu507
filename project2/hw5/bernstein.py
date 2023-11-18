import numpy as np
import math
import scipy.special as sp

def C(n,k):
    return sp.binom(n, k)

def Bn(idx, deg, x):
    i = idx
    n = deg
    c_term = C(n,i)
    return c_term * (1 - x)**(n - i) * x**i

# Bn(idx=0, deg=0, x=99) # 1
# Bn(idx=0, deg=1, x=0.5) == Bn(idx=1, deg=1, x=0.5) == 0.5# 
# Bn(0,2,0.5) # 0.25

