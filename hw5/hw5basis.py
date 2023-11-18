import unittest
import numpy as np
import scipy.special as sp
import sympy as sym
import math



def XMap(x0,x1,xi,p):
    x = 0
    xvals = np.linspace(x0,x1,p+1)
    for j in range(0,len(xvals)):
        x += NBasis(deg=p, N_idx=j, t=xi) * xvals[j]
    return x



def XMapDerv(x0,x1,xi,p):
      x_derv = 0
      xvals = np.linspace(x0,x1,p+1)
      for a in range(0,len(xvals)):
          x_derv += NBasisDerv(deg=p, N_idx=j, t=xi) * xvals[a]
      return x_derv



def NBasis(deg, N_idx, t):
    p = deg
    a = N_idx
    pca = math.comb(p, a)
    N_term1 = pca
    N_term3 = t**a
    N_term2 = (1-t)**(p-a)
    return N_term1 * N_term2 * N_term3

class Test_Nbasis(unittest.TestCase):
    # def test_NBasis_biunit(self):
    #     goldXout = 1
    #     testXout = NBasis(xmin=-1, xmax=1, N_idx=0, x=-1)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 0.5
    #     testXout = NBasis(xmin=-1, xmax=1, N_idx=0, x=0)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 0
    #     testXout = NBasis(xmin=-1, xmax=1, N_idx=0, x=1)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 0
    #     testXout = NBasis(xmin=-1, xmax=1, N_idx=1, x=-1)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 0.5
    #     testXout = NBasis(xmin=-1, xmax=1, N_idx=1, x=0)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 1
    #     testXout = NBasis(xmin=-1, xmax=1, N_idx=1, x=1)
    #     self.assertAlmostEqual(goldXout, testXout)
    def test_NBasis_unit(self):
        goldXout = 1
        testXout = NBasis(deg=1,  N_idx=0, t=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = NBasis(deg=1,  N_idx=0, t=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = NBasis(deg=1,  N_idx=0, t=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = NBasis(deg=1,  N_idx=1, t=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = NBasis(deg=1,  N_idx=1, t=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = NBasis(deg=1,  N_idx=1, t=1)
        self.assertAlmostEqual(goldXout, testXout)
    # ADD THIS:
    # def test_NBasis_arbitrary(self):



def NBasisDerv(deg, N_idx, t):
    p = deg
    a = N_idx
    pca = math.comb(p, a)
    if a == 0:
        N_derv_t1 = 0
    else:
        N_derv_t1 = a * t**(a-1) * (1-t)**(p-a)
    if p == a:
        N_derv_t2 = 0
    else: 
        N_derv_t2 = (p-a) * t**a *(1-t)**(p-a-1)
    N_derv = pca * (N_derv_t1 - N_derv_t2)
    return N_derv

class Test_NBasisDerv(unittest.TestCase):
    def test_NBasisDerv_deg1(self):
        goldXout = -1.0
        testXout = NBasisDerv(deg=1, N_idx=0, t=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -1.0
        testXout = NBasisDerv(deg=1, N_idx=0, t=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -1.0
        testXout = NBasisDerv(deg=1, N_idx=0, t=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1.0
        testXout = NBasisDerv(deg=1, N_idx=1, t=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1.0
        testXout = NBasisDerv(deg=1, N_idx=1, t=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1.0
        testXout = NBasisDerv(deg=1, N_idx=1, t=1)
        self.assertAlmostEqual(goldXout, testXout)
    
    def test_NBasisDerv_deg2(self):
        goldXout = -2.0
        testXout = NBasisDerv(deg=2, N_idx=0, t=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -1.0
        testXout = NBasisDerv(deg=2, N_idx=0, t=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.0 # 0.0
        testXout = NBasisDerv(deg=2, N_idx=0, t=1)
        self.assertAlmostEqual(goldXout, testXout)

        goldXout = 2.0
        testXout = NBasisDerv(deg=2, N_idx=1, t=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.0
        testXout = NBasisDerv(deg=2, N_idx=1, t=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -2.0
        testXout = NBasisDerv(deg=2, N_idx=1, t=1)
        self.assertAlmostEqual(goldXout, testXout)
        
        goldXout = 0.0 
        testXout = NBasisDerv(deg=2, N_idx=2, t=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1.0
        testXout = NBasisDerv(deg=2, N_idx=2, t=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 2.0
        testXout = NBasisDerv(deg=2, N_idx=2, t=1)
        self.assertAlmostEqual(goldXout, testXout)
