import unittest
import numpy as np
import scipy.special as sp
import sympy as sym
import math
import hw7Bernstein_TestValues as bd
import sys


# ----------------------------------------
def XMap(x0,x1,xi,p):
    x = 0
    xvals = np.linspace(x0,x1,p+1)
    for a in range(0,len(xvals)):
        x += NBasis(deg=p, N_idx=a, t=xi) * xvals[a]
    return x

# ----------------------------------------
def XMapDerv(x0,x1,xi,p):
      x_derv = 0
      xvals = np.linspace(x0,x1,p+1)
      for a in range(0,len(xvals)):
          x_derv += NBasisDerv(deg=p, N_idx=a, t=xi) * xvals[a]
      return x_derv


# ----------------------------------------
def NBasis(deg, N_idx, t):
    p = deg
    a = N_idx
    pca = math.comb(p, a)
    N_term1 = pca
    N_term3 = t**a
    N_term2 = (1-t)**(p-a)
    return N_term1 * N_term2 * N_term3


# ----------------------------------------
def NBasisDerv(deg, N_idx, t):
    p = deg
    a = N_idx
    pca = math.comb(p, a)
    N_derv_t1 = 0 if a==0 else (a * t**(a-1) * (1-t)**(p-a))
    N_derv_t2 = 0 if a==p else ((p-a) * t**a *(1-t)**(p-a-1))
    N_derv = pca * (N_derv_t1 - N_derv_t2)
    return N_derv


# ----------------------------------------
# ----------------------------------------
# test points
# 0.00
# 0.25
# 0.67
# 1.00

# ----------------------------------------
b1_vals = bd.b1_tvals_list
b2_vals = bd.b2_tvals_list
b3_vals = bd.b3_tvals_list
b4_vals = bd.b4_tvals_list
b5_vals = bd.b5_tvals_list

class Test_NBasis(unittest.TestCase):
    def test_NBasis_deg1(self):
        vals = b1_vals
        # deg 1, a=0
        goldXout = vals[0][0]
        testXout = NBasis(deg=1, N_idx=0, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][1]
        testXout = NBasis(deg=1, N_idx=0, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][2]
        testXout = NBasis(deg=1, N_idx=0, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][3]
        testXout = NBasis(deg=1, N_idx=0, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 1, a=0
        goldXout = vals[1][0]
        testXout = NBasis(deg=1, N_idx=1, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][1]
        testXout = NBasis(deg=1, N_idx=1, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][2]
        testXout = NBasis(deg=1, N_idx=1, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][3]
        testXout = NBasis(deg=1, N_idx=1, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_NBasis_deg2(self):
        vals = b2_vals
        # deg 2, a=0
        goldXout = vals[0][0]
        testXout = NBasis(deg=2, N_idx=0, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][1]
        testXout = NBasis(deg=2, N_idx=0, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][2]
        testXout = NBasis(deg=2, N_idx=0, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][3]
        testXout = NBasis(deg=2, N_idx=0, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 2, a=1
        goldXout = vals[1][0]
        testXout = NBasis(deg=2, N_idx=1, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][1]
        testXout = NBasis(deg=2, N_idx=1, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][2]
        testXout = NBasis(deg=2, N_idx=1, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][3]
        testXout = NBasis(deg=2, N_idx=1, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 2, a=2
        goldXout = vals[2][0]
        testXout = NBasis(deg=2, N_idx=2, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][1]
        testXout = NBasis(deg=2, N_idx=2, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][2]
        testXout = NBasis(deg=2, N_idx=2, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][3]
        testXout = NBasis(deg=2, N_idx=2, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_NBasis_deg3(self):
        vals = b3_vals
        # deg 3, a=0
        goldXout = vals[0][0]
        testXout = NBasis(deg=3, N_idx=0, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][1]
        testXout = NBasis(deg=3, N_idx=0, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][2]
        testXout = NBasis(deg=3, N_idx=0, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][3]
        testXout = NBasis(deg=3, N_idx=0, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 3, a=1
        goldXout = vals[1][0]
        testXout = NBasis(deg=3, N_idx=1, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][1]
        testXout = NBasis(deg=3, N_idx=1, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][2]
        testXout = NBasis(deg=3, N_idx=1, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][3]
        testXout = NBasis(deg=3, N_idx=1, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 3, a=2
        goldXout = vals[2][0]
        testXout = NBasis(deg=3, N_idx=2, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][1]
        testXout = NBasis(deg=3, N_idx=2, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][2]
        testXout = NBasis(deg=3, N_idx=2, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][3]
        testXout = NBasis(deg=3, N_idx=2, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 3, a=3
        goldXout = vals[3][0]
        testXout = NBasis(deg=3, N_idx=3, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][1]
        testXout = NBasis(deg=3, N_idx=3, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][2]
        testXout = NBasis(deg=3, N_idx=3, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][3]
        testXout = NBasis(deg=3, N_idx=3, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_NBasis_deg4(self):
        vals = b4_vals
        # deg=4, a=0
        goldXout = vals[0][0]
        testXout = NBasis(deg=4, N_idx=0, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][1]
        testXout = NBasis(deg=4, N_idx=0, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][2]
        testXout = NBasis(deg=4, N_idx=0, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][3]
        testXout = NBasis(deg=4, N_idx=0, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=4, a=1
        goldXout = vals[1][0]
        testXout = NBasis(deg=4, N_idx=1, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][1]
        testXout = NBasis(deg=4, N_idx=1, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][2]
        testXout = NBasis(deg=4, N_idx=1, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][3]
        testXout = NBasis(deg=4, N_idx=1, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=4, a=2
        goldXout = vals[2][0]
        testXout = NBasis(deg=4, N_idx=2, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][1]
        testXout = NBasis(deg=4, N_idx=2, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][2]
        testXout = NBasis(deg=4, N_idx=2, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][3]
        testXout = NBasis(deg=4, N_idx=2, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=4, a=3
        goldXout = vals[3][0]
        testXout = NBasis(deg=4, N_idx=3, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][1]
        testXout = NBasis(deg=4, N_idx=3, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][2]
        testXout = NBasis(deg=4, N_idx=3, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][3]
        testXout = NBasis(deg=4, N_idx=3, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=4, a=4
        goldXout = vals[4][0]
        testXout = NBasis(deg=4, N_idx=4, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][1]
        testXout = NBasis(deg=4, N_idx=4, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][2]
        testXout = NBasis(deg=4, N_idx=4, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][3]
        testXout = NBasis(deg=4, N_idx=4, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_NBasis_deg5(self):
        vals = b5_vals
        # deg=5, a=0
        goldXout = vals[0][0]
        testXout = NBasis(deg=5, N_idx=0, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][1]
        testXout = NBasis(deg=5, N_idx=0, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][2]
        testXout = NBasis(deg=5, N_idx=0, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][3]
        testXout = NBasis(deg=5, N_idx=0, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=5, a=1
        goldXout = vals[1][0]
        testXout = NBasis(deg=5, N_idx=1, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][1]
        testXout = NBasis(deg=5, N_idx=1, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][2]
        testXout = NBasis(deg=5, N_idx=1, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][3]
        testXout = NBasis(deg=5, N_idx=1, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=5, a=2
        goldXout = vals[2][0]
        testXout = NBasis(deg=5, N_idx=2, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][1]
        testXout = NBasis(deg=5, N_idx=2, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][2]
        testXout = NBasis(deg=5, N_idx=2, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][3]
        testXout = NBasis(deg=5, N_idx=2, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=5, a=3
        goldXout = vals[3][0]
        testXout = NBasis(deg=5, N_idx=3, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][1]
        testXout = NBasis(deg=5, N_idx=3, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][2]
        testXout = NBasis(deg=5, N_idx=3, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][3]
        testXout = NBasis(deg=5, N_idx=3, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=5, a=4
        goldXout = vals[4][0]
        testXout = NBasis(deg=5, N_idx=4, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][1]
        testXout = NBasis(deg=5, N_idx=4, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][2]
        testXout = NBasis(deg=5, N_idx=4, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][3]
        testXout = NBasis(deg=5, N_idx=4, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=5, a=5
        goldXout = vals[5][0]
        testXout = NBasis(deg=5, N_idx=5, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[5][1]
        testXout = NBasis(deg=5, N_idx=5, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[5][2]
        testXout = NBasis(deg=5, N_idx=5, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[5][3]
        testXout = NBasis(deg=5, N_idx=5, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)

# ----------------------------------------
b2d_vals = bd.b2d_tvals_list
b3d_vals = bd.b3d_tvals_list
b4d_vals = bd.b4d_tvals_list

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
        # [[−2.0, −1.5, −0.66, 0.0], [2.0, 1.0, −0.68, −2.0], [0.0, 0.5, 1.34, 2.0]]
        # deg 2, a=0
        goldXout = -2.0
        testXout = NBasisDerv(deg=2, N_idx=0, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -1.5
        testXout = NBasisDerv(deg=2, N_idx=0, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.66
        testXout = NBasisDerv(deg=2, N_idx=0, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.0
        testXout = NBasisDerv(deg=2, N_idx=0, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 2, a=1
        goldXout = 2.0
        testXout = NBasisDerv(deg=2, N_idx=1, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1.0
        testXout = NBasisDerv(deg=2, N_idx=1, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.68
        testXout = NBasisDerv(deg=2, N_idx=1, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -2.0
        testXout = NBasisDerv(deg=2, N_idx=1, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 2, a=2
        goldXout = 0.0 
        testXout = NBasisDerv(deg=2, N_idx=2, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = NBasisDerv(deg=2, N_idx=2, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1.34
        testXout = NBasisDerv(deg=2, N_idx=2, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 2.0
        testXout = NBasisDerv(deg=2, N_idx=2, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_NBasisDerv_deg3(self):
        # [[-3.0, -1.6875, -0.3266999999999999, -0.0],
        #  [3.0, 0.5625, -0.9999000000000002, 0.0],
        #  [0.0, 0.9375, -0.02010000000000023, -3.0],
        #  [0.0, 0.1875, 1.3467000000000002, 3.0]]
        # deg 3, a=0
        goldXout = -3.0
        testXout = NBasisDerv(deg=3, N_idx=0, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -1.6875
        testXout = NBasisDerv(deg=3, N_idx=0, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.3266999999999999
        testXout = NBasisDerv(deg=3, N_idx=0, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.0
        testXout = NBasisDerv(deg=3, N_idx=0, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 3, a=1
        goldXout = 3.0
        testXout = NBasisDerv(deg=3, N_idx=1, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5625
        testXout = NBasisDerv(deg=3, N_idx=1, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.9999000000000002
        testXout = NBasisDerv(deg=3, N_idx=1, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.0
        testXout = NBasisDerv(deg=3, N_idx=1, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 3, a=2
        goldXout = 0.0
        testXout = NBasisDerv(deg=3, N_idx=2, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.9375
        testXout = NBasisDerv(deg=3, N_idx=2, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.02010000000000023
        testXout = NBasisDerv(deg=3, N_idx=2, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -3.0
        testXout = NBasisDerv(deg=3, N_idx=2, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg 3, a=3
        goldXout = 0.0
        testXout = NBasisDerv(deg=3, N_idx=3, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.1875
        testXout = NBasisDerv(deg=3, N_idx=3, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1.346700000000000
        testXout = NBasisDerv(deg=3, N_idx=3, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 3.0
        testXout = NBasisDerv(deg=3, N_idx=3, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_NBasisDerv_deg4(self):
        vals = [[-4.0, -1.6875, -0.14374799999999996, -0.0],
                [4.0, 0.0, -0.731808, 0.0],
                [0.0, 1.125, -0.9020880000000009, 0.0],
                [0.0, 0.5, 0.574592, -4.0],
                [0.0, 0.0625, 1.2030520000000002, 4.0]]
        # deg=4, a=0
        goldXout = vals[0][0]
        testXout = NBasisDerv(deg=4, N_idx=0, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][1]
        testXout = NBasisDerv(deg=4, N_idx=0, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][2]
        testXout = NBasisDerv(deg=4, N_idx=0, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[0][3]
        testXout = NBasisDerv(deg=4, N_idx=0, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=4, a=1
        goldXout = vals[1][0]
        testXout = NBasisDerv(deg=4, N_idx=1, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][1]
        testXout = NBasisDerv(deg=4, N_idx=1, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][2]
        testXout = NBasisDerv(deg=4, N_idx=1, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1][3]
        testXout = NBasisDerv(deg=4, N_idx=1, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=4, a=2
        goldXout = vals[2][0]
        testXout = NBasisDerv(deg=4, N_idx=2, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][1]
        testXout = NBasisDerv(deg=4, N_idx=2, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][2]
        testXout = NBasisDerv(deg=4, N_idx=2, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2][3]
        testXout = NBasisDerv(deg=4, N_idx=2, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=4, a=3
        goldXout = vals[3][0]
        testXout = NBasisDerv(deg=4, N_idx=3, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][1]
        testXout = NBasisDerv(deg=4, N_idx=3, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][2]
        testXout = NBasisDerv(deg=4, N_idx=3, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3][3]
        testXout = NBasisDerv(deg=4, N_idx=3, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)
        # deg=4, a=4
        goldXout = vals[4][0]
        testXout = NBasisDerv(deg=4, N_idx=4, t=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][1]
        testXout = NBasisDerv(deg=4, N_idx=4, t=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][2]
        testXout = NBasisDerv(deg=4, N_idx=4, t=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[4][3]
        testXout = NBasisDerv(deg=4, N_idx=4, t=1.00)
        self.assertAlmostEqual(goldXout, testXout)

u2u_vals = bd.u2u_
ugu_vals = bd.ugu_
arb_vals = bd.arb_

def cycle_xmap_degrees(dom, xi):
    degs = [1, 2, 3, 4, 5]
    loop = len(degs)
    x0 = dom[0]
    x1 = dom[1]
    deg_1_reference = XMap(x0, x1, xi, 1)
    # print(f"ref is {deg_1_reference}")
    val_out = 0    
    for i in range(0, loop):
        p = degs[i]
        # print(f"p = {p}")
        i_x = XMap(x0, x1, xi, p)
        # print(f"i_x is {i_x}")
        if math.isclose(deg_1_reference, i_x):
            val_out = i_x
        else:
            val_out = 99
    return val_out

class Test_XMap(unittest.TestCase):
    def test_XMap_dom_u2u(self):
        vals = u2u_vals
        
        goldXout = vals[0]
        testXout = cycle_xmap_degrees(dom=bd.dom_u2u, xi=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1]
        testXout = cycle_xmap_degrees(dom=bd.dom_u2u, xi=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2]
        testXout = cycle_xmap_degrees(dom=bd.dom_u2u, xi=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3]
        testXout = cycle_xmap_degrees(dom=bd.dom_u2u, xi=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_XMap_dom_ugu(self):
        vals = ugu_vals
        goldXout = vals[0]
        testXout = cycle_xmap_degrees(dom=bd.dom_ugu, xi=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1]
        testXout = cycle_xmap_degrees(dom=bd.dom_ugu, xi=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2]
        testXout = cycle_xmap_degrees(dom=bd.dom_ugu, xi=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3]
        testXout = cycle_xmap_degrees(dom=bd.dom_ugu, xi=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_XMap_dom_arb(self):
        vals = arb_vals
        goldXout = vals[0]
        testXout = cycle_xmap_degrees(dom=bd.dom_arb, xi=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[1]
        testXout = cycle_xmap_degrees(dom=bd.dom_arb, xi=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[2]
        testXout = cycle_xmap_degrees(dom=bd.dom_arb, xi=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = vals[3]
        testXout = cycle_xmap_degrees(dom=bd.dom_arb, xi=1.00)
        self.assertAlmostEqual(goldXout, testXout)

def cycle_xmapderv_degrees(dom, xi):
    degs = [1, 2, 3, 4, 5]
    loop = len(degs)
    x0 = dom[0]
    x1 = dom[1]
    deg_1_reference = XMapDerv(x0, x1, xi, 1)
    val_out = 0    
    for i in range(0, loop):
        p = degs[i]
        i_dx = XMapDerv(x0, x1, xi, p)
        if math.isclose(deg_1_reference, i_dx):
            val_out = i_dx
        else:
            val_out = 99
    return val_out

u2u_val = bd.xmap_derv_pdom_testvals(bd.dom_u2u)
ugu_val = bd.xmap_derv_pdom_testvals(bd.dom_ugu)
arb_val = bd.xmap_derv_pdom_testvals(bd.dom_arb)

class Test_XMapDerv(unittest.TestCase):
    def test_XMapDerv_dom_u2u(self):
        val = u2u_val
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_u2u, xi=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_u2u, xi=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_u2u, xi=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_u2u, xi=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_XMapDerv_dom_ugu(self):
        val = ugu_val
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_ugu, xi=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_ugu, xi=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_ugu, xi=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_ugu, xi=1.00)
        self.assertAlmostEqual(goldXout, testXout)

    def test_XMapDerv_dom_arb(self):
        val = arb_val
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_arb, xi=0.00)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_arb, xi=0.25)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_arb, xi=0.67)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = val
        testXout = cycle_xmapderv_degrees(dom=bd.dom_arb, xi=1.00)
        self.assertAlmostEqual(goldXout, testXout)
