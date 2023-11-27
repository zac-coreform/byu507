import unittest
import numpy as np
import sys

# def affine_map(from_domain, to_domain, from_variate) # return to_variate
# def param_to_spatial(param_domain, spatial_domain, param_value)
# def spatial_to_param(spatial_domain, param_domain, spatial_value)

# evaluate basis bf_idx at xi in param -1,1 equiv to x_in from xmin, xmax
def eval_basis(bf_idx, xmin, xmax, x_in):
    xi = map_x_to_xi(xmin, xmax, x_in)
    # print(f"x_in of {x_in} mapped to xi={xi}")
    if bf_idx == 0:
        basis_val = (1 - xi) / 2
    elif bf_idx == 1:
        basis_val = (xi + 1) / 2
    return basis_val

def eval_basis_deriv(xmin, xmax, bf_idx, x_in):
    # X_dom = [xmin, xmax]
    # xi = map_x_to_xi(xmin, xmax, x_in)
    if bf_idx == 0:
        basis_deriv = -1 / 2
    elif bf_idx == 1:
        basis_deriv = 1 / 2
    return basis_deriv

def map_x_to_xi(X0,X1,x_in):
    dom_test = X0 <= x_in <= X1
    if not dom_test:
        sys.exit(f"map_x_to_xi error: x_in was {x_in}, but must be between {X0} and {X1}")
    xi0 = -1
    xi1 = 1
    xi = ( (x_in - X0) * (xi1 - xi0) / (X1 - X0) ) + xi0
    return xi

#     return xi
# with D = b and E = a, this is: 
# (((x - A) / (B - A)) * (D - E) + E)
# and the derivative is:
# (E-D)/(A-B)
# translated back, that's 
# ((a - b) / (A - B))

# def map_x_to_xi_deriv(X0,X1,X_in):
#     xi0 = -1
#     xi1 = 1

def map_x_to_xi_deriv(X0,X1,X_in):
    xi0 = -1
    xi1 = 1
    print(f"len is {X1-X0}")
    print(f"denom is {xi1 - xi0}")
    deriv = ((X1 - X0) / (xi1 - xi0))
    return deriv

#     deriv = ((xi1 - xi0) / (X0 - X1))
#     return deriv

def map_xi_to_x(X0,X1,xi):
    A = X0
    B = X1
    xi0 = -1
    xi1 = 1
    X = ((xi - xi0) / (xi1 - xi0)) * (B - A) + A
    return X

class Test_map_xi_to_x(unittest.TestCase):
    def test_biunit_to_biunit(self):
        goldXout = -1
        testXout = map_xi_to_x(X0=-1, X1=1, xi=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = map_xi_to_x(X0=-1, X1=1, xi=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = map_xi_to_x(X0=-1, X1=1, xi=1)
        self.assertAlmostEqual(goldXout, testXout)
    def test_biunit_to_unit(self):
        goldXout = 0
        testXout = map_xi_to_x(X0=0, X1=1, xi=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = map_xi_to_x(X0=0, X1=1, xi=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = map_xi_to_x(X0=0, X1=1, xi=1)
        self.assertAlmostEqual(goldXout, testXout)
    def test_biunit_to_nontrivial(self):
        goldXout = 3
        testXout = map_xi_to_x(X0=3, X1=7, xi=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 5
        testXout = map_xi_to_x(X0=3, X1=7, xi=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 7
        testXout = map_xi_to_x(X0=3, X1=7, xi=1)
        self.assertAlmostEqual(goldXout, testXout)

class Test_map_x_to_xi(unittest.TestCase):
    def test_biunit_to_biunit(self):
        goldXiout = -1
        testXiout = map_x_to_xi(X0=-1, X1=1, x_in=-1)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(X0=-1, X1=1, x_in=0)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(X0=-1, X1=1, x_in=1)
        self.assertAlmostEqual(goldXiout, testXiout)
    def test_unit_to_biunit(self):
        goldXiout = -1
        testXiout = map_x_to_xi(X0=0, X1=1, x_in=0)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(X0=0, X1=1, x_in=0.5)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(X0=0, X1=1, x_in=1)
        self.assertAlmostEqual(goldXiout, testXiout)
    def test_nontrivial_to_biunit(self):
        goldXiout = -1
        testXiout = map_x_to_xi(X0=3, X1=7, x_in=3)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(X0=3, X1=7, x_in=5)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(X0=3, X1=7, x_in=7)
        self.assertAlmostEqual(goldXiout, testXiout)

class Test_eval_basis(unittest.TestCase):
    def test_eval_basis_biunit(self):
        goldXout = 1
        testXout = eval_basis(bf_idx=0, xmin=-1, xmax=1, x_in=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis(bf_idx=0, xmin=-1, xmax=1, x_in=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_basis(bf_idx=0, xmin=-1, xmax=1, x_in=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_basis(bf_idx=1, xmin=-1, xmax=1, x_in=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis(bf_idx=1, xmin=-1, xmax=1, x_in=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = eval_basis(bf_idx=1, xmin=-1, xmax=1, x_in=1)
        self.assertAlmostEqual(goldXout, testXout)
    # def test_eval_basis_unit(self):
    #     goldXout = 1
    #     testXout = eval_basis(bf_idx=0, xmin=0, xmax=1, x_in=0)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 0.5
    #     testXout = eval_basis(bf_idx=0, xmin=0, xmax=1, x_in=0.5)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 0
    #     testXout = eval_basis(bf_idx=0, xmin=0, xmax=1, x_in=1)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 0
    #     testXout = eval_basis(bf_idx=1, xmin=0, xmax=1, x_in=0)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 0.5
    #     testXout = eval_basis(bf_idx=1, xmin=0, xmax=1, x_in=0.5)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 1
    #     testXout = eval_basis(bf_idx=1, xmin=0, xmax=1, x_in=1)
    #     self.assertAlmostEqual(goldXout, testXout)
    # ADD THIS:
    # def test_eval_basis_arbitrary(self):

class Test_eval_basis_deriv(unittest.TestCase):
    def test_eval_basis_deriv_biunit(self):
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, bf_idx=0, x_in=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, bf_idx=0, x_in=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, bf_idx=0, x_in=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, bf_idx=1, x_in=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, bf_idx=1, x_in=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, bf_idx=1, x_in=1)
        self.assertAlmostEqual(goldXout, testXout)
    
    # def test_eval_basis_deriv_unit(self):
    #     goldXout = -1.0
    #     testXout = eval_basis_deriv(xmin=0, xmax=1, bf_idx=0, x_in=0)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = -1.0
    #     testXout = eval_basis_deriv(xmin=0, xmax=1, bf_idx=0, x_in=0.5)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = -1.0
    #     testXout = eval_basis_deriv(xmin=0, xmax=1, bf_idx=0, x_in=1)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 1.0
    #     testXout = eval_basis_deriv(xmin=0, xmax=1, bf_idx=1, x_in=0)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 1.0
    #     testXout = eval_basis_deriv(xmin=0, xmax=1, bf_idx=1, x_in=0.5)
    #     self.assertAlmostEqual(goldXout, testXout)
    #     goldXout = 1.0
    #     testXout = eval_basis_deriv(xmin=0, xmax=1, bf_idx=1, x_in=1)
    #     self.assertAlmostEqual(goldXout, testXout)

# hughes eqns version

def x_xi_deriv(e_x0, e_x1):
    h_e = e_x1 - e_x0
    return  h_e / 2

def x_xi_deriv_inv(e_x0, e_x1): 
    h_e = e_x1 - e_x0
    return 2 / h_e

def N_a_xi(idx):
    neg = (-1)**idx
    return neg / 2

# class Test_hughes_vs_(unittest.TestCase):

# SO FOR AN ELEMENT NOW WE SHOULD GET THAT THE KE IS JUST 1 / H_E * STOCK MATRIX OF 1, -1

def mxm(e_x0, e_x1):
    h_e = e_x1 - e_x0
    sol = 1 / h_e
    return sol

stock_mx_ = np.array(([1, -1], [-1, 1]))

def stock_mx():
    return np.array(([1, -1], [-1, 1]))

def ke_fast(e_x0, e_x1):
    mult = mxm(e_x0, e_x1)
    ke = mult * stock_mx
    return ke