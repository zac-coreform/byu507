import unittest
import numpy as np
import scipy.special as sp
import sympy as sym
import math

# def affine_map(from_domain, to_domain, from_variate) # return to_variate
# def param_to_spatial(param_domain, spatial_domain, param_value)
# def spatial_to_param(spatial_domain, param_domain, spatial_value)




def map_x_to_xi(X0,X1,x, a=-1, basis_lower=-1, basis_upper=1):
    A = X0
    B = X1
    a = basis_lower
    b = basis_upper
    xi = ((x - A) / (B - A)) * (b - a) + a
    return xi

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




def map_xi_to_x(X0,X1,xi,basis_lower=-1, basis_upper=1):
    A = X0
    B = X1
    a = basis_lower
    b = basis_upper
    X = ((xi - a) / (b - a)) * (B - A) + A
    return X

class Test_map_x_to_xi(unittest.TestCase):
    def test_biunit_to_biunit(self):
        goldXiout = -1
        testXiout = map_x_to_xi(X0=-1, X1=1, x=-1)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(X0=-1, X1=1, x=0)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(X0=-1, X1=1, x=1)
        self.assertAlmostEqual(goldXiout, testXiout)
    def test_unit_to_biunit(self):
        goldXiout = -1
        testXiout = map_x_to_xi(X0=0, X1=1, x=0)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(X0=0, X1=1, x=0.5)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(X0=0, X1=1, x=1)
        self.assertAlmostEqual(goldXiout, testXiout)
    def test_biunit_to_nontrivial(self):
        goldXiout = -1
        testXiout = map_x_to_xi(X0=3, X1=7, x=3)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(X0=3, X1=7, x=5)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(X0=3, X1=7, x=7)
        self.assertAlmostEqual(goldXiout, testXiout)




def C(n,k):
    return sp.binom(n, k)

class Test_C(unittest.TestCase):
    def test_C00(self):
        goldC00out = 1
        testC00out = C(n=0, k=0)
        self.assertAlmostEqual(goldC00out, testC00out)
    def test_C0X(self):
        goldC0Xout = 0.0
        testC0Xout_list = []
        for j in range(1,11):
            testC0Xout_list.append(C(n=0, k=j))
            # these return -0.0 for evens, 0.0 odds, always sum to 0.0
        testC0Xout = sum(testC0Xout_list)
        self.assertAlmostEqual(goldC0Xout, testC0Xout)
    def test_CX0(self):
        goldCX0out = True
        testCX0out_ = []
        for j in range(0,11):
            r = bool(C(n=j, k=0) == 1.0)
            testCX0out_.append(r)
        testCX0out = min(testCX0out_) # min is True (1) if all True, otherwise False (0)
        self.assertAlmostEqual(goldCX0out, testCX0out)
    def test_CX1(self):
        goldCX1out = True
        testCX1out_ = []
        for j in range(0,11):
            r = bool(C(n=j, k=1) == float(j))
            testCX1out_.append(r)
        testCX1out = min(testCX1out_) # min is True (1) if all True, otherwise False (0)
        self.assertAlmostEqual(goldCX1out, testCX1out)




def eval_B(xmin, xmax, deg, B_idx, x):
    xi = map_x_to_xi(xmin, xmax, x, basis_lower=0, basis_upper=1)
    i = B_idx
    n = deg
    c_term = C(n,i)
    nix_term = (1 - x)**(n - i) * x**i
    return c_term * nix_term

lmr = [0.0, 0.5, 1.0]
class Test_eval_B(unittest.TestCase):
    def test_unit_lmr_deg0_bf1(self): # lmr left middle right
        goldVals = [1.0, 0.5, 0.0]
        for j in range(0, len(lmr)):
            x = lmr[j]
            testVal = eval_B(xmin=0, xmax=1, deg=1, B_idx=0, x=x)
            self.assertAlmostEqual(testVal, goldVals[j])
    def test_unit_lmr_deg1_bf1(self):
        goldVals = [0.0, 0.5, 1.0]
        for j in range(0, len(lmr)):
            x = lmr[j]
            testVal = eval_B(xmin=0, xmax=1, deg=1, B_idx=1, x=x)
            self.assertAlmostEqual(testVal, goldVals[j])
    # deg 2 x=lmr
    def test_unit_lmr_deg2_bf0(self):
        goldVals = [1.0, 0.25, 0.0]
        for j in range(0, len(lmr)):
            x = lmr[j]
            testVal = eval_B(xmin=0, xmax=1, deg=2, B_idx=0, x=x)
            self.assertAlmostEqual(testVal, goldVals[j])
    def test_unit_lmr_deg2_bf1(self):
        goldVals = [0.0, 0.5, 0.0]
        for j in range(0, len(lmr)):
            x = lmr[j]
            testVal = eval_B(xmin=0, xmax=1, deg=2, B_idx=1, x=x)
            self.assertAlmostEqual(testVal, goldVals[j])
    def test_unit_lmr_deg2_bf2(self):
        goldVals = [0.0, 0.25, 1.0]
        for j in range(0, len(lmr)):
            x = lmr[j]
            testVal = eval_B(xmin=0, xmax=1, deg=2, B_idx=2, x=x)
            self.assertAlmostEqual(testVal, goldVals[j])
    # deg 2 x=0.75 j = bf_idx
    def test_unit_arbx_deg2_bf2(self):
        goldVals = [0.0625, 0.375, 0.5625]
        for j in range(0, len(lmr)):
            x = lmr[j]
            testVal = eval_B(xmin=0, xmax=1, deg=2, B_idx=j, x=0.75)
            self.assertAlmostEqual(testVal, goldVals[j])
# for all degrees (unit domain):
# all index 0 bfs should be 1 at x=0
# all index -1 bfs should be 1 at x=1 <--?




#####################################################
#####################################################
t,n,i = sym.symbols('t,n i')

# per the simple formula, not class formula
def C_prime(p, A):
    sum = 0
    for j in range(1, A+1):
        sum += ((-1)**(j-1) / j) * C(p, A-j)
    return sum

# class Test_C_prime(unittest.TestCase):
#     def test_unit_lmr_deg0_bf1(self):
#         ...

def B_eqn_nix(t,n,i):
    return (1 - t)**(n - i)
def B_eqn_x(t,n,i):
    return t**i
def B_eqn_nix_prime(t,n,i):
    return sym.diff(B_eqn_nix(t,n,i),t)
def B_eqn_x_prime(t,n,i):
    return sym.diff(B_eqn_x(t,n,i),t)
B_nix_prime = sym.lambdify((t,n,i),B_eqn_nix_prime(t,n,i),"numpy")
B_x_prime = sym.lambdify((t,n,i),B_eqn_x_prime(t,n,i),"numpy")

# class Test_B_nix_prime(unittest.TestCase):
#     def test_unit_lmr_deg0_bf1(self):
#         ...
# class Test_B_x_prime(unittest.TestCase):
#     def test_unit_lmr_deg0_bf1(self):
#         ...

def eval_B_deriv(xmin, xmax, deg, B_idx, x):
    # print("running eval_B_prime")
    xi = map_x_to_xi(xmin, xmax, x, basis_lower=0, basis_upper=1)
    # print("ximapping:", bool(xi == x))
    n = deg
    i = B_idx
    # terms
    c_term = C(n,i)
    nix_term = B_eqn_nix(t=x, n=n, i=i)
    x_term = B_eqn_x(t=x, n=n, i=i)
    nixx_term = nix_term * x_term
    # term derivs
    c_term_prime = C_prime(n,i)
    nix_term_prime = B_nix_prime(x, n, i)
    x_term_prime = B_x_prime(x, n, i)
    # applying nested product rule
    return (c_term * (nix_term * x_term_prime + nix_term_prime * x_term)) + c_term_prime * nixx_term

# if the symbolic forms of degs 1 and 2 bfs are as follows: 
    # B10 = 1-t               -1
    # B11 = t                 1
    # B20 = (1-t)^2           2t-2
    # B21 = 2(1-t)t           2-4t
    # B22 = t^2               2t
    # B30 = (1-t)^3           -3t^2-6t-3
    # B31 = 3(1-t)^2t         
    #  -6 (1 - t)^(2 t - 1) (t + (t - 1) log(1 - t))
    # B32 = 3(1-t)t^2         6 t (2 t^2 - 3 t + 1)
    # B33 = t^3               3t
# then these functions give the derivative as a function of t
def B10dx(t):
    return -1
def B11dx(t):
    return 1
def B20dx(t):
    return 2*t - 2
def B21dx(t):
    return 2 - 4*t
def B22dx(t):
    return 2*t
def B30dx(t):
    return -3*t**2 - 6*t - 3
def B32dx(t):
    return 6 * t * (2*t**2 - 3*t +1)
def B33dx(t):
    return t**3

class Test_eval_B_deriv(unittest.TestCase):
    def test_unit_lmr_deg1_bf0(self):
        goldBderiv = -1
        for j in np.random.random((5, 1)):
            testBderiv = eval_B_deriv(xmin=0, xmax=1, deg=1, B_idx=0, x=j)
            self.assertAlmostEqual(goldBderiv, testBderiv)
    def test_unit_lmr_deg1_bf1(self):
        goldBderiv = 1
        for j in np.random.random((5, 1)):
            testBderiv = eval_B_deriv(xmin=0, xmax=1, deg=1, B_idx=1, x=j)
            self.assertAlmostEqual(goldBderiv, testBderiv)
    # def test_unit_lmr_deg2_bf0(self):
    #     goldBderiv = 
    # def test_unit_lmr_deg2_bf1(self):
    #     goldBderiv = 
    # def test_unit_lmr_deg2_bf2(self):
    #     goldBderiv = 

    # def test_unit_lmr_deg3_bf0(self):
    #         goldBderiv = 
    # # def test_unit_lmr_deg3_bf1(self):
    # def test_unit_lmr_deg3_bf2(self):
    #         goldBderiv = 
    # def test_unit_lmr_deg3_bf3(self):
    #         goldBderiv = 







#####################################################
#####################################################




def eval_L_basis(xmin, xmax, N_idx, x):
    xi = map_x_to_xi(xmin, xmax, x)
    if N_idx == 0:
        basis_val = (1 - xi) / 2
    elif N_idx == 1:
        basis_val = (xi + 1) / 2
    return basis_val

class Test_eval_L_basis(unittest.TestCase):
    def test_eval_L_basis_biunit(self):
        goldXout = 1
        testXout = eval_L_basis(xmin=-1, xmax=1, N_idx=0, x=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis(xmin=-1, xmax=1, N_idx=0, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_L_basis(xmin=-1, xmax=1, N_idx=0, x=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_L_basis(xmin=-1, xmax=1, N_idx=1, x=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis(xmin=-1, xmax=1, N_idx=1, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = eval_L_basis(xmin=-1, xmax=1, N_idx=1, x=1)
        self.assertAlmostEqual(goldXout, testXout)
    def test_eval_L_basis_unit(self):
        goldXout = 1
        testXout = eval_L_basis(xmin=0, xmax=1, N_idx=0, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis(xmin=0, xmax=1, N_idx=0, x=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_L_basis(xmin=0, xmax=1, N_idx=0, x=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_L_basis(xmin=0, xmax=1, N_idx=1, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis(xmin=0, xmax=1, N_idx=1, x=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = eval_L_basis(xmin=0, xmax=1, N_idx=1, x=1)
        self.assertAlmostEqual(goldXout, testXout)
    # ADD THIS:
    # def test_eval_L_basis_arbitrary(self):



def eval_L_basis_deriv(xmin, xmax, N_idx, x):
    xi = map_x_to_xi(xmin, xmax, x)
    if N_idx == 0:
        basis_deriv = -0.5
    elif N_idx == 1:
        basis_deriv = 0.5
    return basis_deriv

class Test_eval_L_basis_deriv(unittest.TestCase):
    def test_eval_L_basis_deriv_biunit(self):
        goldXout = -0.5
        testXout = eval_L_basis_deriv(xmin=-1, xmax=1, N_idx=0, x=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_L_basis_deriv(xmin=-1, xmax=1, N_idx=0, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_L_basis_deriv(xmin=-1, xmax=1, N_idx=0, x=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis_deriv(xmin=-1, xmax=1, N_idx=1, x=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis_deriv(xmin=-1, xmax=1, N_idx=1, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis_deriv(xmin=-1, xmax=1, N_idx=1, x=1)
        self.assertAlmostEqual(goldXout, testXout)
    def test_eval_L_basis_deriv_unit(self):
        goldXout = -0.5
        testXout = eval_L_basis_deriv(xmin=0, xmax=1, N_idx=0, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_L_basis_deriv(xmin=0, xmax=1, N_idx=0, x=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_L_basis_deriv(xmin=0, xmax=1, N_idx=0, x=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis_deriv(xmin=0, xmax=1, N_idx=1, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis_deriv(xmin=0, xmax=1, N_idx=1, x=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_L_basis_deriv(xmin=0, xmax=1, N_idx=1, x=1)
        self.assertAlmostEqual(goldXout, testXout)

