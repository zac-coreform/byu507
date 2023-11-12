import unittest
import numpy as np

# def affine_map(from_domain, to_domain, from_variate) # return to_variate
# def param_to_spatial(param_domain, spatial_domain, param_value)
# def spatial_to_param(spatial_domain, param_domain, spatial_value)

def eval_basis(xmin, xmax, N_idx, x):
    xi = map_x_to_xi(xmin, xmax, x)
    if N_idx == 0:
        basis_val = (1 - xi) / 2
    elif N_idx == 1:
        basis_val = (xi + 1) / 2
    return basis_val

def eval_basis_deriv(xmin, xmax, N_idx, x):
    xi = map_x_to_xi(xmin, xmax, x)
    if N_idx == 0:
        basis_deriv = -0.5
    elif N_idx == 1:
        basis_deriv = 0.5
    return basis_deriv

def map_x_to_xi(x0,x1,x):
    A = x0
    B = x1
    a = -1
    b = 1
    xi = ((x - A) / (B - A)) * (b - a) + a
    return xi

def map_xi_to_x(x0,x1,xi):
    A = x0
    B = x1
    a = -1
    b = 1
    X = ((xi - a) / (b - a)) * (B - A) + A
    return X

class Test_map_xi_to_x(unittest.TestCase):
    def test_biunit_to_biunit(self):
        goldXout = -1
        testXout = map_xi_to_x(x0=-1, x1=1, xi=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = map_xi_to_x(x0=-1, x1=1, xi=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = map_xi_to_x(x0=-1, x1=1, xi=1)
        self.assertAlmostEqual(goldXout, testXout)
    def test_biunit_to_unit(self):
        goldXout = 0
        testXout = map_xi_to_x(x0=0, x1=1, xi=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = map_xi_to_x(x0=0, x1=1, xi=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = map_xi_to_x(x0=0, x1=1, xi=1)
        self.assertAlmostEqual(goldXout, testXout)
    def test_biunit_to_nontrivial(self):
        goldXout = 3
        testXout = map_xi_to_x(x0=3, x1=7, xi=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 5
        testXout = map_xi_to_x(x0=3, x1=7, xi=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 7
        testXout = map_xi_to_x(x0=3, x1=7, xi=1)
        self.assertAlmostEqual(goldXout, testXout)

class Test_map_x_to_xi(unittest.TestCase):
    def test_biunit_to_biunit(self):
        goldXiout = -1
        testXiout = map_x_to_xi(x0=-1, x1=1, x=-1)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(x0=-1, x1=1, x=0)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(x0=-1, x1=1, x=1)
        self.assertAlmostEqual(goldXiout, testXiout)
    def test_unit_to_biunit(self):
        goldXiout = -1
        testXiout = map_x_to_xi(x0=0, x1=1, x=0)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(x0=0, x1=1, x=0.5)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(x0=0, x1=1, x=1)
        self.assertAlmostEqual(goldXiout, testXiout)
    def test_biunit_to_nontrivial(self):
        goldXiout = -1
        testXiout = map_x_to_xi(x0=3, x1=7, x=3)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 0
        testXiout = map_x_to_xi(x0=3, x1=7, x=5)
        self.assertAlmostEqual(goldXiout, testXiout)
        goldXiout = 1
        testXiout = map_x_to_xi(x0=3, x1=7, x=7)
        self.assertAlmostEqual(goldXiout, testXiout)

class Test_eval_basis(unittest.TestCase):
    def test_eval_basis_biunit(self):
        goldXout = 1
        testXout = eval_basis(xmin=-1, xmax=1, N_idx=0, x=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis(xmin=-1, xmax=1, N_idx=0, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_basis(xmin=-1, xmax=1, N_idx=0, x=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_basis(xmin=-1, xmax=1, N_idx=1, x=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis(xmin=-1, xmax=1, N_idx=1, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = eval_basis(xmin=-1, xmax=1, N_idx=1, x=1)
        self.assertAlmostEqual(goldXout, testXout)
    def test_eval_basis_unit(self):
        goldXout = 1
        testXout = eval_basis(xmin=0, xmax=1, N_idx=0, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis(xmin=0, xmax=1, N_idx=0, x=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_basis(xmin=0, xmax=1, N_idx=0, x=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0
        testXout = eval_basis(xmin=0, xmax=1, N_idx=1, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis(xmin=0, xmax=1, N_idx=1, x=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 1
        testXout = eval_basis(xmin=0, xmax=1, N_idx=1, x=1)
        self.assertAlmostEqual(goldXout, testXout)
    # def test_eval_basis_arbitrary(self):

class Test_eval_basis_deriv(unittest.TestCase):
    def test_eval_basis_deriv_biunit(self):
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, N_idx=0, x=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, N_idx=0, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, N_idx=0, x=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, N_idx=1, x=-1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, N_idx=1, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=-1, xmax=1, N_idx=1, x=1)
        self.assertAlmostEqual(goldXout, testXout)
    def test_eval_basis_deriv_unit(self):
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=0, xmax=1, N_idx=0, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=0, xmax=1, N_idx=0, x=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = -0.5
        testXout = eval_basis_deriv(xmin=0, xmax=1, N_idx=0, x=1)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=0, xmax=1, N_idx=1, x=0)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=0, xmax=1, N_idx=1, x=0.5)
        self.assertAlmostEqual(goldXout, testXout)
        goldXout = 0.5
        testXout = eval_basis_deriv(xmin=0, xmax=1, N_idx=1, x=1)
        self.assertAlmostEqual(goldXout, testXout)