import unittest
import numpy as np
import basis

import numpy.polynomial.legendre as lg

def get_gauss_quadrature(n_quad):
    pts = lg.leggauss(n_quad)[0]
    wts = lg.leggauss(n_quad)[1]
    return pts, wts

def get_jacobian(X0, X1):
    xi_lower = -1
    xi_upper = 1
    jacobian = (X1 - X0) / (xi_upper - xi_lower)
    return jacobian


def integrate_by_quadrature(function, x_lower, x_upper, n_quad):
    quad = get_gauss_quadrature(n_quad)
    gauss_points = quad[0]
    gauss_weights = quad[1]
    jacobian = get_jacobian(x_lower, x_upper)
    integral = 0
    for p in range(0, len(gauss_points)): 
        x = basis.map_xi_to_x(X0=x_lower, X1=x_upper, xi=gauss_points[p])
        function_val = function(x)
        partial_integral = function_val * gauss_weights[p] * jacobian
        integral += partial_integral
    return integral

class Test_get_gauss_quadrature(unittest.TestCase):
    def test_1quad(self):
        goldPoints = np.array((0.))
        goldWeights = np.array((2.))
        TestPoints, TestWeights = get_gauss_quadrature(1)
        self.assertTrue(np.allclose(goldPoints, TestPoints))
        self.assertTrue(np.allclose(goldWeights, TestWeights))
    def test_2quad(self):
        goldPoints = np.array((-1/np.sqrt(3), 1/np.sqrt(3)))
        goldWeights = np.array((1., 1.))
        TestPoints, TestWeights = get_gauss_quadrature(2)
        self.assertTrue(np.allclose(goldPoints, TestPoints))
        self.assertTrue(np.allclose(goldWeights, TestWeights))
    def test_3quad(self):
        goldPoints = np.array((-np.sqrt(3/5), 0, np.sqrt(3/5)))
        goldWeights = np.array((5/9, 8/9, 5/9))
        TestPoints, TestWeights = get_gauss_quadrature(3)
        self.assertTrue(np.allclose(goldPoints, TestPoints))
        self.assertTrue(np.allclose(goldWeights, TestWeights))


class Test_integrate_by_quadrature(unittest.TestCase):
    def test_constant_polynomial(self):
        function = lambda x: 1
        goldInt = 2
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_linear_polynomial(self):
        function = lambda x: x + 1
        goldInt = 2
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_quadratic_polynomial(self):
        function = lambda x: x**2
        goldInt = 2/3
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=2)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_cubic_polynomial(self):
        function = lambda x: x**3 + 1
        goldInt = 2
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=2)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_quartic_polynomial(self):
        function = lambda x: x**4 + 1
        goldInt = 12/5
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=3)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_quintic_polynomial(self):
        function = lambda x: x**5 + 1
        goldInt = 2
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=3)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_eval_basis_biunit(self):
        function = lambda x: basis.eval_basis(xmin=-1, xmax=1, N_idx=0, x=x)
        goldInt = 1
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_eval_basis_unit(self):
        function = lambda x: basis.eval_basis(xmin=0, xmax=1, N_idx=0, x=x)
        goldInt = 0.5
        TestInt = integrate_by_quadrature(function=function, x_lower=0, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
        TestInt = integrate_by_quadrature(function=function, x_lower=0, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_eval_basis_arbitrary(self):
        function = lambda x: basis.eval_basis(xmin=3, xmax=7, N_idx=0, x=x)
        goldInt = 2
        TestInt = integrate_by_quadrature(function=function, x_lower=3, x_upper=7, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
        TestInt = integrate_by_quadrature(function=function, x_lower=3, x_upper=7, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_eval_basisderiv_biunit(self):
        function = lambda x: basis.eval_basis_deriv(xmin=-1, xmax=1, N_idx=0, x=x)
        goldInt = -1
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
        function = lambda x: basis.eval_basis_deriv(xmin=-1, xmax=1, N_idx=1, x=x)
        goldInt = 1
        TestInt = integrate_by_quadrature(function=function, x_lower=-1, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_eval_basisderiv_unit(self):
        function = lambda x: basis.eval_basis_deriv(xmin=0, xmax=1, N_idx=0, x=x)
        goldInt = -0.5
        TestInt = integrate_by_quadrature(function=function, x_lower=0, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
        function = lambda x: basis.eval_basis_deriv(xmin=0, xmax=1, N_idx=1, x=x)
        goldInt = 0.5
        TestInt = integrate_by_quadrature(function=function, x_lower=0, x_upper=1, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
    def test_eval_basisderiv_arbitrary(self):
        function = lambda x: basis.eval_basis_deriv(xmin=3, xmax=7, N_idx=0, x=x)
        goldInt = -2
        TestInt = integrate_by_quadrature(function=function, x_lower=3, x_upper=7, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)
        function = lambda x: basis.eval_basis_deriv(xmin=3, xmax=7, N_idx=1, x=x)
        goldInt = 2
        TestInt = integrate_by_quadrature(function=function, x_lower=3, x_upper=7, n_quad=1)
        self.assertAlmostEqual(goldInt, TestInt)

class Test_get_jacobian(unittest.TestCase):
    def test_biunit_to_biunit(self):
        goldJacobian = 1
        TestJacobian = get_jacobian(X0=-1, X1=1)
        self.assertAlmostEqual(goldJacobian, goldJacobian)
    def test_biunit_to_unit(self):
        goldJacobian = 0.5
        TestJacobian = get_jacobian(X0=0, X1=1)
        self.assertAlmostEqual(goldJacobian, goldJacobian)
    def test_biunit_to_nontrivial(self):
        goldJacobian = 2
        TestJacobian = get_jacobian(X0=3, X1=7)
        self.assertAlmostEqual(goldJacobian, goldJacobian)

