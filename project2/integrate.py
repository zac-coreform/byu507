import unittest
import numpy as np

import numpy.polynomial.legendre as lg

def get_gauss_quadrature(n_quad):
    pts = lg.leggauss(n_quad)[0]
    wts = lg.leggauss(n_quad)[1]
    return pts, wts

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