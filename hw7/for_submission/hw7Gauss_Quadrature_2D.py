import math
import numpy as np
import sys
import unittest
import numpy.polynomial.legendre as lg
import inspect
# import Basis_Functions as bf


def get_gauss_quadrature(n_quad):
    pts = lg.leggauss(n_quad)[0]
    wts = lg.leggauss(n_quad)[1]
    return pts, wts

def get_jacobian(orig_domain_0, orig_domain_1, targ_domain_0=-1, targ_domain_1=1):
    jacob_numer = (orig_domain_1 - orig_domain_0)
    jacob_denom = (targ_domain_1 - targ_domain_0)
    jacobian = jacob_numer / jacob_denom
    return jacobian

class Gauss_Quadrature1d:
    def __init__(self,n_quad, quad_domain_0=-1, quad_domain_1=1):
        self.n_quad = n_quad
        self.jacobian = 1.0
        quad = get_gauss_quadrature(self.n_quad)
        self.quad_pts = quad[0]
        self.quad_wts = quad[1]
        self.quad_domain_0 = quad_domain_0
        self.quad_domain_1 = quad_domain_1
        self.start = self.quad_domain_0
        self.end = self.quad_domain_1
        self.new_pts = []
        if self.quad_domain_0 != -1 or self.quad_domain_1 != 1:
            self.quad_interval_change()

    def quad_interval_change(self):
        for pt in self.quad_pts:
            self.new_pts.append((self.quad_domain_1 - self.quad_domain_0)/2 * pt + (self.quad_domain_0 + self.quad_domain_1)/2)
        self.quad_pts = self.new_pts
        self.int_chg_jac = get_jacobian(orig_domain_0=-1, orig_domain_1=1, targ_domain_0=self.quad_domain_0, targ_domain_1=self.quad_domain_1)
        self.inverse_int_chg_jac = 1 / self.int_chg_jac
        self.jacobian *= self.inverse_int_chg_jac

    def integrate_by_quadrature(self, function, orig_domain_0, orig_domain_1):
        self.function = function
        self.orig_domain_0 = orig_domain_0
        self.orig_domain_1 = orig_domain_1
        self.jac_initial = get_jacobian(orig_domain_0, orig_domain_1, self.quad_domain_0, self.quad_domain_1)
        self.modified_jac = self.jac_initial * self.jacobian
        std_quad = get_gauss_quadrature(self.n_quad)
        self.std_quad_pts = std_quad[0]
        integral = 0
        q_integral = 0
        std_integral = 0
        for p in range(0, len(self.quad_pts)):
            wt = self.quad_wts[p]
            pt = self.quad_pts[p]
            std_pt = self.std_quad_pts[p]
            std_pt_equiv = x_of_ξ(std_pt,  self.orig_domain_0,  self.orig_domain_1, -1, 1)
            function_term = function(std_pt_equiv)
            unscaled_contrib = function_term * wt
            scaled_contrib = unscaled_contrib * self.modified_jac
            integral += scaled_contrib
        return integral

# I'm not sure if these are exactly the quadrature changes that were intended -- I'm having this take in n_quads for xi and eta, whereupon: 
    # calling the get_point method with i, j arguments will return the index=i point from the xi quadrature and the index=j point from the eta quadrature
    # calling the get_weight method with i, j will return the product of the index=i weight from the xi quadrature and the index=j weight from the eta quadrature
class Gauss_Quadrature2d:
    def __init__(self,n_quad_xi, n_quad_eta, quad_domain_xi0=-1, quad_domain_xi1=1, quad_domain_eta0=-1, quad_domain_eta1=1):
        self.n_quad_xi = n_quad_xi
        self.n_quad_eta = n_quad_eta
        
        self.jacobian = 1.0
                
        self.quad_domain_xi0 = quad_domain_xi0
        self.quad_domain_xi1 = quad_domain_xi1
        
        self.quad_domain_eta0 = quad_domain_eta0
        self.quad_domain_eta1 = quad_domain_eta1

        self.start_xi = self.quad_domain_xi0
        self.end_xi = self.quad_domain_xi1

        self.start_eta = self.quad_domain_eta0
        self.end_eta = self.quad_domain_eta1

        quad_xi = get_gauss_quadrature(self.n_quad_xi)
        quad_eta = get_gauss_quadrature(self.n_quad_eta)
        
        # weights        
        self.quad_wts_xi = quad_xi[1]
        self.quad_wts_eta = quad_eta[1]
        self.quad_wts = np.array((self.quad_wts_xi, self.quad_wts_eta))

        # points
        self.quad_pts_xi = quad_xi[0]
        self.quad_pts_eta = quad_eta[0]
        self.quad_pts = np.array((self.quad_pts_xi, self.quad_pts_eta))

    def get_weight(self, i_xi, j_eta):
        wt_i_xi = self.quad_wts[0][i_xi] 
        wt_j_eta = self.quad_wts[1][j_eta]
        wt_ = wt_i_xi * wt_j_eta
        return wt_
    
    def get_point(self, i_xi, j_eta):
        pt_i_xi = self.quad_pts[0][i_xi] 
        pt_j_eta = self.quad_pts[1][j_eta]
        pt_ = np.array((pt_i_xi, pt_j_eta))
        return pt_