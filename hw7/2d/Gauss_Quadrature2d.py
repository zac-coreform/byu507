import math
import numpy as np
import sys
import unittest
import numpy.polynomial.legendre as lg
import inspect
import Basis_Functions as bf


def get_gauss_quadrature(n_quad):
    pts = lg.leggauss(n_quad)[0]
    wts = lg.leggauss(n_quad)[1]
    return pts, wts

def get_jacobian(orig_domain_0, orig_domain_1, targ_domain_0=-1, targ_domain_1=1):
    jacob_numer = (orig_domain_1 - orig_domain_0)
    jacob_denom = (targ_domain_1 - targ_domain_0)
    jacobian = jacob_numer / jacob_denom
    return jacobian

class Gauss_Quadrature2d:
    def __init__(self,n_quad, quad_dom_xi_0=-1, quad_dom_xi_1=1, quad_dom_eta_0=-1, quad_dom_eta_1=1):
        self.n_quad = n_quad
        self.quad_dom_xi_0 = quad_dom_xi_0
        self.quad_dom_xi_1 = quad_dom_xi_1
        self.quad_dom_eta_0 = quad_dom_eta_0
        self.quad_dom_eta_1 = quad_dom_eta_1
        self.jacobian_xi = 1.0
        self.jacobian_eta = 1.0

        quad_xi = get_gauss_quadrature(self.n_quad)
        quad_eta = get_gauss_quadrature(self.n_quad)

        self.quad_pts_xi = quad_xi[0]
        self.quad_wts_xi = quad_xi[1]
        self.quad_pts_eta = quad_eta[0]
        self.quad_wts_eta = quad_eta[1]
        self.new_pts_xi = []
        self.new_pts_eta = []
        if self.quad_dom_xi_0 != -1 or self.quad_dom_xi_1 != 1:
            self.quad_interval_change_xi()
        if self.quad_dom_eta_0 != -1 or self.quad_dom_eta_1 != 1:
            self.quad_interval_change_eta()

        # zipping xi, eta to form 2d point coordinates
        self.pts_2d = list(map (list, zip(self.quad_pts_xi, self.quad_pts_eta)))
        # mapping xi, eta to form 2d weight products
        self.wts_2d = list(map(lambda x,y: x * y, self.quad_wts_xi, self.quad_wts_eta))

    def quad_interval_change_xi(self):
        for pt in self.quad_pts_xi:
            self.new_pts_xi.append((self.quad_dom_xi_1 - self.quad_dom_xi_0)/2 * pt + (self.quad_dom_xi_0 + self.quad_dom_xi_1)/2)
        self.quad_pts_xi = self.new_pts_xi
        self.int_chg_jac_xi = get_jacobian(orig_domain_0=-1, orig_domain_1=1, targ_domain_0=self.quad_dom_xi_0, targ_domain_1=self.quad_dom_xi_1)
        self.inverse_int_chg_jac_xi = 1 / self.int_chg_jac_xi
        self.jacobian_xi *= self.inverse_int_chg_jac_xi

    def quad_interval_change_eta(self):
        for pt in self.quad_pts_eta:
            self.new_pts_eta.append((self.quad_dom_eta_1 - self.quad_dom_eta_0)/2 * pt + (self.quad_dom_eta_0 + self.quad_dom_eta_1)/2)
        self.quad_pts_eta = self.new_pts_eta
        self.int_chg_jac_eta = get_jacobian(orig_domain_0=-1, orig_domain_1=1, targ_domain_0=self.quad_dom_eta_0, targ_domain_1=self.quad_dom_eta_1)
        self.inverse_int_chg_jac_eta = 1 / self.int_chg_jac_eta
        self.jacobian_eta *= self.inverse_int_chg_jac_eta

    # def integrate_by_quadrature2d(self, function, orig_dom_xi_0, orig_dom_xi_1, orig_dom_eta_0, orig_dom_eta_1):
    #     self.function = function
    #     self.orig_dom_xi_0 = orig_dom_xi_0
    #     self.orig_dom_xi_1 = orig_dom_xi_1
    #     self.orig_dom_eta_0 = orig_dom_eta_0
    #     self.orig_dom_eta_1 = orig_dom_eta_1
    #     self.jac_initial_xi = get_jacobian(orig_dom_xi_0, orig_dom_xi_1, self.quad_dom_xi_0, self.quad_dom_xi_1)
    #     self.jac_initial_eta = get_jacobian(orig_dom_eta_0, orig_dom_eta_1, self.quad_dom_eta_0, self.quad_dom_eta_1)
    #     self.modified_jac_xi = self.jac_initial_xi * self.jacobian
    #     self.modified_jac_eta = self.jac_initial_eta * self.jacobian
        
    #     # std_quad = get_gauss_quadrature(self.n_quad)
    #     # self.std_quad_pts = std_quad[0]
    #     # integral = 0
        
    #     xi_int = 0
    #     eta_int = 0
        
    #     for p in range(0, len(self.quad_pts_xi)):
    #         x_loc = bf.XMap(self.quad_pts_xi[p])
    #         fn_term = function()
    #         for q in range(0, len(self.quad_pts_eta)):
    #             wt = self.quad_pts_xi[p] * self.quad_pts_eta[q]
    #             val = 


    #         function_term = function(std_pt_equiv)
    #         unscaled_contrib = function_term * wt
    #         scaled_contrib = unscaled_contrib * self.modified_jac
    #         integral += scaled_contrib
    #     self.integral = integral