import math
import numpy as np
import sys
import unittest
import numpy.polynomial.legendre as lg
import inspect


class Gauss_Quadrature2d:
    def __init__(self,n_quad, quad_dom_xi_0=-1, quad_dom_xi_1=1, quad_dom_eta_0=-1, quad_dom_eta_1=1):
        self.n_quad = n_quad
        self.jacobian_xi = 1.0
        self.jacobian_eta = 1.0
        quad_xi = get_gauss_quadrature(self.n_quad)
        quad_eta = get_gauss_quadrature(self.n_quad)
        self.quad_pts_xi = quad_xi[0]
        self.quad_wts_xi = quad_xi[1]
        self.quad_pts_eta = quad_eta[0]
        self.quad_wts_eta = quad_eta[1]
        self.quad_dom_xi_0 = quad_dom_xi_0
        self.quad_dom_xi_1 = quad_dom_xi_1
        self.quad_dom_eta_0 = quad_dom_eta_0
        self.quad_dom_eta_1 = quad_dom_eta_1
        self.new_pts_xi = []
        self.new_pts_eta = []
        if self.quad_dom_xi_0 != -1 or self.quad_dom_xi_1 != 1:
            self.quad_interval_change_xi()
        if self.quad_dom_eta_0 != -1 or self.quad_dom_eta_1 != 1:
            self.quad_interval_change_eta()

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
