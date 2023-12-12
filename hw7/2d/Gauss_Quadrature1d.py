import math
import numpy as np
import sys
import unittest
import numpy.polynomial.legendre as lg
import inspect


def x_of_ξ(ξ, x0, x1, ξ0, ξ1):
    x = ((ξ - ξ0)*(x1 - x0) / (ξ1 - ξ0)) + x0
    return x
def ξ_of_x(x, x0, x1, ξ0, ξ1):
    ξ = ((x - x0)*(ξ1 - ξ0) / (x1 - x0)) + ξ0
    return ξ

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
        # return integral
        self.integral = integral




# for -1, 1 to 3 7, x(ξ=0)=5
# x_of_ξ(0,3,7,-1,1)
# 5.0
# for 3, 7 to -1 1, ξ(x=5)=0
# ξ_of_x(5,3,7,-1,1)
# 0.0
# for -1, 1 to 0,1, ξ(x=0)=0.5
# ξ_of_x(0,-1,1,0,1)
# 0.5
# for -1, 1 to 0,1, ξ(x=0)=0.5
# ξ_of_x(0,-1,1,0,1)
# 0.5
# ξ_of_x(0.5,0,1,3,7) 
# # 5.0


# for 3 7 to -1 1, j=2.0
# get_jacobian_(3,7,-1,1)
# 2.0
# for -1 1 to 0 1, j=2.0
# get_jacobian_(-1,1,0,1)
# 2.0 

# for 0 1 to -1 1, j= 0.5
# get_jacobian_(0,1,-1,1)
# 0.5 
# for -1 1 to 3 7, j= 0.5
# get_jacobian_(-1,1,3,7)
# 0.5