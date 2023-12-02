#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:03:40 2023

@author: kendrickshepherd
"""

import numpy as np
import sys
from matplotlib import pyplot as plt

import GaussianQuadrature as gq
import Boundary_Conditions as bc

import Basis_Functions as bf


def CreateIENArray(n_basis_function_on_elem,n_elem):
    IEN = np.zeros((n_basis_function_on_elem,n_elem)).astype('int')
    for e in range(0,n_elem):
        for a in range(0,n_basis_function_on_elem):
            Q = a+e
            IEN[a,e]=Q
    return IEN

def CreateIDArray(bc_left,bc_right,n_basis_functions):
    ID = np.zeros(n_basis_functions).astype('int')
    
    if bc_left.IsDirichlet():
        ID[0] = -1
        start_idx = 1
    elif bc_left.IsNeumann() or bc_left.IsRobin():
        start_idx = 0
    else:
        sys.exit("Don't know this boundary condition")
        
    counter = 0
    for i in range(start_idx,n_basis_functions):
        ID[i] = counter
        counter += 1
    
    if bc_right.IsDirichlet():
        ID[-1] = -1
    elif bc_right.IsNeumann() or bc_right.IsRobin():
        pass
    else:
        sys.exit("Don't know this boundary condition")

    return ID

def LocalStiffness(e,xvals,n_elem_funcs,bc_left,bc_right,quadrature):
    x0 = xvals[e]
    x1 = xvals[e+1]
    
    quad_wts = quadrature.quad_wts
    quad_pts = quadrature.quad_pts
    n_quad = quadrature.n_quad
    
    ke = np.zeros((n_elem_funcs,n_elem_funcs))

    for g in range(0,n_quad):
        w_g = quad_wts[g]
        ksi_g = quad_pts[g]
        x_g = bf.XMap(x0,x1,ksi_g)
        x_derv = bf.XMapDerv(x0,x1,ksi_g)
        x_derv_inv = 1.0/x_derv
        
        for a in range(0,n_elem_funcs):
            Na_x = bf.NBasisDerv(a,-1,1,ksi_g)
            for b in range(a,n_elem_funcs):
                Nb_x = bf.NBasisDerv(b,-1,1,ksi_g)
                ke[a,b] += w_g * Na_x * Nb_x * x_derv_inv

    # enforce symmetry
    for a in range(0,n_elem_funcs):
        for b in range(a,n_elem_funcs):
            ke[b,a] = ke[a,b]

    # address boundary conditions
    if e == 0:
        if bc_left.IsDirichlet():
            pass
        elif bc_left.IsNeumann():
            pass
        elif bc_left.IsRobin():
            coeff = bc_left.sol_multiplier/bc_left.sol_derv_multiplier
            ke[0,0] -= coeff * bf.NBasis(0,-1,1,-1)**2
    
    if e == len(xvals) - 2:
        last_func = n_elem_funcs-1
        if bc_right.IsDirichlet():
            pass
        elif bc_right.IsNeumann():
            pass
        elif bc_right.IsRobin():
            coeff = bc_right.sol_multiplier/bc_right.sol_derv_multiplier
            ke[last_func,last_func]  += coeff * bf.NBasis(last_func,-1,1,1)**2
    
    return ke

def LocalForce(e,xvals,n_elem_funcs,f,bc_left,bc_right,quadrature):
    x0 = xvals[e]
    x1 = xvals[e+1]
    
    quad_wts = quadrature.quad_wts
    quad_pts = quadrature.quad_pts
    n_quad = quadrature.n_quad
    
    fe = np.zeros((n_elem_funcs,1))
    
    for g in range(0,n_quad):
        w_g = quad_wts[g]
        ksi_g = quad_pts[g]
        x_g = bf.XMap(x0,x1,ksi_g)
        x_derv = bf.XMapDerv(x0,x1,ksi_g)
        
        for a in range(0,n_elem_funcs):
            fe[a] += w_g * bf.NBasis(a,-1,1,ksi_g) * f(x_g) * x_derv
            
    
    if e == 0:
        if bc_left.IsDirichlet():
            g_val = bc_left.rhs / bc_left.sol_multiplier
            for g in range(0,n_quad):
                w_g = quad_wts[g]
                ksi_g = quad_pts[g]
                x_g = bf.XMap(x0,x1,ksi_g)
                x_derv = bf.XMapDerv(x0,x1,ksi_g)
                
                for a in range(0,n_elem_funcs):
                    fe[a] -= w_g * bf.NBasisDerv(a,-1,1,ksi_g) * \
                                   bf.NBasisDerv(0,-1,1,ksi_g) * \
                                   x_derv**(-1) * g_val
        elif bc_left.IsNeumann():
            h_val = bc_left.rhs / bc_left.sol_derv_multiplier
            fe[0] -= bf.NBasis(0,x0,x1,bf.XMap(x0,x1,-1)) * h_val
        elif bc_left.IsRobin():
            mult_val = bc_left.rhs / bc_left.sol_derv_multiplier
            fe[0] -= bf.NBasis(0,x0,x1,bf.XMap(x0,x1,-1)) * mult_val
            if bc_right.IsDirichlet():
                # This may not be precise in very weird circumstances
                pass

    if e == len(xvals)-2:
        last_func = n_elem_funcs-1
        if bc_right.IsDirichlet():
            g_val = bc_right.rhs / bc_right.sol_multiplier
            for g in range(0,n_quad):
                w_g = quad_wts[g]
                ksi_g = quad_pts[g]
                x_g = bf.XMap(x0,x1,ksi_g)
                x_derv = bf.XMapDerv(x0,x1,ksi_g)
                
                for a in range(0,n_elem_funcs):
                    fe[a] -= w_g * bf.NBasisDerv(a,-1,1,ksi_g) * \
                                   bf.NBasisDerv(last_func,-1,1,ksi_g) * \
                                   x_derv**(-1) * g_val
        elif bc_right.IsNeumann():
            h_val = bc_right.rhs / bc_right.sol_derv_multiplier
            fe[last_func] += bf.NBasis(last_func,x0,x1,bf.XMap(x0,x1,1)) * h_val
        elif bc_right.IsRobin():
            mult_val = bc_right.rhs / bc_right.sol_derv_multiplier
            fe[last_func] += bf.NBasis(last_func,x0,x1,bf.XMap(x0,x1,1)) * mult_val
            if bc_right.IsDirichlet():
                # This may not be precise in very weird circumstances
                pass

    
    return fe

def FEM_Poisson(bc_left,bc_right,f,xvals,quadrature):
    n_functions = len(xvals)
    n_elems = len(xvals)-1
    degree = 1
    n_functions_per_element = degree + 1
    
    IEN = CreateIENArray(n_functions_per_element, n_elems)
    ID = CreateIDArray(bc_left, bc_right, n_functions)
    
    n_unknowns = int(max(ID)) + 1
    
    K = np.zeros((n_unknowns,n_unknowns))
    F = np.zeros((n_unknowns,1))
    
    for e in range(0,n_elems):
        
        ke = LocalStiffness(e,xvals,n_functions_per_element,bc_left,bc_right,quadrature)
        fe = LocalForce(e,xvals,n_functions_per_element,f,bc_left,bc_right,quadrature)
        
        for a in range(0,n_functions_per_element):
            P = IEN[a,e]
            A = ID[P]
            if A == -1:
                continue
            
            F[A] += fe[a]
            
            for b in range(0,n_functions_per_element):
                Q = IEN[b,e]
                B = ID[Q]
                if B == -1:
                    continue
                
                K[A,B] += ke[a,b]
    
    D = np.linalg.solve(K,F)
    
    return D
            
        
        
def PlotSolution(bc_left,bc_right,xvals,D,title=""):
    if bc_left.IsDirichlet():
        D = np.concatenate([[[bc_left.rhs/bc_left.sol_multiplier]],D], 0)
    if bc_right.IsDirichlet():
        D = np.concatenate([D,[[bc_right.rhs / bc_right.sol_multiplier]]], 0)
    
    plt.plot(xvals,D)
    if title != "":
        plt.title(title)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
