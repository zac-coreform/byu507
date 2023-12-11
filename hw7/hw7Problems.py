#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 08:29:03 2023

@author: kendrickshepherd
"""
import numpy as np
from matplotlib import pyplot as plt

import hw7Boundary_Conditions as bc
import hw7Poisson_1D
import hw7Gauss_Quadrature

def Testing_Problem():
    bc_left = bc.BoundaryCondition(0)
    bc_right = bc.BoundaryCondition(1)
    L = 1
    n_elem = 3
    f = lambda x: x
    g = 1
    h = 0.2
    bc_left.InitializeData(bc.BCType.Neumann, 
                           bc.BCOrientation.Left,
                           h, 
                           0, 
                           -1)
    bc_right.InitializeData(bc.BCType.Dirichlet, 
                           bc.BCOrientation.Right,
                           g, 
                           1, 
                           0)
    
    n_quad = 5
    quadrature = GaussianQuadrature.GaussQuadrature(n_quad)
    
    xvals = np.linspace(0,L,n_elem+1)
    
    D = Poisson_1D.FEM_Poisson(bc_left, bc_right, f, xvals, quadrature)
    
    Poisson_1D.PlotSolution(bc_left, bc_right, xvals, D, "Test Problem")

    xtrue = np.linspace(0,L,101)
    ytrue = g + (1-xtrue)*h+1/6-xtrue**3/6
    plt.plot(xtrue,ytrue)

def Problem_1():
    bc_left = bc.BoundaryCondition(0)
    bc_right = bc.BoundaryCondition(1)
    u_infty_left = 303.15
    u_infty_right = 293.15
    kappa = 2.711
    omega = 2.2
    L = 0.5
    n_elem = 1
    f = lambda x: 0 / kappa
    bc_left.InitializeData(bc.BCType.Robin, 
                           bc.BCOrientation.Left,
                           omega*u_infty_left, 
                           omega, 
                           -kappa)
    bc_right.InitializeData(bc.BCType.Robin, 
                           bc.BCOrientation.Right,
                           omega*u_infty_right, 
                           omega, 
                           kappa)
    
    n_quad = 5
    quadrature = GaussianQuadrature.GaussQuadrature(n_quad)
    
    xvals = np.linspace(0,L,n_elem+1)
    
    D = Poisson_1D.FEM_Poisson(bc_left, bc_right, f, xvals, quadrature)
    
    Poisson_1D.PlotSolution(bc_left, bc_right, xvals, D, "Problem 1")

def Problem_2():
    bc_top = bc.BoundaryCondition(0)
    bc_bottom = bc.BoundaryCondition(1)
    u_top = -0.5
    u_bottom = -1
    E = 29000 # ksi
    r = 2 # inches
    A = np.pi*r**2
    L = 120
    n_elem = 20
    f = lambda x: 0 / (E*A)
    bc_top.InitializeData(bc.BCType.Dirichlet, 
                           bc.BCOrientation.Left,
                           u_top, 
                           1, 
                           0)
    bc_bottom.InitializeData(bc.BCType.Dirichlet, 
                           bc.BCOrientation.Right,
                           u_bottom, 
                           1, 
                           0)
    
    n_quad = 5
    quadrature = GaussianQuadrature.GaussQuadrature(n_quad)
    
    xvals = np.linspace(0,L,n_elem+1)
    
    D = Poisson_1D.FEM_Poisson(bc_top, bc_bottom, f, xvals, quadrature)
    
    Poisson_1D.PlotSolution(bc_top, bc_bottom, xvals, D, "Problem 2")

def Problem_3():
    bc_left = bc.BoundaryCondition(0)
    bc_right = bc.BoundaryCondition(1)
    L = 5
    n_elem = 4
    u_left = -0.02841
    alpha = 1.5 * 10**-2
    f = lambda x: 3.16 * 10**-6 / alpha
    h = 1.75 * 10**-5
    
    bc_left.InitializeData(bc.BCType.Dirichlet, 
                           bc.BCOrientation.Left,
                           u_left, 
                           1, 
                           0)
    bc_right.InitializeData(bc.BCType.Neumann, 
                           bc.BCOrientation.Right,
                           h, 
                           0, 
                           alpha)

    n_quad = 5
    quadrature = GaussianQuadrature.GaussQuadrature(n_quad)
    
    xvals = np.linspace(0,L,n_elem+1)
    
    D = Poisson_1D.FEM_Poisson(bc_left, bc_right, f, xvals, quadrature)
    
    Poisson_1D.PlotSolution(bc_left, bc_right, xvals, D, "Problem 3")

    
def Problem_4():
    bc_left = bc.BoundaryCondition(0)
    bc_right = bc.BoundaryCondition(1)
    L = 4
    n_elem = 20
    flux_left = 1
    rho = lambda x: 12**-10*x
    epsilon = 8.854 * 10**-12
    f = lambda x: rho(x)/epsilon
    delta = 1
    
    bc_left.InitializeData(bc.BCType.Neumann, 
                           bc.BCOrientation.Left,
                           flux_left, 
                           0, 
                           -1)
    bc_right.InitializeData(bc.BCType.Robin, 
                           bc.BCOrientation.Right,
                           0, 
                           1, 
                           delta)
    
    n_quad = 5
    quadrature = GaussianQuadrature.GaussQuadrature(n_quad)
    
    xvals = np.linspace(0,L,n_elem+1)
    
    D = Poisson_1D.FEM_Poisson(bc_left, bc_right, f, xvals, quadrature)
    
    Poisson_1D.PlotSolution(bc_left, bc_right, xvals, D,"Problem 4")

    
Problem_3()