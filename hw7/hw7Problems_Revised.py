
import numpy as np
from matplotlib import pyplot as plt

import hw7Boundary_Conditions as bc
import hw7Poisson_1D as p1
import hw7Gauss_Quadrature as gq
import hw7Basis_Functions as bf

# bc_left.InitializeData(bc.BCType.Neumann, 
#                         bc.BCOrientation.Left,
#                         h, rhs / b3
#                         0, b1
#                         -1) b2
# bc_right.InitializeData(bc.BCType.Dirichlet, 
#                         bc.BCOrientation.Right,
#                         g, 
#                         1, 
#                         0)
# def InitializeData(self, bdry_type, bdry_orientation, Rhs, sol_mult, sol_derv_mult):

# Rhs,  b3
# sol_mult,  b1
# sol_derv_mult b2

def Testing_Problem():
    L = 1
    n_elem = 3
    fn = lambda x: x
    g = 1
    h = 0.2
    bc_left = bc.BoundaryCondition("left", "Neu", 0, -1, h)
    bc_right = bc.BoundaryCondition("right", "Dir", 1, 0, g)
    n_quad = 5
    quadrature = gq.Gauss_Quadrature1d(n_quad, 0, 1)
    xvals = np.linspace(0,L,n_elem+1)
    
    # def FEM_Poisson(bc_left,bc_right,f,xvals,deg):
    D = p1.FEM_Poisson(bc_left=bc_left, bc_right=bc_right, f=fn, xvals=xvals, deg=1)
    
    p1.PlotSolution(bc_left, bc_right, xvals, D, "Test Problem")

    xtrue = np.linspace(0,L,101)
    ytrue = g + (1-xtrue)*h+1/6-xtrue**3/6
    plt.plot(xtrue,ytrue)

def Problem_1():
    u_infty_left = 303.15
    u_infty_right = 293.15
    kappa = 2.711
    omega = 2.2
    L = 0.5
    n_elem = 1
    f = lambda x: 0 / kappa
    bc_left = bc.BoundaryCondition("left", "Rob", omega, -kappa, omega*u_infty_left)
    bc_right = bc.BoundaryCondition("right", "Rob", omega, kappa, omega*u_infty_right)
    n_quad = 5
    quadrature = gq.Gauss_Quadrature1d(n_quad, 0, 1)
    xvals = np.linspace(0,L,n_elem+1)
    
    D = p1.FEM_Poisson(bc_left, bc_right, f, xvals, 1)
    
    p1.PlotSolution(bc_left, bc_right, xvals, D, "Problem 1")

def Problem_2():
    # bc_top = bc.BoundaryCondition(0)
    # bc_bottom = bc.BoundaryCondition(1)
    u_top = -0.5
    u_bottom = -1
    E = 29000 # ksi
    r = 2 # inches
    A = np.pi*r**2
    L = 120
    n_elem = 20
    f = lambda x: 0 / (E*A)
    bc_top = bc.BoundaryCondition("left", "Dir", 1, 0, u_top)
    bc_bottom = bc.BoundaryCondition("right", "Dir", 1, 0, u_bottom)
    
    n_quad = 5
    quadrature = gq.Gauss_Quadrature1d(n_quad)
    
    xvals = np.linspace(0,L,n_elem+1)
    
    D = p1.FEM_Poisson(bc_top, bc_bottom, f, xvals, 1)
    
    p1.PlotSolution(bc_top, bc_bottom, xvals, D, "Problem 2")

def Problem_3():
    L = 5
    n_elem = 4
    u_left = -0.02841
    alpha = 1.5 * 10**-2
    f = lambda x: 3.16 * 10**-6 / alpha
    h = 1.75 * 10**-5
    bc_left = bc.BoundaryCondition("left", "Dir", 1, 0, u_left)
    bc_right = bc.BoundaryCondition("right", "Neu", 0, alpha, h)

    n_quad = 5
    quadrature = gq.Gauss_Quadrature1d(n_quad)
    
    xvals = np.linspace(0,L,n_elem+1)
    
    D = p1.FEM_Poisson(bc_left, bc_right, f, xvals, 1)
    
    p1.PlotSolution(bc_left, bc_right, xvals, D, "Problem 3")

    
def Problem_4():

    L = 4
    n_elem = 20
    flux_left = 1
    rho = lambda x: 12**-10*x
    epsilon = 8.854 * 10**-12
    f = lambda x: rho(x)/epsilon
    delta = 1
    bc_left = bc.BoundaryCondition("left", "Neu", 0, -1, flux_left)
    bc_right = bc.BoundaryCondition("right", "Rob", 1, delta, 0)
    
    n_quad = 5
    quadrature = gq.Gauss_Quadrature1d(n_quad)
    
    xvals = np.linspace(0,L,n_elem+1)
    
    D = p1.FEM_Poisson(bc_left, bc_right, f, xvals, 1)
    
    p1.PlotSolution(bc_left, bc_right, xvals, D,"Problem 4")

    
