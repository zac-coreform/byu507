import numpy as np
import sys
from matplotlib import pyplot as plt
import hw6Gauss_Quadrature as gq
import hw6Boundary_Conditions as bc
import hw6Basis_Functions as bf
import math
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


class generate_mesh():
    def __init__(self, phys_xmin, phys_xmax, n_elem):
        self.phys_xmin = phys_xmin
        self.phys_xmax = phys_xmax
        self.n_elem = n_elem
        self.gen_all_nodes()
        self.mesh_nodes = self.all_nodes
        self.gen_elem_node_coord_indices()
        self.enci = self.elem_node_coord_indices
    def gen_all_nodes(self):
        self.all_nodes = np.linspace(self.phys_xmin, self.phys_xmax, self.n_elem + 1)
        return self.all_nodes
    def gen_elem_node_coord_indices(self):
        arr = np.zeros((2, self.n_elem), dtype=int)
        for n in range(0, self.n_elem):
            arr[0,n] = n
            arr[1,n] = n+1
        self.elem_node_coord_indices = arr
        return self.elem_node_coord_indices
    def m_get_element_domain(self, e):
        n0_idx = int(self.enci[0,e])
        n1_idx = int(self.enci[1,e])
        self.element_domain = np.array((self.all_nodes[n0_idx], self.all_nodes[n1_idx]))
        return self.element_domain


def CreateIENArray(deg, n_elem):
    p = deg
    bfs_per_elem = p + 1
    # total_bfs = p * n_elem + 1
    IEN = np.zeros((bfs_per_elem, n_elem)).astype('int')
    for e in range(0, n_elem):
        for a in range(0, bfs_per_elem):
            Q = a + p * e
            IEN[a,e]=Q
    return IEN

# def CreateIDArray(bc_left,bc_right,n_basis_functions):
def CreateIDArray(bc_left, bc_right, deg, n_elem):
    p = deg
    n_basis_functions = p * n_elem + 1
    ID = np.zeros(n_basis_functions).astype('int')
    if bc_left.isDir:
        ID[0] = -1
        start_idx = 1
    elif bc_left.isNeu or bc_left.isRob:
        start_idx = 0
    else:
        sys.exit("Don't know this boundary condition")
        
    counter = 0
    for i in range(start_idx,n_basis_functions):
        ID[i] = counter
        counter += 1
    
    if bc_right.isDir:
        ID[-1] = -1
    elif bc_right.isNeu or bc_right.isRob:
        pass
    else:
        sys.exit("Don't know this boundary condition")

    return ID

def CreateXVals(phys_dom_0, phys_dom_1, ien):
    n_elem = ien.shape[1]
    xvals = np.linspace(phys_dom_0,phys_dom_1,n_elem+1)
    return xvals

# def LocalStiffness(e, xvals, n_elem_funcs, bc_left, bc_right, deg):
def LocalStiffness(e, xvals, bc_left, bc_right, deg):
    # the quadrature arg passed in previously here was of the form: 
        # quadrature = GaussianQuadrature.GaussQuadrature(n_quad)
        # returns: points, weights
    # changing to deg-determined quadrature
    p = deg
    n_elem_funcs = p + 1
    # degree 2n − 1 polynomials, where n is number of sample points used // n = (p+1)/2
    n_quad = math.ceil((p + 1) / 2)
    quadrature = gq.Gauss_Quadrature(n_quad=n_quad,quad_domain_0=0, quad_domain_1=1)
    #. ^ with quad_domain_N args, returns pts mapped to [0,1]
    #. the `Gauss_Quadrature.integrate_by_quadrature` method has the jacobians and 0.5 multiplier built in, but I haven't yet implemented it below. 

    x0 = xvals[e]
    x1 = xvals[e+1]
    
    quad_wts = quadrature.quad_wts
    quad_pts = quadrature.quad_pts
    n_quad = quadrature.n_quad
    
    ke = np.zeros((n_elem_funcs,n_elem_funcs))

    for g in range(0,n_quad):
        w_g = quad_wts[g]
        xi_g = quad_pts[g]
        x_g = bf.XMap(x0,x1,xi_g,p)
        x_derv = bf.XMapDerv(x0,x1,xi_g,p)
        x_derv_inv = 1.0/x_derv
        
        for a in range(0,n_elem_funcs):
            Na_x = bf.NBasisDerv(deg=p, N_idx=a, t=xi_g)
            for b in range(a,n_elem_funcs):
                Nb_x = bf.NBasisDerv(deg=p, N_idx=b, t=xi_g)
                ke[a,b] += w_g * Na_x * Nb_x * x_derv_inv * 0.5
                #. ^ quad sums must be multiplied by 0.5

    # enforce symmetry
    for a in range(0,n_elem_funcs):
        for b in range(a,n_elem_funcs):
            ke[b,a] = ke[a,b]

    # address boundary conditions
    if e == 0:
        if bc_left.isDir:
            pass
        elif bc_left.isNeu:
            pass
        elif bc_left.isRob:
            coeff = bc_left.b1/bc_left.b2
            ke[0,0] -= coeff * bf.NBasis(0,-1,1,-1)**2
    
    if e == len(xvals) - 2:
        last_func = n_elem_funcs-1
        if bc_right.isDir:
            pass
        elif bc_right.isNeu:
            pass
        elif bc_right.isRob:
            coeff = bc_right.b1/bc_right.b2
            ke[last_func,last_func]  += coeff * bf.NBasis(last_func,-1,1,1)**2
    
    return ke

def LocalForce(e,xvals,f,bc_left,bc_right,deg):
    # as above in ke, changing to deg-determined quadrature
    p = deg
    n_elem_funcs = p + 1
    n_quad = math.ceil((p + 1) / 2)
    quadrature = gq.Gauss_Quadrature(n_quad=n_quad,quad_domain_0=0, quad_domain_1=1)

    x0 = xvals[e]
    x1 = xvals[e+1]
    
    quad_wts = quadrature.quad_wts
    quad_pts = quadrature.quad_pts
    n_quad = quadrature.n_quad
    
    fe = np.zeros((n_elem_funcs,1))
    
    for g in range(0,n_quad):
        w_g = quad_wts[g]
        xi_g = quad_pts[g]
        x_g = bf.XMap(x0,x1,xi_g,p)
        x_derv = bf.XMapDerv(x0,x1,xi_g,p)
        
        for a in range(0,n_elem_funcs):
            fe[a] += w_g * bf.NBasis(p, a, xi_g) * f(x_g) * x_derv * 0.5
            #. ^ quad sums must be multiplied by 0.5
    
    if e == 0:
        if bc_left.isDir:
            g_val = bc_left.b3 / bc_left.b1
            for g in range(0,n_quad):
                w_g = quad_wts[g]
                xi_g = quad_pts[g]
                x_g = bf.XMap(x0,x1,xi_g,p)
                x_derv = bf.XMapDerv(x0,x1,xi_g,p)
                
                for a in range(0,n_elem_funcs):
                    fe[a] -= w_g * bf.NBasisDerv(p,a,xi_g) * \
                                   bf.NBasisDerv(p,0,xi_g) * \
                                   x_derv**(-1) * g_val * 0.5
                    #. ^ quad sums must be multiplied by 0.5

        elif bc_left.isNeu:
            h_val = bc_left.b3 / bc_left.b2
            fe[0] -= bf.NBasis(p,0,bf.XMap(x0,x1,-1,p)) * h_val
        elif bc_left.isRob:
            mult_val = bc_left.b3 / bc_left.b2
            fe[0] -= bf.NBasis(p,0,bf.XMap(x0,x1,-1,p)) * mult_val
            if bc_right.isDir:
                # This may not be precise in very weird circumstances
                pass

    if e == len(xvals)-2:
        last_func = n_elem_funcs-1
        if bc_right.isDir:
            g_val = bc_right.b3 / bc_right.b1
            for g in range(0,n_quad):
                w_g = quad_wts[g]
                xi_g = quad_pts[g]
                x_g = bf.XMap(x0,x1,xi_g,p)
                x_derv = bf.XMapDerv(x0,x1,xi_g,p)
                
                for a in range(0,n_elem_funcs):
                    fe[a] -= w_g * bf.NBasisDerv(p,a,xi_g) * \
                                   bf.NBasisDerv(p,last_func,xi_g) * \
                                   x_derv**(-1) * g_val * 0.5
                    #. ^ quad sums must be multiplied by 0.5

        elif bc_right.isNeu:
            h_val = bc_right.b3 / bc_right.b2
            fe[last_func] += bf.NBasis(p,last_func,bf.XMap(x0,x1,1,p)) * h_val
        elif bc_right.isRob:
            mult_val = bc_right.b3 / bc_right.b2
            fe[last_func] += bf.NBasis(p,last_func,bf.XMap(x0,x1,1,p)) * mult_val
            if bc_right.isDir:
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
    if bc_left.isDir:
        D = np.concatenate([[[bc_left.b3/bc_left.b1]],D], 0)
    if bc_right.isDir:
        D = np.concatenate([D,[[bc_right.b3 / bc_right.b1]]], 0)
    
    plt.plot(xvals,D)
    if title != "":
        plt.title(title)
    
    
    
    # make this take multiple lists of functions with different visual params?
    
def baseplot(fns, res=100, xlim=[0,1], ylim=[0,1], equal_aspect_ratio=False):
    if isinstance(fns[0], list):
        mainfns = fns[0]
        auxfns = fns[1]
    elif isinstance(fns[0], function):
        mainfns = fns
        auxfns = False

    fig, ax = plt.subplots()
    xdom = np.linspace(xlim[0], xlim[1], res)

    ax.set_prop_cycle('color', ["red", "green","blue"])
    for item in mainfns:
        leg_str = item.__name__
        ax.plot(xdom, item(xdom), label=leg_str)
    if auxfns:
        # ax.set_prop_cycle('color', ["magenta", "mediumseagreen","teal"])
        ax.set_prop_cycle('color', ["red", "green","blue"])
        for item in auxfns: 
            leg_str = item.__name__
            ax.plot(xdom, item(xdom), label=leg_str, linestyle=':', linewidth=1.25)
    if equal_aspect_ratio == True:
        ax.set_aspect('equal')
    
    # set axis range limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # set grid / axis ticks
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(True, which='both', alpha=0.5)
    # add zero line
    ax.axhline(y=0, color='k')
    # ax.axvline(x=0.5, color='m', linestyle=':', linewidth=1)
    # ax.axvline(x=0.25, color='m', linestyle=':', linewidth=1)
    # ax.axvline(x=0.75, color='m', linestyle=':', linewidth=1)
    # ax.axhline(y=0.5, color='m', linestyle=':', linewidth=1)
    # ax.axhline(y=0.25, color='m', linestyle=':', linewidth=1)
    # ax.axhline(y=0.75, color='m', linestyle=':', linewidth=1)
    ax.legend()
    # plt.xlim(xlim)
    # plt.ylim(ylim)

    
    
    
    
    
    
    
    
    
