import numpy as np
import sys
from matplotlib import pyplot as plt
import hw7Gauss_Quadrature as gq
import hw7Boundary_Conditions as bc
import hw7Basis_Functions as bf
import math
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import hw7Error_Values as ev
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

def CreateIENArray(deg, n_elems):
    p = deg
    bfs_per_elem = p + 1
    IEN = np.zeros((bfs_per_elem, n_elems)).astype('int')
    for e in range(0, n_elems):
        for a in range(0, bfs_per_elem):
            Q = a + p * e
            IEN[a,e] = Q
    return IEN

def CreateIDArray(bc_left, bc_right, deg, n_elems):
    p = deg
    n_basis_functions = p * n_elems + 1
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

def CreateNodes(phys_dom_0, phys_dom_1, ien_or_n_elems):
    if isinstance(ien_or_n_elems, np.ndarray):
        n_elems = ien_or_n_elems.shape[1]
    elif isinstance(ien_or_n_elems, int):
        n_elems = ien_or_n_elems
    else:
        sys.exit("last arg must be ien array or n_elems int")
    nodes = np.linspace(phys_dom_0,phys_dom_1,n_elems+1)
    return nodes

CreateXVals = CreateNodes

def LocalStiffness(e, xvals, bc_left, bc_right, deg, quadrature_in):
    p = deg
    bfs_per_elem = p + 1
    x0 = xvals[e]
    x1 = xvals[e+1]
    quadrature = quadrature_in
    quad_wts = quadrature.quad_wts
    quad_pts = quadrature.quad_pts
    n_quad = quadrature.n_quad
    
    ke = np.zeros((bfs_per_elem,bfs_per_elem))

    for g in range(0,n_quad):
        w_g = quad_wts[g]
        xi_g = quad_pts[g]
        x_g = bf.XMap(x0,x1,xi_g,p)
        x_derv = bf.XMapDerv(x0,x1,xi_g,p)
        x_derv_inv = 1.0/x_derv
        
        for a in range(0,bfs_per_elem):
            Na_x = bf.NBasisDerv(deg=p, N_idx=a, t=xi_g)
            for b in range(a,bfs_per_elem):
                Nb_x = bf.NBasisDerv(deg=p, N_idx=b, t=xi_g)
                ke[a,b] += w_g * Na_x * Nb_x * x_derv_inv * 0.5
                #. ^ quad sums must be multiplied by 0.5

    # enforce symmetry
    for a in range(0,bfs_per_elem):
        for b in range(a,bfs_per_elem):
            ke[b,a] = ke[a,b]

    # address boundary conditions
    if e == 0:
        if bc_left.isDir:
            pass
        elif bc_left.isNeu:
            pass
        elif bc_left.isRob:
            coeff = bc_left.b1/bc_left.b2
            ke[0,0] -= coeff * bf.NBasis(deg=p, N_idx=0, t=0)**2
            
    if e == len(xvals) - 2:
        last_func = bfs_per_elem-1
        if bc_right.isDir:
            pass
        elif bc_right.isNeu:
            pass
        elif bc_right.isRob:
            coeff = bc_right.b1/bc_right.b2
            ke[last_func,last_func]  += coeff * bf.NBasis(deg=p, N_idx=last_func, t=1)**2
    
    return ke

def LocalForce(e,xvals,f,bc_left,bc_right,deg,quadrature_in):
    p = deg
    bfs_per_elem = p + 1
    x0 = xvals[e]
    x1 = xvals[e+1]
    quadrature = quadrature_in
    quad_wts = quadrature.quad_wts
    quad_pts = quadrature.quad_pts
    n_quad = quadrature.n_quad
    
    fe = np.zeros((bfs_per_elem,1))
    
    for g in range(0,n_quad):
        w_g = quad_wts[g]
        xi_g = quad_pts[g]
        x_g = bf.XMap(x0,x1,xi_g,p)
        x_derv = bf.XMapDerv(x0,x1,xi_g,p)
        
        for a in range(0,bfs_per_elem):
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
                
                for a in range(0,bfs_per_elem):
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
        last_func = bfs_per_elem-1
        if bc_right.isDir:
            g_val = bc_right.b3 / bc_right.b1
            for g in range(0,n_quad):
                w_g = quad_wts[g]
                xi_g = quad_pts[g]
                x_g = bf.XMap(x0,x1,xi_g,p)
                x_derv = bf.XMapDerv(x0,x1,xi_g,p)
                
                for a in range(0,bfs_per_elem):
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

class FEM_Poisson():
    def __init__(self, bc_left, bc_right, f, nodes, deg):
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.f = f
        self.nodes = nodes
        self.p = deg
        self.bfs_per_elem = self.p + 1
        self.quadrature = gq.Gauss_Quadrature1d(n_quad=7,quad_domain_0=0, quad_domain_1=1)
        self.n_elems = len(nodes)-1
        self.IEN = CreateIENArray(self.p, self.n_elems)
        self.ID = CreateIDArray(self.bc_left, self.bc_right, self.p, self.n_elems)
        self.n_unknowns = int(max(self.ID)) + 1
        self.plot_domain = np.linspace(0,1,self.n_unknowns)
        self.K = np.zeros((self.n_unknowns,self.n_unknowns))
        self.F = np.zeros((self.n_unknowns,1))
        self.ke_list = []
        self.fe_list = []
        for e in range(0,self.n_elems):
            self.ke = LocalStiffness(e,self.nodes,self.bc_left,self.bc_right,self.p, self.quadrature)
            self.ke_list.append(self.ke)
            self.fe = LocalForce(e,self.nodes,self.f,self.bc_left,self.bc_right,self.p, self.quadrature)
            self.fe_list.append(self.fe)
            for a in range(0,self.bfs_per_elem):
                P = self.IEN[a,e]
                A = self.ID[P]
                if A == -1:
                    continue
                self.F[A] += self.fe[a]
                for b in range(0,self.bfs_per_elem):
                    Q = self.IEN[b,e]
                    B = self.ID[Q]
                    if B == -1:
                        continue
                    self.K[A,B] += self.ke[a,b]
        self.D = np.linalg.solve(self.K,self.F)

class PlotSolution():
    def __init__(self, bc_left, bc_right, xvals, D_in, title=""):
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.xvals = xvals
        self.D_in = D_in
        self.title = title
        if self.bc_left.isDir:
            self.D_out = np.concatenate([[[self.bc_left.b3/self.bc_left.b1]],self.D_in], 0)
        elif self.bc_right.isDir:
            self.D_out = np.concatenate([self.D_in,[[self.bc_right.b3 / self.bc_right.b1]]], 0)
        else:
            self.D_out = self.D_in
    def make_plot(self):
        plt.plot(self.xvals,self.D_out)
        if self.title != "":
            plt.title(self.title) 

class GetFullD():
    def __init__(self, bcl, bcr, D_in):
        self.bc_left = bcl
        self.bc_right = bcr
        # self.xvals = xvals
        self.D_in = D_in
        self.fullD = self.augment_D()
    def augment_D(self):
        if self.bc_left.isDir:
            fullD = np.concatenate([[[self.bc_left.b3/self.bc_left.b1]],self.D_in], 0)
        if self.bc_right.isDir:
            fullD = np.concatenate([self.D_in,[[self.bc_right.b3 / self.bc_right.b1]]], 0)
        return fullD

class PlotConvergenceComparison():
    def __init__(self, deg, bcl, bcr, f, u_exact, u_exact_derv, n_elems_vec=[2,4,8,16,32,64]):
        self.p = deg
        self.bfs_per_elem = self.p + 1
        self.quadrature = gq.Gauss_Quadrature1d(5,0,1)
        self.quad_wts = self.quadrature.quad_wts
        self.quad_pts = self.quadrature.quad_pts
        self.bcl = bcl
        self.bcr = bcr
        self.f = f
        self.u_exact = u_exact
        self.u_exact_derv = u_exact_derv
        self.n_elems_vec = n_elems_vec
        
        self.generate_log_h()
        self.generate_nodes()
        self.generate_D_vec()
        self.generate_err_vecs()


    def generate_log_h(self):
        h_vec_ = []
        log_h_vec_ = []
        log_h_vec_trunc_ = []
        for i in range(0, len(self.n_elems_vec)):
            n = self.n_elems_vec[i]
            h = 1 / n
            log_h = math.log(h)
            h_vec_.append(h)
            log_h_vec_.append(log_h)
            log_h_vec_trunc_.append(round(log_h, 3))
        self.h_vec = h_vec_
        self.log_h_vec = log_h_vec_
        self.log_h_vec_trunc = log_h_vec_trunc_
        self.log_h_vec_trunc.reverse()

    def generate_nodes(self, L=1.):
        # make nodes list for each number of elements, n=2,4,8... 
        self.L = L
        nodes_dict = {}
        for i in range(0, len(self.n_elems_vec)):
            # get number of elements n from 2,4,8...
            n = self.n_elems_vec[i]
            n_key = str(n)
            # create and store nodes list for n elements
            n_xvals = CreateNodes(0,self.L,n)
            nodes_dict[n_key] = n_xvals
        # list of nodes lists for n_elems=2,4,8...
        self.nodes_all_dict = nodes_dict
        return self.nodes_all_dict
    
    # WHERE FEM_POISSON IS ACTUALLY EXECUTED
    def generate_D_vec(self):
        fem_dict_ = {}
        initial_D_dict_ = {}
        full_D_dict_ = {}
        for i in range(0, len(self.n_elems_vec)):
            # get number of elements from 2,4,8...
            n = self.n_elems_vec[i]
            n_key = str(n)
            # get nodes list for n elements
            nodes = self.nodes_all_dict[n_key]
            # run FEM_Poisson to get initial D vec
            n_fem = FEM_Poisson(self.bcl, self.bcr, self.f, nodes, self.p)
            # store initial D vecs
            fem_dict_[n_key] = n_fem
            n_initial_D = n_fem.D
            initial_D_dict_[n_key] = n_initial_D
            # augment initial D vec if Dir BCs require
            n_full_D = GetFullD(self.bcl, self.bcr, n_initial_D).fullD
            full_D_dict_[n_key] = n_full_D
        self.fem_instances_dict = fem_dict_
        self.initial_D_dict = initial_D_dict_
        self.full_D_dict = full_D_dict_

    def generate_err_vecs(self):
        err_vecs_dict_ = {}
        l2_err_dict_ = {}
        h1_err_dict_ = {}
        l2_err_vec_ = []
        h1_err_vec_ = []
        for i in range(0, len(self.n_elems_vec)):
            n = self.n_elems_vec[i]
            n_key = str(n)
            n_dfull = self.full_D_dict[n_key]
            n_xvals = self.nodes_all_dict[n_key]
            # get the quadrature of this n_elem's instance of FEM_Poisson
            # n_quadrature = self.fem_instances_dict[n_key].quadrature
            n_quadrature = gq.Gauss_Quadrature1d(7,0,1)
            n_err_vec = ev.ErrorValues(n_dfull, n_xvals, self.p, n_quadrature, self.u_exact, self.u_exact_derv)
            n_err_l2 = n_err_vec[0]
            n_err_h1 = n_err_vec[1]
            l2_err_vec_.append(n_err_l2)
            h1_err_vec_.append(n_err_h1)
            err_vecs_dict_[n_key] = n_err_vec
            l2_err_dict_[n_key] = n_err_l2
            h1_err_dict_[n_key] = n_err_h1
        self.err_vecs_dict = err_vecs_dict_
        self.l2_err_dict = l2_err_dict_
        self.h1_err_dict = h1_err_dict_
        self.l2_err_vec = l2_err_vec_
        self.h1_err_vec = h1_err_vec_

    def plot_l2_h1_errs(self):
        fig, ax = plt.subplots()
        plt.xlabel("Element length (h)")
        plt.ylabel("Error ($u^h - u$)")
        xdom = self.h_vec # logs of n=2,4,8...
        l2err = self.l2_err_vec
        h1err = self.h1_err_vec
        ax.plot(xdom, l2err, label='L2 error')
        ax.plot(xdom, h1err, label='H1 error')
        ax.loglog()
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(ticker.FixedLocator(self.h_vec))

        ax.legend()
        plt.show()

    


            