import numpy as np
import sys
from matplotlib import pyplot as plt
import Gauss_Quadrature1d as gq
import Boundary_Conditions as bc
import Basis_Functions as bf
import math
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

def CreateIENArray(deg, n_elems):
    p = deg
    bfs_per_elem = p + 1
    # total_bfs = p * n_elems + 1
    IEN = np.zeros((bfs_per_elem, n_elems)).astype('int')
    for e in range(0, n_elems):
        for a in range(0, bfs_per_elem):
            Q = a + p * e
            IEN[a,e]=Q
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

def CreateXVals(phys_dom_0, phys_dom_1, ien_or_n_elems):
    if isinstance(ien_or_n_elems, np.ndarray):
        n_elems = ien.shape[1]
    elif isinstance(ien_or_n_elems, int):
        n_elems = ien_or_n_elems
    else:
        sys.exit("last arg must be ien array or n_elems int")
    xvals = np.linspace(phys_dom_0,phys_dom_1,n_elems+1)
    return xvals

def LocalStiffness(e, xvals, bc_left, bc_right, deg):
    # changing to degree-determined quadrature
    p = deg
    bfs_per_elem = p + 1
    # degree 2n − 1 polynomials, where n is number of sample points used // n = (p+1)/2
    n_quad = math.ceil((p + 1) / 2)
    quadrature = gq.Gauss_Quadrature1d(n_quad=n_quad,quad_domain_0=0, quad_domain_1=1)
    #. ^ with quad_domain_N args, returns pts mapped to [0,1]

    x0 = xvals[e]
    x1 = xvals[e+1]
    
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

def LocalForce(e,xvals,f,bc_left,bc_right,deg):
    # as above in ke, changing to deg-determined quadrature
    p = deg
    bfs_per_elem = p + 1
    n_quad = math.ceil((p + 1) / 2)
    quadrature = gq.Gauss_Quadrature1d(n_quad=n_quad,quad_domain_0=0, quad_domain_1=1)

    x0 = xvals[e]
    x1 = xvals[e+1]
    
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

def FEM_Poisson(bc_left,bc_right,f,xvals,deg):
    # print(f"STARTING FEM_POISSON")
    p = deg
    bfs_per_elem = p + 1
    n_quad = math.ceil((p + 1) / 2)
    # # print(f"calc'd n_quad for p={p} is {n_quad}")
    quadrature = gq.Gauss_Quadrature1d(n_quad=n_quad,quad_domain_0=0, quad_domain_1=1)
    n_elems = len(xvals)-1
    IEN = CreateIENArray(p, n_elems)
    # def CreateIDArray(bc_left, bc_right, deg, n_elems):
    ID = CreateIDArray(bc_left, bc_right, p, n_elems)
    n_unknowns = int(max(ID)) + 1
    K = np.zeros((n_unknowns,n_unknowns))
    F = np.zeros((n_unknowns,1))
    ke_list = []
    fe_list = []
    for e in range(0,n_elems):
        ke = LocalStiffness(e,xvals,bc_left,bc_right,p)
        # print(f"ke for elem {e} is \n{ke}")
        ke_list.append(ke)
        fe = LocalForce(e,xvals,f,bc_left,bc_right,p)
        fe_list.append(fe)
        # print(f"fe for elem {e} is \n{fe}")
        for a in range(0,bfs_per_elem):
            P = IEN[a,e]
            A = ID[P]
            if A == -1:
                continue
            F[A] += fe[a]
            for b in range(0,bfs_per_elem):
                Q = IEN[b,e]
                B = ID[Q]
                if B == -1:
                    continue
                K[A,B] += ke[a,b]
    # print(f"final K is \n{K}")
    # print(f"final F is \n{F}")

    Kdet = np.linalg.det(K)
    Kcond = np.linalg.cond(K)

    D = [IEN, ID, K, F, ke_list, fe_list]

    # D = np.linalg.solve(K,F)
    return D

def PlotSolution(bc_left,bc_right,xvals,D,title=""):
    if bc_left.isDir:
        D = np.concatenate([[[bc_left.b3/bc_left.b1]],D], 0)
    if bc_right.isDir:
        D = np.concatenate([D,[[bc_right.b3 / bc_right.b1]]], 0)
    plt.plot(xvals,D)
    if title != "":
        plt.title(title)    
    
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
        self.n_quad = math.ceil((self.p + 1) / 2)
        self.quadrature = gq.Gauss_Quadrature1d(n_quad=self.n_quad,quad_domain_0=0, quad_domain_1=1)
        self.quad_wts = self.quadrature.quad_wts
        self.quad_pts = self.quadrature.quad_pts
        self.bcl = bcl
        self.bcr = bcr
        self.f = f
        self.u_exact = u_exact
        self.u_exact_derv = u_exact_derv
        self.n_elems_vec = n_elems_vec
        
        self.generate_log_h()
        self.generate_xvals()
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

    def generate_xvals(self):
        x_vals_dict = {}
        for i in range(0, len(self.n_elems_vec)):
            n = self.n_elems_vec[i]
            n_key = str(n)
            n_xvals = CreateXVals(0,1,n)
            x_vals_dict[n_key] = n_xvals
        self.x_vals_all_dict = x_vals_dict
        return self.x_vals_all_dict
    
    def generate_D_vec(self):
        initial_D_dict_ = {}
        full_D_dict_ = {}
        for i in range(0, len(self.n_elems_vec)):
            n = self.n_elems_vec[i]
            n_key = str(n)
            xvals = self.x_vals_all_dict[n_key]
            n_initial_D = FEM_Poisson(self.bcl, self.bcr, self.f, xvals, self.p)
            initial_D_dict_[n_key] = n_initial_D
            n_full_D = GetFullD(self.bcl, self.bcr, n_initial_D).fullD
            full_D_dict_[n_key] = n_full_D
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
            n_xvals = self.x_vals_all_dict[n_key]
            n_err_vec = ev.ErrorValues(n_dfull, n_xvals, self.p, self.quadrature, self.u_exact, self.u_exact_derv)
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
        # self.h_vec.reverse()
        xdom = self.h_vec
        l2err = self.l2_err_vec
        h1err = self.h1_err_vec
        ax.plot(xdom, l2err, label='L2 error')
        ax.plot(xdom, h1err, label='H1 error')
        ax.loglog()
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(ticker.FixedLocator(self.h_vec))
        # instantiate top axis
        # ax2 = ax.twiny()
        # top = self.n_elems_vec
        # ax2.plot(top, label='num elements')
        # ax2.xaxis.set_major_locator(ticker.FixedLocator(self.n_elems_vec))
        ax.legend()
        plt.show()

    


            