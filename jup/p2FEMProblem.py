import numpy as np
from matplotlib import pyplot as plt
import p2GaussianQuadrature as gq
import p2FEMElement as fe

class SolutionPackage():
    def __init__(self, Dout, Dplot, K, F, element_list, fe_list, ke_list, ien, id, n_elems, node_coords, unknowns, Klen, Flen, Dlen, felen, kelen):
        self.D_out = Dout
        self.D_plot = Dplot
        self.K = K
        self.F = F
        self.element_list = element_list
        self.fe_list = fe_list
        self.ke_list = ke_list
        self.IEN = ien
        self.ID = id
        self.n_elems = n_elems
        self.node_coords = node_coords
        self.n_nodes = len(self.node_coords)
        self.n_total_bfs = self.node_coords
        self.nun = unknowns
        self.Klen = Klen
        self.Flen = Flen
        self.Dlen = Dlen
        self.felen = felen
        self.kelen = kelen



class Problem():
    def __init__(self, bcl, bcr, f, n_elems, L=int(1), n_quad=int(3), deg=int(1)):
        self.n_quad = n_quad
        self.quad = gq.GaussianQuadrature(n_quad)
        self.q_pts = self.quad.pts
        self.q_wts = self.quad.wts
        self.degree = deg
        self.n_bf_per_elem = int(deg + 1)
        self.n_elems = n_elems
        
        self.n_elems = n_elems
        self.n_nodes = int(n_elems + 1)
        self.n_total_bfs = self.n_nodes
        self.node_coords = np.linspace(0, L, self.n_nodes)
        self.nodeloop = range(0, self.n_nodes)
        self.elemloop = range(0, self.n_elems)
        self.elemlist_initial = []

        self.domain_start = min(self.node_coords)
        self.domain_end = max(self.node_coords)
        self.left = bcl
        self.right = bcr
        self.f = f

        self.createElements()
        self.elemlist = self.elemlist_initial
        self.elemcount = len(self.elemlist)

        self.createIENArray()
        self.createIDArray()
        self.n_unknowns = int(max(self.ID) + 1)

        self.F_ = np.zeros((self.n_unknowns, 1))
        self.K_ = np.zeros((self.n_unknowns, self.n_unknowns))

        self.assembleFandK()


# =========================================== METHODS

    def createIENArray(self):
        self.IEN = np.zeros((self.n_bf_per_elem,self.n_elems))
        for e in range(0,self.n_elems):
            for a in range(0,self.n_bf_per_elem):
                self.Q = a+e
                self.IEN[a,e] = self.Q
        return self.IEN

    def createIDArray(self):
        self.ID = np.zeros(self.n_total_bfs)
        
        if self.left.isDir:
            self.ID[0] = -1
            self.start_idx = 1
        elif self.left.isNeu or self.left.isRob:
            self.start_idx = 0
        else:
            sys.exit("Don't know LEFT boundary condition")
            
        self.counter = 0
        for i in range(self.start_idx, self.n_total_bfs):
            self.ID[i] = self.counter
            self.counter += 1
        
        if self.right.isDir:
            self.ID[-1] = -1
        elif self.right.isNeu or self.right.isRob:
            pass
        else:
            sys.exit("Don't know RIGHT boundary condition")

        return self.ID
        
    def createElements(self):
        for e in self.elemloop:

            elem = fe.Element(self, e)
            self.elemlist_initial.append(elem)        

# ================================================= GLOBAL ASSEMBLY

    def assembleFandK(self):
        for e in range(0,self.n_elems):
            elem = self.elemlist[e]
        
            for a in range(0,self.n_bf_per_elem):
                self.P = int(self.IEN[a, e])
                self.A = int(self.ID[self.P])
                if self.A == -1:
                    continue
                self.F_[self.A] += elem.fe[a]
        
                for b in range(0,self.n_bf_per_elem):
                    self.Q = int(self.IEN[b, e])
                    self.B = int(self.ID[self.Q])
                    if self.B == -1:
                        continue
                    self.K_[self.A, self.B] += elem.ke[a,b]
        
        self.F = self.F_
        self.K = self.K_

    def SolveOnly(self):
        self.D_out = np.linalg.solve(self.K,self.F)
        self.D_plot = self.D_out
        if self.left.isDir:
            self.node_first = self.left.g_val
            self.D_plot = np.insert(self.D_out, 0, self.node_first)

        if self.right.isDir:
            self.node_last = self.right.g_val
            self.D_plot = np.append(self.D_plot, self.node_last)

        self.fe_list = [self.elemlist[e].fe for e in range(0, len(self.elemlist))]
        self.ke_list = [self.elemlist[e].ke for e in range(0, len(self.elemlist))]
        self.Klen = len(self.K)
        self.Flen = len(self.F)
        self.Dlen = len(self.D_plot)
        self.felen = len(self.fe_list)
        self.kelen = len(self.ke_list)

        sol = SolutionPackage(self.D_out, self.D_plot, self.K, self.F, self.elemlist, self.fe_list, self.ke_list, self.IEN, self.ID, self.n_elems, self.node_coords, self.n_unknowns, self.Klen, self.Flen, self.Dlen, self.kelen, self.felen)

        return sol


    def PlotSolution(self):
        plt.plot(self.node_coords, self.D_plot, linewidth=2)

