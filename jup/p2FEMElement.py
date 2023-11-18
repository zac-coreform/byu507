import numpy as np
import p2BasisFunctions as bf

np.set_printoptions(suppress=True)

class Element():
    def __init__(self, problem, index):
        # problem-wide data
        self.n_quad = problem.n_quad
        self.quad = problem.quad
        self.q_pts = problem.quad.pts
        self.q_wts = problem.quad.wts
        self.n_bfs = problem.n_bf_per_elem
        self.left = problem.left
        self.right = problem.right
        self.f = problem.f
        self.nodes = problem.node_coords
        self.n_nodes = int(len(problem.node_coords))
        self.n_elems = problem.n_elems

        # element data
        self.index = index
        self.x0 = problem.node_coords[index]
        self.x1 = problem.node_coords[index + 1]
        self.length = self.x1 - self.x0
        self.first = (True if (self.x0 == problem.domain_start) else False)
        self.last = (True if (self.x1 == problem.domain_end) else False)
        self.fe_initial = np.zeros((self.n_bfs, 1))
        self.ke_initial = np.zeros((self.n_bfs, self.n_bfs)) 

        # generate fe
        self.generate_fe(index)
        self.fe = self.fe_initial

        # generate ke
        self.generate_ke(index)
        self.ke = self.ke_initial

    def generate_fe(self, index):
        for q in range(0, self.n_quad):
            w_q = self.q_wts[q]
            xi_q = self.q_pts[q]
            x_q = bf.XMap(self.x0, self.x1, xi_q)
            x_derv = bf.XMapDerv(self.x0, self.x1, xi_q)
            
            for a in range(0, self.n_bfs):
                fe_bf_addition = w_q * bf.NBasis(a, -1, 1, xi_q) * self.f(x_q) * x_derv
                self.fe_initial[a] += fe_bf_addition

        if index == 0:

            if self.left.isDir:

                g_0 = self.left.g_val
                for q in range(0, self.n_quad):
                    w_q = self.q_wts[q]
                    xi_q = self.q_pts[q]
                    x_q = bf.XMap(self.x0, self.x1, xi_q)
                    x_derv = bf.XMapDerv(self.x0, self.x1, xi_q)

                    for a in range(0, self.n_bfs):
                        fe_bc_contribution = w_q * bf.NBasisDerv(a, -1, 1, xi_q) * bf.NBasisDerv(0, -1, 1, xi_q) * x_derv**(-1) * g_0
                        self.fe_initial[a] += fe_bc_contribution

            elif self.left.isNeu:
                h_0 = self.left.h_val
                for a in range(0, self.n_bfs):
                    fe_bc_contribution = bf.NBasis(a, self.x0, self.x1, bf.XMap(self.x0, self.x1, -1)) * h_0
                    self.fe_initial[a] -= fe_bc_contribution # lhs so * -1

            elif self.left.isRob:
                r_0 = self.left.r_val
                for a in range(0, self.n_bfs):
                    fe_bc_contribution = bf.NBasis(a, self.x0, self.x1, bf.XMap(self.x0, self.x1, -1)) * r_0
                    self.fe_initial[a] -= fe_bc_contribution # lhs so * -1
                if self.right.isDir:
                    pass

        if index == self.n_elems - 1:

            if self.right.isDir:
                g = self.right.g_val
                for q in range(0, self.n_quad):
                    w_q = self.q_wts[q]
                    xi_q = self.q_pts[q]
                    x_q = bf.XMap(self.x0, self.x1, xi_q)
                    x_derv = bf.XMapDerv(self.x0, self.x1, xi_q)

                    for a in range(0, self.n_bfs):
                        fe_bc_contribution = w_q * bf.NBasisDerv(a, -1, 1, xi_q) * bf.NBasisDerv(0, -1, 1, xi_q) * x_derv**(-1) * g
                        self.fe_initial[a] += fe_bc_contribution 

            elif self.right.isNeu:
                h_L = self.right.h_val
                for a in range(0, self.n_bfs):
                    fe_bc_contribution = bf.NBasis(a, self.x0, self.x1, bf.XMap(self.x0, self.x1, 1)) * h_L
                    # self.fe_initial[a] -= fe_bc_contribution
                    self.fe_initial[a] += fe_bc_contribution # rhs so * +1

            elif self.right.isRob:
                r_L = self.right.r_val
                for a in range(0, self.n_bfs):
                    fe_bc_contribution = bf.NBasis(a, self.x0, self.x1, bf.XMap(self.x0, self.x1, 1)) * r_L
                    self.fe_initial[a] += fe_bc_contribution # rhs so * +1
                if self.left.isDir:
                    pass

        self.fe = self.fe_initial
        
    def generate_ke(self, index):
        for q in range(0, self.n_quad):
            w_q = self.q_wts[q]
            xi_q = self.q_pts[q]
            x_q = bf.XMap(self.x0, self.x1, xi_q)
            x_derv = bf.XMapDerv(self.x0, self.x1, xi_q)
            x_derv_inv = 1 / x_derv

            for a in range(0, self.n_bfs):
                Na_x = bf.NBasisDerv(a, -1, 1, xi_q)
                for b in range(a, self.n_bfs):
                    Nb_x = bf.NBasisDerv(b, -1, 1, xi_q)
                    ke_bf_addition = w_q * Na_x * Nb_x * x_derv_inv
                    self.ke_initial[a,b] += ke_bf_addition

        # symmetry
        for a in range(0, self.n_bfs):
            for b in range(a, self.n_bfs):
                self.ke_initial[b,a] = self.ke_initial[a,b]

        if index == 0:
            if self.left.isDir:
                pass

            elif self.left.isNeu:
                pass

            elif self.left.isRob:
                r_0 = self.left.r_val_k
                # ke_bc_contribution = bf.NBasis(0, -1, 1, -1) * r_0
                # self.ke_initial[0,0] -= ke_bc_contribution
                for a in range(0, self.n_bfs):
                    ke_bc_contribution = bf.NBasis(a, -1, 1, bf.XMap(self.x0, self.x1, -1)) * bf.NBasis(0, -1, 1, bf.XMap(self.x0, self.x1, -1)) * r_0
                    self.ke_initial[a] -= ke_bc_contribution # lhs so * -1

        if index == self.n_elems - 1:
            if self.right.isDir:
                pass
            
            elif self.right.isNeu:
                pass
            
            elif self.right.isRob:
                r_L = self.right.r_val_k
                # ke_bc_contribution = r * bf.NBasis(0, -1, 1, -1)**2
                # self.ke_initial[-1,-1] -= ke_bc_contribution
                for a in range(0, self.n_bfs):
                    ke_bc_contribution = bf.NBasis(a, -1, 1, bf.XMap(self.x0, self.x1, 1)) * bf.NBasis(1, -1, 1, bf.XMap(self.x0, self.x1, 1)) * r_L
                    self.ke_initial[a] += ke_bc_contribution # rhs so * +1


        self.ke = self.ke_initial

# Na(0/L)Nb(0/L)
# bf.NBasis(a, -1, 1, bf.XMap(self.x0, self.x1, -1/1)) * bf.NBasis(0/L, -1, 1, bf.XMap(self.x0, self.x1, -1/1))
# if e0, bcl
# bf.NBasis(a, -1, 1, bf.XMap(self.x0, self.x1, -1)) * bf.NBasis(0, -1, 1, bf.XMap(self.x0, self.x1, -1))
# if eL, bcr
# bf.NBasis(a, -1, 1, bf.XMap(self.x0, self.x1, 1)) * bf.NBasis(1, -1, 1, bf.XMap(self.x0, self.x1, 1))

