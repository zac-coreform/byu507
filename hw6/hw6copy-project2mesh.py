import unittest
import numpy as np
import integrate as ig
import basis as bs
import inspect

# IN: PHYSICAL DOMAIN MIN, MAX AND N_ELEMS
# OUT: 
    # NODE COORDS ARRAY, 
    # ELEM NODE PAIRS ARRAY (COL 0 == ELEM 0 PHYS XMIN, XMAX)
    # PER-ELEMENT DOMAIN
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


class generate_elements():
    def __init__(self, mesh):
        # self.e = e
        self.mesh = mesh
        self.phys_xmin = mesh.phys_xmin
        self.phys_xmax = mesh.phys_xmax
        self.n_elem = mesh.n_elem
        self.all_nodes = mesh.all_nodes
        self.enci = mesh.elem_node_coord_indices
    def get_element_domain(self, e):
        n0_idx = int(self.enci[0,e])
        n1_idx = int(self.enci[1,e])
        self.element_domain = np.array((self.all_nodes[n0_idx], self.all_nodes[n1_idx]))
        return self.element_domain


    def gen_ke(self, e):
        mx_ = np.zeros((2,2))
        e_dom = self.get_element_domain(e)
        e_x0 = e_dom[0]
        e_x1 = e_dom[1]
        inv_term = bs.x_xi_deriv_inv(e_x0, e_x1)
        jac = ig.get_jacobian(e_x0, e_x1)
        for a in range(0,2):
            N_a_xi_term = bs.eval_basis_deriv(xmin=e_x0, xmax=e_x1, bf_idx=a, x_in=0) # x_in does not matter
            for b in range(0,2):
                N_b_xi_term = bs.eval_basis_deriv(xmin=e_x0, xmax=e_x1, bf_idx=b, x_in=0) # x_in does not matter
                prod_fn = lambda xi: N_a_xi_term * N_b_xi_term * inv_term
                ke_entry_incr = ig.integrate_by_quadrature(function=prod_fn, x_lower=e_x0, x_upper=e_x1, n_quad=1)
                mx_[a, b] += ke_entry_incr
        ke = mx_
        return ke
    
    def gen_fe(self, e, f):
        print(f"\n gen_fe for element idx={e}")
        # temp fast_fe
        e_dom = generate_elements(self.mesh).get_element_domain(e)
        e_x0 = e_dom[0]
        e_x1 = e_dom[1]
        f_x0 = f(e_x0)
        f_x1 = f(e_x1)
        f_21 = (2*f_x0 + f_x1)
        print(f"f21 is {f_21}")
        f_12 = (f_x0 + 2 * f_x1)
        print(f"f12 is {f_12}")
        he = e_x1 - e_x0
        he6 = he/6
        print(f"he is {he}")
        fe_0 = he6 * f_21
        print(f"fe_0 is {fe_0}")
        fe_1 = he6 * f_12
        print(f"fe_1 is {fe_1}")
        fe = np.array(([fe_0], [fe_1]))
        return fe



# using Hughes, we have fe = he/6 * [(2f1 + f2), (f1 + 2f2)]
# what are f1 and f2? in my project 1 code they are just f(e_x0) and f(e_x1)
# why? sth to do with interpolating the forcing function
# meantime, can write fast_fe as:

def fast_fe(e, mesh, f):
    e_dom = generate_elements(mesh).get_element_domain(e)
    e_x0 = e_dom[0]
    e_x1 = e_dom[1]
    f_x0 = f(e_x0)
    f_x1 = f(e_x1)
    f_21 = (2*f_x0 + f_x1)
    f_12 = (f_x0 + 2 * f_x1)
    he = e_x1 - e_x0
    fe_0 = he * f_21
    fe_1 = he * f_12
    fe = np.array(([fe_0], [fe_1]))
    return fe





