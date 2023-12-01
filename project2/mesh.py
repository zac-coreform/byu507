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

class Test_gen_all_nodes(unittest.TestCase):
    def test_single_element_all_nodes(self):
        m1 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=1)    
        Gold_all_nodes = np.array((0.,1.))
        Test_all_nodes = m1.all_nodes
        self.assertTrue(np.allclose(Gold_all_nodes, Test_all_nodes))
    def test_four_element_all_nodes(self):
        m4 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=4)   
        Gold_all_nodes = np.array((0., 0.25, 0.5, 0.75, 1.))
        Test_all_nodes = m4.all_nodes
        self.assertTrue(np.allclose(Gold_all_nodes, Test_all_nodes))     

class Test_gen_elem_node_coord_indices(unittest.TestCase):
    def test_single_elem_node_coord_indices(self):
        m1 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=1) 
        Gold_elem_node_coord_indices = np.array([[0],
                            [1]])
        Test_elem_node_coord_indices = m1.elem_node_coord_indices
        self.assertTrue(np.allclose(Gold_elem_node_coord_indices, Test_elem_node_coord_indices))
    def test_four_elem_node_coord_indices(self):
        m4 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=4)   
        Gold_elem_node_coord_indices = np.array([[0, 1, 2, 3],
                                                 [1, 2, 3, 4]])
        Test_elem_node_coord_indices = m4.elem_node_coord_indices
        self.assertTrue(np.allclose(Gold_elem_node_coord_indices, Test_elem_node_coord_indices)) 


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

    # ke for a single element e given the mesh info passed to generate_elements above    
    # def gen_ke(self, e):
    #     # mx = bs.stock_mx()
    #     mx_ = np.zeros((2,2))
    #     e_dom = self.get_element_domain(e)
    #     e_x0 = e_dom[0]
    #     e_x1 = e_dom[1]
    #     # mx_mult = bs.mxm(e_x0, e_x1)
    #     inv_term = bs.x_xi_deriv_inv(e_x0, e_x1)
    #     jac = ig.get_jacobian(e_x0, e_x1)
    #     print(f"jac is {jac} and inv term is {inv_term}")  
    #     theory = bool(inv_term == 1 / jac)
    #     print(theory) 
    #     for a in range(0,2):
    #         N_a_xi_term = bs.eval_basis_deriv(xmin=e_x0, xmax=e_x1, bf_idx=a, x_in=0) # x_in does not matter
    #         print(f"bf a term = {N_a_xi_term}")
    #         for b in range(0,2):
    #             # N_b_xi_term = bs.N_a_xi(b)
    #             N_b_xi_term = bs.eval_basis_deriv(xmin=e_x0, xmax=e_x1, bf_idx=b, x_in=0) # x_in does not matter
    #             print(f"bf b term = {N_b_xi_term}")
    #             # def prodfn(aa, bb):
    #             #     return jac * N_a_xi_term * N_b_xi_term
    #             # def quad_fn(aa,bb):
    #             #     return lambda xi: prodfn(a,b)
    #             def quad_fn(xi=0, a=0, b=0):
    #                 return lambda xi: N_a_xi_term * N_b_xi_term * inv_term
    #             ke_entry_incr = ig.integrate_by_quadrature(function=quad_fn, x_lower=e_x0, x_upper=e_x1, n_quad=1, a_in=a, b_in=b)
    #             print(f"entry incr is {ke_entry_incr}")
    #             mx_[a, b] += ke_entry_incr
    #     ke = mx_
    #     return ke

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
    # def gen_fe2(self, e, f):
    #     print(f"\n gen_fe for element idx={e}")
    #     # temp fast_fe
    #     e_dom = generate_elements(self.mesh).get_element_domain(e)
    #     e_x0 = e_dom[0]
    #     e_x1 = e_dom[1]
    #     # f_x0 = f(e_x0)
    #     # f_x1 = f(e_x1)
    #     f_x = lambda x: f(x)
    #     Na_x = lambda a, x: bs.eval_basis(bf_idx=a, xmin=e_x0, xmax = e_x1, x_in=x)
    #     prod = lambda a, x: Na_x(x) * f_x(x)
    #     integ = ig.integrate_by_quadrature(function=prod, x_lower=e_x0, x_upper=e_x1, n_quad=1)
    #     print(f"integral of Na*f = {integ}")
    #     Na_x0 = bs.eval_basis(bf_idx=)
    #     f_21 = (2*f_x0 + f_x1)
    #     print(f"f21 is {f_21}")
    #     f_12 = (f_x0 + 2 * f_x1)
    #     print(f"f12 is {f_12}")
    #     he = e_x1 - e_x0
    #     he6 = he/6
    #     print(f"he is {he}")
    #     fe_0 = he6 * f_21
    #     print(f"fe_0 is {fe_0}")
    #     fe_1 = he6 * f_12
    #     print(f"fe_1 is {fe_1}")
    #     fe = np.array(([fe_0], [fe_1]))
    #     return fe



class Test_get_element_domain(unittest.TestCase):
    def test_get_element_domain_1_elem(self):
        m1 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=1)
        m1_elems = generate_elements(m1)  
        Gold_element_domain = np.array((0.,1.))
        Test_element_domain = m1.all_nodes
        self.assertTrue(np.allclose(Gold_element_domain, Test_element_domain))
    def test_get_element_domain_2_elems(self):
        m2 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=2)
        m2_elems = generate_elements(m2) 
        Gold_elem_coords = [np.array((0.,0.5)), np.array((0.5,1.))]
        for e in range(0, m2.n_elem):
            Gold_element_domain = Gold_elem_coords[e]
            Test_element_domain = m2_elems.get_element_domain(e)
            self.assertTrue(np.allclose(Gold_element_domain, Test_element_domain))
    def test_get_element_domain_4_elems(self):
        m4 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=4)
        m4_elems = generate_elements(m4)
        Gold_elem_coords = [np.array((0., 0.25)), np.array((0.25, 0.5)), np.array((0.5, 0.75)), np.array((0.75, 1.))]
        for e in range(0, m4.n_elem):
            Gold_element_domain = Gold_elem_coords[e]
            Test_element_domain = m4_elems.get_element_domain(e)
            self.assertTrue(np.allclose(Gold_element_domain, Test_element_domain))

class Test_get_element_ke(unittest.TestCase):
    def test_m1_element_ke(self):
        m1 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=1)
        m1_elems = generate_elements(m1)  
        Gold_ke = np.array(([1., -1.], [-1., 1.]))
        print(f"Gold ke is\n {Gold_ke}")
        Test_ke = m1_elems.gen_ke(0)
        print(f"test ke is\n {Test_ke}")
        self.assertTrue(np.allclose(Gold_ke, Test_ke))
    def test_m2_element_ke(self):
        m1 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=2)
        m1_elems = generate_elements(m1)  
        Gold_ke = np.array(([2., -2.], [-2., 2.]))
        print(f"Gold ke is\n {Gold_ke}")
        Test_ke = m1_elems.gen_ke(0)
        print(f"test ke is\n {Test_ke}")
        self.assertTrue(np.allclose(Gold_ke, Test_ke))
    def test_m2_element_ke(self):
        m1 = generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=3)
        m1_elems = generate_elements(m1)  
        Gold_ke = np.array(([3., -3.], [-3., 3.]))
        print(f"Gold ke is\n {Gold_ke}")
        Test_ke = m1_elems.gen_ke(0)
        print(f"test ke is\n {Test_ke}")
        self.assertTrue(np.allclose(Gold_ke, Test_ke))
    # def test_single_element(self):
    #     ien_, nodes_ = generate_mesh(x0=0, x1=1, n_elem=2)
    #     # print(f"ien:\n {ien_} \nnodes\n{nodes_}")
    #     n_elem_ = get_num_elem_from_ien(ien_)
    #     # print("n_elem\n", n_elem_)
    #     elem_index_iterable = range(0, n_elem_)
    #     constit_coeff = 1
    #     ke_list = []
    #     for e in elem_index_iterable:
    #         ke = get_element_ke(e, ien_, nodes_, constit_coeff)
    #         ke_list.append(ke)
    #     # print("ke_list", ke_list)
    #     return ke_list
    #     # goldke = 
    #     # Testke = 
    # def test_two_elements(self):
    #     ien_, nodes_ = generate_mesh(x0=0, x1=0.5, n_elem=1)
    #     print(f"ien:\n {ien_} \nnodes\n{nodes_}")
    #     n_elem_ = get_num_elem_from_ien(ien_)
    #     print("n_elem\n", n_elem_)
    #     elem_index_iterable = range(0, n_elem_)
    #     constit_coeff = 1
    #     ke_list = []
    #     for e in elem_index_iterable:
    #         ke = get_element_ke(e, ien_, nodes_, constit_coeff)
    #         ke_list.append(ke)
    #     # print("ke_list", ke_list)
    #     return ke_list

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





