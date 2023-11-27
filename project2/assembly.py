import unittest
import numpy as np
import basis
import integrate 
import mesh as mm


# for scalar/constant constit_coeff
def assemble_H1_inner_product(mesh, constit_coeff):
    elems = mm.generate_elements(mesh)
    enci = elems.enci
    nodes = elems.all_nodes
    n_elem = elems.n_elem
    num_nodes = (len(nodes))
    dof_per_node = 1
    num_global_coeffs = num_nodes * dof_per_node
    K = np.zeros((num_global_coeffs, num_global_coeffs))
    for e in range(0, n_elem):
        ke = elems.gen_ke(e)
        for a in range(0,2):
            A = enci[a, e]
            for b in range(0,2):
                B = enci[b, e]
                K[A,B] += ke[a, b]
    return K

# the function below creates Gold_K from rubric:
    # K.shape[0] = n_elem + 1
    # elem_len = (x1 - x0) / n_elem
    # n = (1 / elem_len) * 1
    # main diag = [n, 2n, 2n, ..., n] and 
    # neighboring diags = [-n, -n, ..., -n]
def make_gold_K(x0, x1, n_elem):
    dom = x1 - x0
    elem_len = dom / n_elem
    base = (1 / elem_len) * 1
    K_dim = n_elem + 1
    rows = cols = range(0, K_dim)
    K_ = np.zeros((K_dim, K_dim))
    for r in rows:
        for c in cols:
            if r == c:
                if r == c == 0 or r == c == (K_dim - 1):
                    K_[r,c] = base
                else:
                    K_[r,c] = 2 * base
            elif c == (r + 1) or r == (c + 1):
                K_[r,c] = base * -1
    return K_

class Test_assemble_H1_inner_product(unittest.TestCase):
    def test_unit_single_element(self):
        m1 = mm.generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=1)
        Gold_K = make_gold_K(0, 1, 1)
        Test_K = assemble_H1_inner_product(mesh=m1, constit_coeff=1)
        self.assertTrue(np.allclose(Gold_K, Test_K))
    def test_unit_two_elements(self):
        m2 = mm.generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=2)
        Gold_K = make_gold_K(0, 1, 2)
        Test_K = assemble_H1_inner_product(mesh=m2, constit_coeff=1)
        self.assertTrue(np.allclose(Gold_K, Test_K))
    def test_unit_five_elements(self):
        m5 = mm.generate_mesh(phys_xmin=0, phys_xmax=1, n_elem=5)
        Gold_K = make_gold_K(0, 1, 5)
        Test_K = assemble_H1_inner_product(mesh=m5, constit_coeff=1)
        self.assertTrue(np.allclose(Gold_K, Test_K))
    def test_nontrivial_single_element(self):
        m1 = mm.generate_mesh(phys_xmin=3, phys_xmax=7, n_elem=1)
        Gold_K = make_gold_K(3, 7, 1)
        Test_K = assemble_H1_inner_product(mesh=m1, constit_coeff=1)
        self.assertTrue(np.allclose(Gold_K, Test_K))
    def test_nontrivial_two_elements(self):
        m2 = mm.generate_mesh(phys_xmin=3, phys_xmax=7, n_elem=2)
        Gold_K = make_gold_K(3, 7, 2)
        Test_K = assemble_H1_inner_product(mesh=m2, constit_coeff=1)
        self.assertTrue(np.allclose(Gold_K, Test_K))
    def test_nontrivial_five_elements(self):
        m5 = mm.generate_mesh(phys_xmin=3, phys_xmax=7, n_elem=5)
        Gold_K = make_gold_K(3, 7, 5)
        Test_K = assemble_H1_inner_product(mesh=m5, constit_coeff=1)
        self.assertTrue(np.allclose(Gold_K, Test_K))






def assemble_l2_inner_product(ien, nodes, forcing_function):
    n_elem = mesh.get_num_elem_from_ien(ien)
    num_nodes = (len(nodes))
    dof_per_node = 1
    num_global_coeffs = num_nodes * dof_per_node
    F = np.zeros((num_global_coeffs, 1))
    for e in range(0, n_elem):
        ke = np.zeros((2,1))
        elem_domain = GetElementDomain(ien, e, nodes)
        xmin, xmax = elem_domain
        for a in range(0,2):
            N_a = lambda x: basis.eval_basis(xmin=xmin, xmax=xmax, N_idx=a, x=x)
            integrand = lambda x: N_a(x) * forcing_function(x)
            fe[a] = integrate_by_quadrature(function=integrand, x_lower=xmin, x_upper=xmax, n_quad=1)
        # insert local fe into global F
        for a in range(0,2):
            A = ien[a, e]
            F[A] += fe[a]
    return F



# class Test_assemble_l2_inner_product(unittest.TestCase):
    # def test_biunit_to_biunit(self):

    # def test_biunit_to_unit(self):

    # def test_biunit_to_nontrivial(self):
