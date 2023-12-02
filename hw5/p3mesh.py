import unittest
import numpy as np
import integrate
import basis

def generate_mesh(x0, x1, n_elem):
    nodes = np.linspace(x0, x1, n_elem + 1)
    ien = np.zeros((2,n_elem), dtype=int)
    for n in range(0, n_elem):
        ien[0,n] = n
        ien[1,n] = n+1
    return ien, nodes

def get_element_domain(ien, nodes, elem_idx):
    node0_idx = int(ien[0,elem_idx])# 0 0 
    node1_idx = int(ien[1,elem_idx])# 0 1
    domain = np.array((nodes[node0_idx], nodes[node1_idx]))
    return domain

def get_num_elem_from_ien(ien):
    return ien.shape[1]

def get_element_ke(e, ien, nodes, constit_coeff):

    ke = np.zeros((2,2))
    print("initial ke", ke)
    elem_domain = get_element_domain(ien, e, nodes)
    xmin, xmax = elem_domain
    print("xmin, xmax", xmin, xmax)
    for a in range(0,2):
        N_a = lambda x: basis.eval_basis_deriv(xmin=xmin, xmax=xmax, N_idx=a, x=x)
        # print("N_a", N_a)
        for b in range(0,2):
            N_b = lambda x: basis.eval_basis_deriv(xmin=xmin, xmax=xmax, N_idx=b, x=x)
            # print("N_b", N_b)
            integrand = lambda x: N_a(x) * constit_coeff * N_b(x)
            # print("integrand", integrand)
            soln = integrate.integrate_by_quadrature(function=integrand, x_lower=xmin, x_upper=xmax, n_quad=1)
            ke[a,b] = soln
    print("resulting ke\n", ke)
    return ke


class Test_get_element_ke(unittest.TestCase):
    def test_single_element(self):
        ien_, nodes_ = generate_mesh(x0=0, x1=1, n_elem=1)
        print("ien, nodes\n", ien_, nodes_)
        n_elem_ = get_num_elem_from_ien(ien_)
        print("n_elem\n", n_elem_)
        elem_index_iterable = range(0, n_elem_)
        constit_coeff = 1
        ke_list = []
        for e in elem_index_iterable:
            ke = get_element_ke(e, ien_, nodes_, constit_coeff)
            ke_list.append(ke)
        print("ke_list", ke_list)
        return ke_list
        # goldke = 
        # Testke = 

class Test_get_num_elem_from_ien(unittest.TestCase):
    def test_single_element(self):
        IEN = np.array([[0],[1]])
        goldNum = 1
        TestNum = get_num_elem_from_ien(ien=IEN)
        self.assertEqual(goldNum, TestNum)
    def test_four_element(self):
        IEN = np.array([[0, 1, 2, 3],
                            [1, 2, 3, 4]])
        goldNum = 4
        TestNum = get_num_elem_from_ien(ien=IEN)
        self.assertEqual(goldNum, TestNum)


class Test_generate_mesh(unittest.TestCase):
    def test_single_element(self):
        goldIEN = np.array([[0],
                            [1]])
        
        goldNodes = np.array((0.,1.))
        
        TestIEN, TestNodes = generate_mesh(x0=0, x1=1, n_elem=1)

        self.assertTrue(np.allclose(goldIEN, TestIEN))
        self.assertTrue(np.allclose(goldNodes, TestNodes))    
    
    def test_1(self):
        goldIEN = np.array([[0, 1, 2, 3],
                            [1, 2, 3, 4]])
        
        goldNodes = np.array((0., 0.25, 0.5, 0.75, 1.))
        
        TestIEN, TestNodes = generate_mesh(x0=0, x1=1, n_elem=4)

        self.assertTrue(np.allclose(goldIEN, TestIEN))
        self.assertTrue(np.allclose(goldNodes, TestNodes))

class Test_get_element_domain(unittest.TestCase):
    def test_1(self):
        goldDomain = np.array((0.5, 0.75))
        ien, nodes = generate_mesh(x0=0, x1=1, n_elem=4)
        TestDomain = get_element_domain(ien=ien, elem_idx=2, nodes=nodes)
        self.assertTrue(np.allclose(goldDomain, TestDomain))
