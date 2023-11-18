import unittest
import numpy as np
import basis
import integrate 
import mesh


# for scalar/constant constit_coeff
def assemble_H1_inner_product(ien, nodes, constit_coeff):
    n_elem = mesh.get_num_elem_from_ien(ien)
    num_nodes = (len(nodes))
    dof_per_node = 1
    num_global_coeffs = num_nodes * dof_per_node
    K = np.zeros((num_global_coeffs, num_global_coeffs))# make this a function in mesh.py
    # 16 - 27 break out as function: local assemble H1 inner product 
    # for test, use constit_coeff=1, for 3 elems, write out on paper what k1, k2, k3 should be
    for e in range(0, n_elem):
    #     ke = np.zeros((2,2))
    #     elem_domain = GetElementDomain(ien, e, nodes)
    #     xmin, xmax = elem_domain
    #     for a in range(0,2):
    #         N_a = lambda x: basis.eval_basis_deriv(xmin=xmin, xmax=xmax, N_idx=a, x=x)
    #         for b in range(0,2):
    #             N_b = lambda x: basis.eval_basis_deriv(xmin=xmin, xmax=xmax, N_idx=b, x=x)
    #             integrand = lambda x: N_a(x) * constit_coeff * N_b(x)
    #             ke[a,b] = integrate_by_quadrature(function=integrand, x_lower=xmin, x_upper=xmax, n_quad=1)
        # insert local ke into global K
        ke = mesh.get_element_ke(e, ien, nodes)
        print("ke from sep function", ke)
        for a in range(0,2):
            A = ien[a, e]
            for b in range(0,2):
                B = ien[b, e]
                K[A,B] += ke[a, b]
    return K
        
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

# class Test_assemble_H1_inner_product(unittest.TestCase):
    # def test_biunit_to_biunit(self):

    # def test_biunit_to_unit(self):

    # def test_biunit_to_nontrivial(self):

# class Test_assemble_l2_inner_product(unittest.TestCase):
    # def test_biunit_to_biunit(self):

    # def test_biunit_to_unit(self):

    # def test_biunit_to_nontrivial(self):
