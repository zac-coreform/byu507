import unittest
import numpy as np

def generateMesh(x0, x1, n_elem):
    nodes = np.linspace(x0, x1, n_elem + 1)
    ien = np.zeros((2,n_elem), dtype=int)
    for n in range(0, n_elem):
        ien[0,n] = n
        ien[1,n] = n+1
    return ien, nodes

def GetElementDomain(ien, elem_idx, nodes):
    node0_idx = ien[0,elem_idx]
    node1_idx = ien[1,elem_idx]
    domain = np.array((nodes[node0_idx], nodes[node1_idx]))
    return domain

def get_num_elem_from_ien(ien):
    return ien.shape[1]


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


class Test_GenerateMesh(unittest.TestCase):
    def test_single_element(self):
        goldIEN = np.array([[0],
                            [1]])
        
        goldNodes = np.array((0.,1.))
        
        TestIEN, TestNodes = generateMesh(x0=0, x1=1, n_elem=1)

        self.assertTrue(np.allclose(goldIEN, TestIEN))
        self.assertTrue(np.allclose(goldNodes, TestNodes))    
    
    def test_1(self):
        goldIEN = np.array([[0, 1, 2, 3],
                            [1, 2, 3, 4]])
        
        goldNodes = np.array((0., 0.25, 0.5, 0.75, 1.))
        
        TestIEN, TestNodes = generateMesh(x0=0, x1=1, n_elem=4)

        self.assertTrue(np.allclose(goldIEN, TestIEN))
        self.assertTrue(np.allclose(goldNodes, TestNodes))

class Test_GetElementDomain(unittest.TestCase):
    def test_1(self):
        goldDomain = np.array((0.5, 0.75))
        ien, nodes = generateMesh(x0=0, x1=1, n_elem=4)
        TestDomain = GetElementDomain(ien=ien, elem_idx=2, nodes=nodes)
        self.assertTrue(np.allclose(goldDomain, TestDomain))
