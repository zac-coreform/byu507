#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:58:35 2023

@author: kendrickshepherd
"""

from enum import Enum
import numpy as np
import sys
import pprint as pp

class BCType(Enum):
    Dirichlet = 0
    Neumann = 1
    Robin = 2
    none = 100
    
class BoundaryCondition():
    
    def __init__(self,idx,edge_side_set):
        self.index = idx
        self.boundary_type = BCType.none
        self.node_to_edge = {}
        self.rhs = 0
        self.sol_multiplier = 0 
        self.sol_derv_multiplier = 0
        self.edge_side_set = edge_side_set
        
        self.__ConvertSideSetToDict__(edge_side_set)
        
    # convert a side set of edges into a dictionary of nodes and edges
    def __ConvertSideSetToDict__(self,edge_side_set):
        for edge in edge_side_set:
            for vidx in edge:
                if vidx in self.node_to_edge:
                    self.node_to_edge[vidx].append(edge)
                else:
                    self.node_to_edge[vidx] = [edge]
        
    def InitializeData(self, bdry_type, Rhs, sol_mult, sol_derv_mult):
        self.boundary_type = bdry_type
        self.rhs = Rhs
        self.sol_multiplier = sol_mult
        self.sol_derv_multiplier = sol_derv_mult
        
    def IsDirichlet(self):
        return self.boundary_type == BCType.Dirichlet

    def IsNeumann(self):
        return self.boundary_type == BCType.Neumann

    def IsRobin(self):
        return self.boundary_type == BCType.Robin

# MY HW7 VERSION  
# class BoundaryCondition():
#     def __init__(self, bc_posn, bc_type, b1, b2, b3):
#         self.bc_type = bc_type
#         self.bc_posn = bc_posn        
#         if self.bc_posn == "left":
#             self.bc_index = 0
#         elif self.bc_posn == "right":
#             self.bc_index = 1
#         else:
#             sys.exit("Unknown BC position: must be left or right")
        
#         self.b1 = b1
#         self.b2 = b2
#         self.b3 = b3
#         self.isDir = False
#         self.isNeu = False
#         self.isRob = False
        
#         if self.bc_type == "Dir":
#             self.isDir = True
#             self.g_val = self.b3 / self.b1
#         elif self.bc_type == "Neu":
#             self.isNeu = True
#             self.h_val = self.b3 / self.b2
#         elif self.bc_type == "Rob":
#             self.isRob = True
#             self.r_val = self.b3 / self.b2
#             self.r_val_k = self.b1 / self.b2
#         else:
#             sys.exit("Unknown BC type: must be Dir, Neu, or Rob")

# # usage
# # bcl0 = bc.BoundaryCondition("left", "Neu", 0, 1, 0)
# # bcr0 = bc.BoundaryCondition("right", "Dir", 1, 0, 0)
    
def get_NTE(boundaries):
    NTE_dict = {}
    NTE_dict['00_NTE'] = 1
    for bd in boundaries:
        idx = str(bd.index)
        bt = bd.boundary_type
        key = str(idx + '_' + str(bt))
        NTE_dict[key] = bd.node_to_edge
    pp.pprint(NTE_dict)

def get_ESS(boundaries):
    ESS_dict = {}
    ESS_dict['00_ESS'] = 1
    for bd in boundaries:
        idx = str(bd.index)
        bt = bd.boundary_type
        key = str(idx + '_' + str(bt))
        ESS_dict[key] = bd.edge_side_set
    pp.pprint(ESS_dict)

class get_dirdofs():
    def __init__(self,boundaries):
        self.dirichlet_dofs_set = set()
        self.dirichlet_dofs = {}
        self.bdry_idxs = len(boundaries)
        for b in range(0,self.bdry_idxs):
            bdry = boundaries[b]
            if bdry.IsDirichlet():
                for key in bdry.node_to_edge:
                    self.dirichlet_dofs_set.add(key)
                    self.dirichlet_dofs[key] = bdry.rhs/bdry.sol_multiplier
            # else: 
                # print(f"bdry type is {bdry.boundary_type}")
        # return dirichlet_dofs, dirichlet_dofs_set
        # pp.pprint(self.dirichlet_dofs)
        # pp.pprint(self.dirichlet_dofs_set)
        self.dd = list(self.dirichlet_dofs_set)

# Sets are unordered
# Set items are unchangeable
# Duplicates Not Allowed
# You cannot access items in a set by referring to an index, since sets are unordered the items has no index. But you can loop through the set items using a for loop, or ask if a specified value is present in a set, by using the in keyword.