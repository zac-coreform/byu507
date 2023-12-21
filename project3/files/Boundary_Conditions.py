#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:58:35 2023

@author: kendrickshepherd
"""

from enum import Enum
import numpy as np
import sys

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