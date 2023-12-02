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
    
class BCOrientation(Enum):
    Left = 0
    Right = 1
    none = 100
    
class BoundaryCondition():
    
    def __init__(self,idx):
        self.index = idx
        self.boundary_type = BCType.none
        self.boundary_orientation = BCOrientation.none
        self.rhs = 0
        self.sol_multiplier = 0 
        self.sol_derv_multiplier = 0
        
    def InitializeData(self, bdry_type, bdry_orientation, Rhs, sol_mult, sol_derv_mult):
        self.boundary_type = bdry_type
        self.boundary_orientation = bdry_orientation
        self.rhs = Rhs
        self.sol_multiplier = sol_mult
        self.sol_derv_multiplier = sol_derv_mult
        
    def IsDirichlet(self):
        return self.boundary_type == BCType.Dirichlet

    def IsNeumann(self):
        return self.boundary_type == BCType.Neumann

    def IsRobin(self):
        return self.boundary_type == BCType.Robin