#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:36:38 2023

@author: kendrickshepherd
"""

def NBasis(a,x0,x1,x):
    denom = x1-x0
    if a == 0:
        numer = x1-x
    elif a == 1:
        numer = x-x0
    return numer/denom

def NBasisDerv(a,x0,x1,x):
    denom = x1-x0
    if a == 0:
        numer = -1
    elif a == 1:
        numer = 1
    return numer/denom

def XMap(x0,x1,ksi):
    x = 0
    xvals = [x0,x1]
    for a in range(0,2):
        x += NBasis(a,-1,1,ksi) * xvals[a]
    return x

def XMapDerv(x0,x1,ksi):
    x_derv = 0
    xvals = [x0,x1]
    for a in range(0,2):
        x_derv += NBasisDerv(a,-1,1,ksi) * xvals[a]
    return x_derv