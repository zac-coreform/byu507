#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:03:40 2023

@author: kendrickshepherd
"""

import numpy as np
import sys
import matplotlib
from scipy import integrate
from matplotlib import pyplot as plt

import p3Gauss_Quadrature_2D as gq
import p3Surface_Mesh as sm
import p3Boundary_Conditions as bc
import p3Basis_Functions_2D as bf
import pprint as pp


def CreateIENArray(mesh):
    return mesh.faces.transpose()
#     n_elems = len(mesh.faces) # 8
#     elems = mesh.faces
#     bfs_per_elem = NBasisFunctions(mesh,0) # random representative
#     IEN = np.zeros((bfs_per_elem, n_elems)).astype('int')
#     for e_ in range(0, n_elems):
#         for a_ in range(0, bfs_per_elem):
#             elem_arr = elems[e_]
#             Q = elem_arr[a_]
#             IEN[a_,e_] = Q
#     return IEN


# hacky tired brain; surely there's some simple way to do this
def get_all_vertices(mesh):
    fcs = mesh.faces
    loop = len(fcs)
    mins = []
    maxes = []
    for i in range(0, loop):
        f_vs = fcs[i]
        f_min = min(f_vs)
        f_max = max(f_vs)
        mins.append(f_min)
        maxes.append(f_max)
    min_ = min(mins)
    max_ = max(maxes) + 1
    vs = np.arange(min_, max_, 1, int)
    return vs

# BELOW, ID ARRAY: 
# IN dirichlet_dofs = set of kv pairs where 
    # boundaries = [self.left_bc,self.right_bc,self.down_bc,self.top_bc]
    # keys = if Dir boundary --> add keys in boundary's node_to_edge dict
    # value = boundary.rhs/boundary.sol_mult = (g?)
def CreateIDArray(mesh,dirichlet_dofs):
    n_elems = len(mesh.faces)
    vs = get_all_vertices(mesh)
    n_vs = len(vs)
    ID = np.zeros((2, n_vs)).astype('int')
    ID[0] += vs
    b_vs = mesh.boundary_vertices()
    for i in range(0, len(vs)):
        if vs[i] in b_vs:
            if vs[i] in dirichlet_dofs:
                ID[1,i] = -1
    counter = 0
    for j in range(0, len(vs)):
        if ID[1,j] != -1:
            ID[1,j] = counter
            counter += 1
    return ID


# determine the number of basis functions on a mesh element
def NBasisFunctions(mesh,elem_idx):
    if mesh.is_triangle_face(elem_idx):
        sys.exit("Cannot currently operate on triangular elements")
    elif mesh.is_quadrilateral_face(elem_idx):
        n_bfs = 4
    else:
        sys.exit("Cannot operate on non-triangular or quadrilateral face")
        
    return n_bfs


# Evaluate thermal conductivity of an element at its center
def EvaluateKappaAtElementCenter(mesh,elem_idx,kappa):
    x_pts = mesh.get_face_points(elem_idx)[:,0:2]
    if len(x_pts)==4:
        x_center = bf.XMap(x_pts, 0, 0)
    else:
        x_center = bf.XMap(x_pts, 0.33, 0.33)
    elem_kappa = kappa(x_center[0],x_center[1])
    return elem_kappa


def LocalStiffness(mesh,elem_idx,kappa,quadrature):    
    elem_kappa = EvaluateKappaAtElementCenter(mesh, elem_idx, kappa)
    n_quad = quadrature.n_quad
    xiloop = etaloop = n_quad
    pts_xi = quadrature.quad_pts[0]
    pts_eta = quadrature.quad_pts[1]
    wts_xi = quadrature.quad_wts[0]
    wts_eta = quadrature.quad_wts[1]
    xiloop = len(pts_xi)
    etaloop = len(pts_eta)
    x_pts = mesh.get_face_points(elem_idx)[:,0:2]
    #. ^ global xi, eta coords of face vertices, x0 and x1
    #. ^ equiv of x_pts arg in 1D: used to calc XMapDerv component of base ke equation
    n_bfs = NBasisFunctions(mesh, elem_idx)
    ke = np.zeros((n_bfs, n_bfs))
    
    # start quadrature points loop:
    for i in range(0, xiloop):
        for j in range(0, etaloop):
            pt = quadrature.get_point(i, j)
            wt = quadrature.get_weight(i, j)

            for a in range(0, n_bfs):
                for b in range(0, n_bfs):
                    J = bf.JacobianDet(x_pts, pt[0], pt[1])                    
                    a_grad = bf.SpatialGradient(a, x_pts, pt[0], pt[1])
                    b_grad = bf.SpatialGradient(b, x_pts, pt[0], pt[1])
                    kappa_term = np.dot(elem_kappa, b_grad)                    
                    grad_terms = np.dot(kappa_term, a_grad)
                    incr = wt * grad_terms * J
                    ke[a,b] += incr 
    #// TODO: Complete this local assembly routine
    return ke


def LocalForceBoundaryConditions(mesh, elem_idx, fe, boundaries, quadrature, kappa_in):
    face_vert_idxs = mesh.faces[elem_idx]
    dirichlet_nodes = {}
    neumann_edges = {}
    robin_edges = {}


    # ascribe boundary information on this element
    for boundary in boundaries:
        if boundary.IsDirichlet():
            for vert_idx in face_vert_idxs:
                if vert_idx in boundary.node_to_edge:
                    dirichlet_nodes[vert_idx] = boundary.rhs/boundary.sol_multiplier
        elif boundary.IsNeumann() or boundary.IsRobin():
            for vert_idx in face_vert_idxs:
                if vert_idx in boundary.node_to_edge:
                    edges = boundary.node_to_edge[vert_idx]
                    for edge in edges:
                        if edge[0] in face_vert_idxs and edge[1] in face_vert_idxs:
                            if boundary.IsNeumann():
                                neumann_edges[edge] = boundary.rhs/boundary.sol_derv_multiplier
                            elif boundary.IsRobin():
                                sys.exit("Robin conditions not yet computed")
                                robin_edges.add(edge)
    
    # TODO: Comment on what this is doing
    # the code above ^ checks each boundary edge of the domain against the current face's vertices 
    # if a current face vertex appears in the vertices listed as part of a Neumann boundary, 
    # we make a dictionary entry with the edge vertices and the associated Neumann h-value
    # below, we pop off each dictionary entry one by one, split the edge into its two vertices, 
    # then check the current face's four vertices to see if they match either of the current dictionary entry vertices
    # if they do, we add them to a list, which we use below to assign the correct adjustment to the element fe vector
    # then 
    if len(neumann_edges) != 0:
        while len(neumann_edges) != 0:
            
            d_data = neumann_edges.popitem()
            d_edge = d_data[0]
            v0 = d_edge[0]
            v1 = d_edge[1]
            h_val = d_data[1]
            idxs = []
            for idx in range(0,4):
                if v0 == face_vert_idxs[idx]:
                    idxs.append(idx)
                elif v1 == face_vert_idxs[idx]:
                    idxs.append(idx)
            idxs.sort()
            
            # jacobian is constant and half the length of the mesh element
            length = np.linalg.norm(mesh.vs[v0]-mesh.vs[v1])
            halflength = length/2
            
            # sys.exit("The following needs to take into account thermal conductivity. It also seems to be 2x bigger than it should be")
            # bottom side
            if 0 in idxs and 1 in idxs:
                n0 = lambda x: bf.NBasis(0, x, -1)
                n1 = lambda x: bf.NBasis(1, x, -1)
                fe[0] += integrate.quadrature(n0,-1,1)[0] * h_val * halflength
                fe[1] += integrate.quadrature(n1,-1,1)[0] * h_val * halflength
            # left side
            elif 1 in idxs and 2 in idxs:
                n1 = lambda x: bf.NBasis(1, 1, x)
                n2 = lambda x: bf.NBasis(2, 1, x)
                fe[1] += integrate.quadrature(n1,-1,1)[0] * h_val * halflength
                fe[2] += integrate.quadrature(n2,-1,1)[0] * h_val * halflength
            # top side
            elif 2 in idxs and 3 in idxs:
                n2 = lambda x: bf.NBasis(2, x, 1)
                n3 = lambda x: bf.NBasis(3, x, 1)
                fe[2] += integrate.quadrature(n2,-1,1)[0] * h_val * halflength
                fe[3] += integrate.quadrature(n3,-1,1)[0] * h_val * halflength
            # right side
            elif 0 in idxs and 3 in idxs:
                n0 = lambda x: bf.NBasis(0, -1, x)
                n3 = lambda x: bf.NBasis(3, -1, x)
                fe[0] += integrate.quadrature(n0,-1,1)[0] * h_val * halflength
                fe[3] += integrate.quadrature(n3,-1,1)[0] * h_val * halflength

    # TODO: Complete and comment this
    ien = CreateIENArray(mesh)
    n_bfs = NBasisFunctions(mesh, elem_idx)
    # if there are any Dirichlet nodes (dirichlet_nodes is not empty)
    if len(dirichlet_nodes) != 0:
        kappa = kappa_in
        elem_kappa = EvaluateKappaAtElementCenter(mesh, elem_idx, kappa)
        # check each member of dirichlet_nodes to see if it 
        while len(dirichlet_nodes) != 0:
            # TODO: Fill in here
            # the code above ^ (around line 143) checks each boundary edge of the domain against the current face's vertices 
            # if a current face vertex appears in the vertices listed as part of a Dirichlet boundary, 
            # we make a dictionary entry with the edge vertices and the associated Dirichlet g-value
            # below, we pop off each dictionary entry one by one
            # then check the current face's four vertices to see if they match any of the current dictionary entry vertices
            # if they do, we take the associated g-value and kick off a quadrature routine that uses g 
            # to calculate the decrement adjustment prescribed for the current element's fe
            d_data = dirichlet_nodes.popitem()
            for fix in range(0, n_bfs):
                vx = ien[fix, elem_idx]
                if vx == d_data[0]:
                    g = d_data[1]
                    n_quad = quadrature.n_quad
                    xiloop = etaloop = n_quad
                    pts_xi = quadrature.quad_pts[0]
                    pts_eta = quadrature.quad_pts[1]
                    wts_xi = quadrature.quad_wts[0]
                    wts_eta = quadrature.quad_wts[1]
                    xiloop = len(pts_xi)
                    etaloop = len(pts_eta)
                    x_pts = mesh.get_face_points(elem_idx)[:,0:2]
                    
                    for i in range(0, xiloop):
                        for j in range(0, etaloop):
                            pt = quadrature.get_point(i, j)
                            fixed_grad = bf.SpatialGradient(fix, x_pts, pt[0], pt[1])
                            J = bf.JacobianDet(x_pts, pt[0], pt[1])
                            wt = quadrature.get_weight(i, j)
                            for a in range(0, n_bfs):
                                
                                a_grad = bf.SpatialGradient(a, x_pts, pt[0], pt[1])
                                kappa_term = np.dot(elem_kappa, fixed_grad)
                                grad_terms = np.dot(a_grad, kappa_term)
                                incr = wt * g * J * grad_terms
                                fe[a] -= incr

    return fe


def LocalForce(mesh,elem_idx,f,boundaries,kappa,quadrature):
    n_quad = quadrature.n_quad
    xiloop = etaloop = n_quad
    pts_xi = quadrature.quad_pts[0]
    pts_eta = quadrature.quad_pts[1]
    wts_xi = quadrature.quad_wts[0]
    wts_eta = quadrature.quad_wts[1]
    xiloop = len(pts_xi)
    etaloop = len(pts_eta)

    n_bfs = NBasisFunctions(mesh, elem_idx)
    fe = np.zeros((n_bfs,1))
    x_pts = mesh.get_face_points(elem_idx)[:,0:2]     
    
    #// TODO: Fill in here
    for i in range(0, xiloop):
        for j in range(0, etaloop):
            pt = quadrature.get_point(i, j)
            wt = quadrature.get_weight(i, j)
            
            for a in range(0, n_bfs):
                # for b in range(0, n_bfs):
                J = bf.JacobianDet(x_pts, pt[0], pt[1])
                x_g = bf.XMap(x_pts, pt[0], pt[1])
                f_term = f(x_g)
                a_grad = bf.NBasis(a, pt[0], pt[1])
                incr = wt * J * a_grad * f_term
                fe[a] += incr
    
    fe = LocalForceBoundaryConditions(mesh,elem_idx,fe, boundaries, quadrature, kappa)
    return fe


class FEM_Diffusion2d():
    def __init__(self,mesh,boundaries,f,quadrature,kappa):
        self.mesh = mesh
        self.n_elems = len(mesh.faces)
        self.boundaries = boundaries
        self.f = f
        self.quadrature = quadrature
        self.kappa = kappa
        self.n_elem = len(mesh.faces)

        # self.D = np.zeros(len(self.mesh.vs))
        #.^ ?? where'd this come from? cf. below, D = linalg solve

        self.dirichlet_dofs = set()
        for bdry in self.boundaries:
            if bdry.IsDirichlet():
                for key in bdry.node_to_edge:
                    self.dirichlet_dofs.add(key)

        self.IEN = CreateIENArray(self.mesh)
        self.ID = CreateIDArray(self.mesh,self.dirichlet_dofs)
        self.n_unknowns = int(max(self.ID[1])) + 1
        
        self.K = np.zeros((self.n_unknowns,self.n_unknowns))
        self.F = np.zeros((self.n_unknowns,1))
        self.ke_list = []
        self.fe_list = []

        for e in range(0, self.n_elems):
            ke = LocalStiffness(self.mesh,e,self.kappa,self.quadrature)
            self.ke_list.append(ke)
            fe = LocalForce(self.mesh,e,self.f,self.boundaries,self.kappa,self.quadrature)
            self.fe_list.append(fe)

            # careful here -- see email about BFs, one per vertex, etc
            n_elem_bfs = NBasisFunctions(self.mesh, e)
            
            for a in range(0,n_elem_bfs):
                P = self.IEN[a,e]
                A = self.ID[1,P]
                if A == -1:
                    continue
                self.F[A] += fe[a]
                for b in range(0,n_elem_bfs):
                    Q = self.IEN[b,e]
                    B = self.ID[1,Q]
                    if B == -1:
                        continue
                    self.K[A,B] += ke[a,b]
        self.D = np.linalg.solve(self.K,self.F)

        # return self.D


# TODO: Comment and complete this code
# basically we're just adding the Dirichlet values to the initial solution's D-vector
# in more detail:
# here we take the boundary edges and check each one for Dirichlet-ness
# when we find a Dir, we retrieve its two boundary vertices
# and make a dictionary entry with key = vertex and value = g
# then we make a new, empty D-vector list, 
# and we check every mesh vertex against our Dirichlet dictionary
# when we find a match, we add the vertex g-value to our new D-vector
# when there's no match, we just add the value found in the original D-vector
def ConcatenateToFullD(mesh,boundaries,D):

    flatD = D.flatten()
    
    dirichlet_dofs = {}
    for bdry in boundaries:
        if bdry.IsDirichlet():
            for key in bdry.node_to_edge:
                dirichlet_dofs[key] = bdry.rhs/bdry.sol_multiplier

    ID = CreateIDArray(mesh,dirichlet_dofs)
    
    Dtotal = []
    # TODO: Populate Dtotal based on computed values of D
    #       and the known values on dirichlet boundary points
    for i in range(0,len(mesh.vs)):
        if i in dirichlet_dofs:
            Dtotal.append(dirichlet_dofs[i])
        else:
            Dtotal.append(flatD[ID[1,i]])
            # Dtotal.append(0) # TODO: editme!
    return np.array(Dtotal)


# TODO: Comment on and use this code
# hmm. I think I have a general idea, but it's getting late
# 
def PlotTriangulationSolution(mesh,DTotal):
    # we split all the mesh vertices and make separate lists of all the x's and y's
    X = mesh.vs[:,0]
    Y = mesh.vs[:,1]
    faces = []
    # then check each face: if quad, then we grab its vertices and add them to a list 
    # in sets of three in order hard-coded below... I think we're making triangles? 
    # i confess I'm not fully following... 
    for i in range(0,len(mesh.faces)):
        if mesh.is_quadrilateral_face(i):
            faces.append([mesh.faces[i][0],\
                          mesh.faces[i][1],\
                          mesh.faces[i][2]])

            faces.append([mesh.faces[i][2],\
                          mesh.faces[i][3],\
                          mesh.faces[i][0]])
        else:
            faces.append([mesh.faces[i][0],\
                          mesh.faces[i][1],\
                          mesh.faces[i][2]])
    
    
    array_faces = np.array(faces,dtype=int)
    
    triangles = matplotlib.tri.Triangulation(X, Y, triangles=array_faces)
    
    matplotlib.rcParams['figure.dpi'] = 300
    
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tcf = ax1.tricontourf(triangles, DTotal)
    fig1.colorbar(tcf)
    ax1.tricontour(triangles, DTotal, colors='k')
    ax1.set_title('Contour plot of temperature distribution')
    
    ax1.triplot(triangles, color='1',linewidth=0.1)


# Define thermal conductivity information for a brick
# 20 cm x 10 cm brick
def BrickKappa_BrickTC_Larger(x, y):
    # mortarkappa = 3.3
    mortarkappa = 0.00001 # 100000x smaller
    claykappa = 1.0
    if (x-15)**2 + (y-5)**2 <= 2:
        return mortarkappa
    elif (x-5)**2 + (y-5)**2 <= 2:
        return mortarkappa
    else:
        return claykappa

def BrickKappa_EqualTC(x, y):
    mortarkappa = 1.0 # equal 
    claykappa = 1.0
    if (x-15)**2 + (y-5)**2 <= 2:
        return mortarkappa
    elif (x-5)**2 + (y-5)**2 <= 2:
        return mortarkappa
    else:
        return claykappa

def BrickKappa_MortarTC_XLarger(x, y):
    mortarkappa = 10000 # 10000x larger
    claykappa = 1.0
    if (x-15)**2 + (y-5)**2 <= 2:
        return mortarkappa
    elif (x-5)**2 + (y-5)**2 <= 2:
        return mortarkappa
    else:
        return claykappa


# Define side sets for use in boundary conditions
def ProcessIntoSideSets(mesh):
    left = []
    right = []
    top = []
    bottom = []
    for edge in mesh.boundary_edges():
        v0 = mesh.vs[edge[0]]
        v1 = mesh.vs[edge[1]]
        v = v0-v1
        if abs(v[0]) < 1e-6:
            if abs(v0[0]) < 1e-6:
                left.append(edge)
            else:
                right.append(edge)
        else:
            if abs(v0[1]) < 1e-6:
                bottom.append(edge)
            else:
                top.append(edge)
    return [left,bottom,right,top]
            

####################################################
####################################################

# Problem execution

class run_problem_():
    def __init__(self, n_quad):
        self.n_quad = n_quad
        self.temp = 10
        self.f = lambda x: 0
        self.quadrature = gq.Gauss_Quadrature2d(self.n_quad,-1,1)
        self.kappa = BrickKappa_EqualTC
        self.make_mesh()
        self.ien = CreateIENArray(self.mesh)
        self.make_bcs()
    def make_mesh(self):
        # self.mesh = sm.SurfaceMesh.FromOBJ_FileName("files/rectangle_plate_COARSE.obj")
        self.mesh = sm.SurfaceMesh.FromOBJ_FileName("files/brick.obj")
        [self.left,self.bottom,self.right,self.top]=ProcessIntoSideSets(self.mesh)
    def make_bcs(self):
        self.left_bc = bc.BoundaryCondition(0, self.left)
        self.right_bc = bc.BoundaryCondition(2, self.right)
        self.down_bc = bc.BoundaryCondition(1, self.bottom)
        self.top_bc = bc.BoundaryCondition(3, self.top)
        self.left_bc.InitializeData(bc.BCType.Neumann, 0, 0, 1)
        self.right_bc.InitializeData(bc.BCType.Neumann, 0, 0, 1)
        self.down_bc.InitializeData(bc.BCType.Dirichlet, self.temp, 1, 0)
        self.top_bc.InitializeData(bc.BCType.Neumann, 0.3, 0, 1)
        self.boundaries = [self.left_bc,self.right_bc,self.down_bc,self.top_bc]

    def solve(self):
        self.prob = FEM_Diffusion2d(self.mesh,self.boundaries,self.f,self.quadrature,self.kappa)

        self.D = self.prob.D
        self.DTotal = ConcatenateToFullD(self.mesh,self.boundaries,self.D)

    def plot(self):
        PlotTriangulationSolution(self.mesh,self.DTotal)

