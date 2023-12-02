import numpy as np
import sys
from matplotlib import pyplot as plt

def NBasis(a,ksi,eta):
    if a == 0:
      ksi_a = -1
      eta_a = -1
    elif a == 1:
      ksi_a = 1
      eta_a = -1
    elif a == 2:
      ksi_a = 1
      eta_a = 1
    else:
      ksi_a = -1
      eta_a = 1

    val=(1/4)*(1+ksi*ksi_a)*(1+eta*eta_a)
    return val

def NBasisPartial(a,ksi,eta,direction):
    pass

def XMap(x_pts,ksi,eta):  #evaluate the transformation mapping
    x = np.zeros(len(x_pts[0]))
    for a in range(0,4):
        x += NBasis(a,ksi,eta) * x_pts[a]    #The range is from 0 to 1
    return x

def XMapPartial(x_pts,ksi,eta,direction):
    xpartial = np.zeros(len(x_pts[0]))
    for a in range(0,4):
        xpartial += NBasisPartial(a, ksi, eta, direction)
    return xpartial

def JacobianMatrix(x_pts,ksi,eta):
    partial_ksi = XMapPartial(x_pts, ksi, eta, 0)
    partial_eta = XMapPartial(x_pts, ksi, eta, 1)
    jac_matrix = np.zeros((2,2))
    jac_matrix[0,0] = partial_ksi[0]
    jac_matrix[1,0] = partial_ksi[1]
    jac_matrix[0,1] = partial_eta[0]
    jac_matrix[1,1] = partial_eta[1]

def JacobianDet(x_pts,ksi,eta):
    return np.linalg.det(JacobianMatrix(x_pts, ksi, eta))

def ParametricGradient(a,ksi,eta):
    grad = np.zeros(2)
    grad[0] = NBasisPartial(a, ksi, eta, 0)
    grad[1] = NBasisPartial(a, ksi, eta, 1)
    return grad

def SpatialGradient(a,x_pts,ksi,eta):
    F = JacobianMatrix(x_pts,ksi,eta)
    p_grad = ParametricGradient(a, ksi, eta)
    return np.linalg.solve(F.transpose(),p_grad)

# input four points as a list of lists
# e.g. xpts = [[2,1],[5,-1],([8,0]),([4,4])]  
def PlotTransformationMap(x_pts):
    n_x = 100
    n_y = 101     
    
    x_pts_arrays = []
    for pt in x_pts:
        x_pts_arrays.append(np.array(pt))
    
    xvals = np.linspace(-1,1,n_x) 
    yvals = np.linspace(-1,1,n_y)
    
    X, Y = np.meshgrid(xvals, yvals)   
    
    
    N = np.zeros((3,n_x,n_y))
    for i in range(0,len(xvals)):
      for j in range(0,len(yvals)):
        temp = XMap(x_pts_arrays,xvals[i],yvals[j])
        N[0,i,j] = temp[0]       #in ksi direction
        N[1,i,j] = temp[1]       #in eta direction
        if len(temp) > 2:
          N[2,i,j] = temp[2]
        else:
          N[2,i,j] = 0
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(N[0],N[1],N[2])
    if(len(x_pts_arrays[0])) == 2:     # ?
      ax.view_init(azim=270,elev=90)
    plt.show()


# input the index of the basis function you want to plot
def PlotBasisFunction(a, derivative = False, derv_num = 0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-1.0, 1.0, 0.05)
    X, Y = np.meshgrid(x, y)
    if not derivative:
        zs = np.array(NBasis(a,np.ravel(X), np.ravel(Y)))
    else:
        zs = np.array(NBasisPartial(a,np.ravel(X), np.ravel(Y), derv_num))
        
    Z = zs.reshape(X.shape)
    
    ax.plot_surface(X, Y, Z)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()