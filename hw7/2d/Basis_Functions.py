import numpy as np
import sys
from matplotlib import pyplot as plt

# Also, create a function to evaluate the partial derivative of a two-dimensional basis function (NBasisPartial) in the accompanying code. 
# Plot the resulting derivatives using PlotBasisFunction(a, True, 0) and PlotBasisFunction(a, True, 1). 
# Then comment the code for 
    # NBasisPartial, 
    # XMapPartial, 
    # JacobianMatrix, 
    # JacobianDet, 
    # ParametricGradient, and 
    # SpatialGradient.

# previously: NBasis(a,x0,x1,x)
def NBasis(a,xi,eta):
    if a == 0:
      xi_a = -1
      eta_a = -1
    elif a == 1:
      xi_a = 1
      eta_a = -1
    elif a == 2:
      xi_a = 1
      eta_a = 1
    else:
      xi_a = -1
      eta_a = 1

    val=(1/4)*(1+xi*xi_a)*(1+eta*eta_a)
    return val

# for a = 0, vals are -1, -1, so
# (1/4)*(1-x)*(1-y)
# for a = 1, vals are 1, -1, so
# (1/4)*(1+x)*(1-y)
# for a = 2, vals are 1, 1, so
# (1/4)*(1+x)*(1+y)
# for a = 3, vals are -1, 1, so
# (1/4)*(1-x)*(1+y)

# code from last time
def NBasisZ(a,xi,eta):
    xi_vals = [-1,1,1,-1]
    eta_vals = [-1,-1,1,1]
    return (0.5 * (1 + xi_vals[a] * xi)) * (0.5 * (1 + eta_vals[a] * eta))


def NBasisPartial(a,xi,eta,direction):
    # partial derivatives of BFs 0-3 are:
    # A   WRT_XI          WRT_ETA
    # 0   eta/4 - 1/4     xi/4 - 1/4
    # 1   1/4 - eta/4    -xi/4 - 1/4
    # 2   eta/4 + 1/4     xi/4 + 1/4
    # 3  -eta/4 - 1/4     1/4 - xi/4

    # tabulating the coeffs and constants with 0.25 factored out gives:
    wrt_xi_coeff_base = [1, -1, 1, -1]
    wrt_xi_const_base = [-1, 1, 1, -1]
    wrt_eta_coeff_base = [1, -1, 1, -1]
    wrt_eta_const_base = [-1, -1, 1, 1]
    
    # selecting the correct free variable and coeffs and consts lists
    if direction == 0:
       coeffs = wrt_xi_coeff_base
       consts = wrt_xi_const_base
       var = eta
    elif direction == 1:
        coeffs = wrt_eta_coeff_base
        consts = wrt_eta_const_base
        var = xi
    # selecting correct entry using BF index
    coeff = coeffs[a]
    const = consts[a]
    # reconstructing the derivative expression
    expr = 0.25 * (coeff * var + const)
    return expr

    
#evaluate the transformation mapping
def XMap(x_pts,xi,eta):
    x = np.zeros(len(x_pts[0]))
    for a in range(0,4):
        x += NBasis(a,xi,eta) * x_pts[a]    #The range is from 0 to 1
    return x

# code from last time
def XMapZ(x_pts,xi,eta):
    pts_arrays = []
    for pt in x_pts:
        pts_arrays.append(np.array(pt))
    xy_out = np.zeros((2,1))
    for i in range(0, 4):
        nbi = NBasis(i, xi, eta)
        pti = pts_arrays[i]
        prod = nbi * pti
        xy_out[0] += prod[0]
        xy_out[1] += prod[1]
    return xy_out

# For XMap, examples of xi, eta from below are:
    # xvals = np.linspace(-1,1,n_x) # n_x = 100
    # yvals = np.linspace(-1,1,n_y) # n_y = 101
    # for i in len xvals: xvals[i],
    # for j in len xvals: yvals[j]
# So: sample coordinate in the 2d space
# Ex x_pts is:
    # xpts = [[2,1],[5,-1],([8,0]),([4,4])]


def XMapPartial(x_pts,xi,eta,direction):
    # xpts is the 2d equivalent of providing the original domain bounds
    # creating 2-entry array
    xpartial = np.zeros(len(x_pts[0]))
    for a in range(0,4):
        # same as XMap but with derivs
        xpartial += NBasisPartial(a, xi, eta, direction)
    # with linear BFs, these are going to almost always be zero, no?
    # returns a "point"?
    return xpartial


def JacobianMatrix(x_pts,xi,eta):
    partial_xi = XMapPartial(x_pts, xi, eta, 0)
    partial_eta = XMapPartial(x_pts, xi, eta, 1)
    jac_matrix = np.zeros((2,2))
    # assuming idea here is that the scaling factor that is the jacobian can vary both the xi and eta dimensions, so we have an array, essentially an array of the respective 1d multipliers, or something like that? 
    jac_matrix[0,0] = partial_xi[0]
    jac_matrix[1,0] = partial_xi[1]
    jac_matrix[0,1] = partial_eta[0]
    jac_matrix[1,1] = partial_eta[1]


def JacobianDet(x_pts,xi,eta):
    # tbh I don't remember what the purpose of this is... does sound familiar from class; I'll have to go review my notes
    return np.linalg.det(JacobianMatrix(x_pts, xi, eta))


def ParametricGradient(a,xi,eta):
    grad = np.zeros(2)
    grad[0] = NBasisPartial(a, xi, eta, 0)
    grad[1] = NBasisPartial(a, xi, eta, 1)
    # makes sense: the partial deriv in each direction
    return grad


def SpatialGradient(a,x_pts,xi,eta):
    F = JacobianMatrix(x_pts,xi,eta)
    p_grad = ParametricGradient(a, xi, eta)
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
        # 
        N[0,i,j] = temp[0]       #in xi direction
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