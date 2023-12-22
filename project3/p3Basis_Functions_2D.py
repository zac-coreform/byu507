import numpy as np
import sys
from matplotlib import pyplot as plt
import p3Gauss_Quadrature_2D as gq2

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

# code from last time
def NBasisZ(a,xi,eta):
    xi_vals = [-1,1,1,-1]
    eta_vals = [-1,-1,1,1]
    return (0.5 * (1 + xi_vals[a] * xi)) * (0.5 * (1 + eta_vals[a] * eta))

def NBasisPartial(a,xi,eta,direction):
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
    # x_pts is the 2D equivalent of providing the global x interval to which you want xi to be mapped -- the target domain
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
        # xpartial += NBasisPartial(a, xi, eta, direction)
        xpartial += x_pts[a] * NBasisPartial(a, xi, eta, direction)
        #.^ corrected, from email
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
    return jac_matrix

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
    # print(f"x_pts passed in = {x_pts}")
    # n_x = 100
    # n_y = 101
    n_x = 10
    n_y = 11
    
    # change the list of lists to array of arrays
    x_pts_arrays = []
    for pt in x_pts:
        x_pts_arrays.append(np.array(pt))
    # print(f"x_pts_arrays = {x_pts_arrays}")
    
    xvals = np.linspace(-1,1,n_x) 
    yvals = np.linspace(-1,1,n_y)
    #.^ 100, 101 points between -1 and 1
    # print(f"xvals is \n{xvals}")
    # print(f"yvals is \n{yvals}")
    X, Y = np.meshgrid(xvals, yvals)   
    
    N = np.zeros((3,n_x,n_y))
    #.^ array of 3 arrays of 100x101
    # print(len(N)) # 3
    # print(f"N array is {N}")
    for i in range(0,len(xvals)):
      for j in range(0,len(yvals)):
        # print(f"xvals[i],yvals[j] are {round(xvals[i], 2)} and {round(yvals[j], 2)}")
        temp = XMap(x_pts_arrays,xvals[i],yvals[j])
        # print(f"XMap conversion is: {temp}")
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
def PlotBasisFunction(a, derivative = False, derv_direction = 0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-1.0, 1.0, 0.05)
    X, Y = np.meshgrid(x, y)
    if not derivative:
        zs = np.array(NBasis(a,np.ravel(X), np.ravel(Y)))
    else:
        zs = np.array(NBasisPartial(a,np.ravel(X), np.ravel(Y), derv_direction))
        
    Z = zs.reshape(X.shape)
    
    ax.plot_surface(X, Y, Z)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()

def PlotPoints2d(x_,y_,z_):
    # if type(x_) == int and type(y_) == int and type(z_) == int:
    #    x_plot = x_
    #    y_plot = y_
    #    z_plot = z_

    

    # plt.rcParams["figure.figsize"] = [7.50, 7.50]
    plt.rcParams["figure.autolayout"] = True

    fig = plt.figure()
    fig.set_size_inches(6, 6)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_, y_, z_, c='red', marker='o', s=100)

    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")

    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    plt.show()

# def PlotQuadrature2d(x_,y_,z_):
#     # plt.rcParams["figure.figsize"] = [7.50, 7.50]
#     plt.rcParams["figure.autolayout"] = True

#     fig = plt.figure()
#     fig.set_size_inches(6, 6)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x_, y_, z_, c='red', marker='o', s=100)

#     ax.set_xlabel("x axis")
#     ax.set_ylabel("y axis")
#     ax.set_zlabel("z axis")

#     ax.set_xlim(-1,1)
#     ax.set_ylim(-1,1)
#     ax.set_zlim(-1,1)

#     plt.show()

# PLOT LINE
# import numpy as np
# from matplotlib import pyplot as plt
# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
# x = np.linspace(-4 * np.pi, 4 * np.pi, 50)
# y = np.linspace(-4 * np.pi, 4 * np.pi, 50)
# z = x ** 2 + y ** 2
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z)
# plt.show()
    
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    
# https://jakevdp.github.io/PythonDataScienceHandbook/