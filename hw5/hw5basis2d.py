import numpy as np
from matplotlib import pyplot as plt



# Complete the following code to define a 2D bilinear basis function (Eq. 3.2.14). 

# Plot your results for each of the four basis functions (as given in the python functionality) to visualize the basis functions. 

# Also complete code that will 
#   take a list of 4 numpy arrays (representing x and y coordinates of corners of a quadrilateral) and 
#   evaluate the mapping from parametric domain into this quadrilateral. 

# Again, plot your results to validate that it works. Submit your completed code.


def NBasis(a,xi,eta):
    xi_vals = [-1,1,1,-1]
    eta_vals = [-1,-1,1,1]
    return (0.5 * (1 + xi_vals[a] * xi)) * (0.5 * (1 + eta_vals[a] * eta))


# before: for xi in, return x
def XMap(x_pts,xi,eta):
    pts_arrays = []
    for pt in x_pts:
        pts_arrays.append(np.array(pt))
    xy_out = np.zeros((2,1))
    for i in range(0, 4):
        nbi = NBasis(i, xi, eta)
        # print("nbi =", nbi)
        pti = pts_arrays[i]
        # print("pti =", pti)
        prod = nbi * pti
        # print("prod =", prod)
        xy_out[0] += prod[0]
        xy_out[1] += prod[1]
    return xy_out


def PlotTransformationMap(x_pts):
    n_x = 100
    n_y = 101     
    
    x_pts_arrays = []
    # put each point as an array into an array of arrays
    for pt in x_pts:
        x_pts_arrays.append(np.array(pt))
    
    xvals = np.linspace(-1,1,n_x) 
    yvals = np.linspace(-1,1,n_y)
    
    X, Y = np.meshgrid(xvals, yvals)
    
    N = np.zeros((3,n_x,n_y))
    # 3 arrays of 100x101
    for i in range(0,len(xvals)):
      for j in range(0,len(yvals)):
        temp = XMap(x_pts_arrays,xvals[i],yvals[j])
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
def PlotBasisFunction(a):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-1.0, 1.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array(NBasis(a,np.ravel(X), np.ravel(Y)))
    Z = zs.reshape(X.shape)
    
    ax.plot_surface(X, Y, Z)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()