import numpy as np
from matplotlib import pyplot as plt



# Complete the following code to define a 2D bilinear basis function (Eq. 3.2.14). 
# Plot your results for each of the four basis functions (as given in the python functionality) to visualize the basis functions. 
# Also complete code that will take a list of 4 numpy arrays (representing x and y coordinates of corners of a quadrilateral) and evaluate the mapping from parametric domain into this quadrilateral. 
# Again, plot your results to validate that it works. Submit your completed code.



def NBasis(a,ksi,eta):
    ksi_vals = [-1,1,1,-1]
    eta_vals = [-1,-1,1,1]
    return (0.5 * (1 + ksi_vals[a] * ksi)) * (0.5 * (1 + eta_vals[a] * eta))


# N = np.zeros((3,n_x,n_y))
    # 3 arrays of 100x101 between -1 and 1
# for i in range(0,len(xvals)):
#     for j in range(0,len(yvals)):
        # temp = XMap(x_pts_arrays,xvals[i],yvals[j])
                                    #. ^100     ^101

# N[0,i,j] = temp[0]       #in ksi direction
# N[1,i,j] = temp[1]       #in eta direction
# N[2,i,j] = temp[2] if XMap is outputting len 3, otherwise, zeros

def XMap(x_pts,ksi,eta):  #evaluate the transformation mapping
    # input:
        # x_pts = [array([2, 1]), array([ 5, -1]), array([8, 0]), array([4, 4])]
        # ksi = one of 100 points between -1 and 1
        # eta = one of 101 points between -1 and 1
        xy = [0,0]
        for i in range(0,2):
            for j in range(0,2):

            NBasis(a,ksi,eta) * xvals[j]

    # output is vector [x(ksi), y(eta)]
    xvals = [x0,x1]
    for a in range(0,2):
        x += NBasis(a,xi,eta) * xvals[a]
    return x
    
    xvals = np.linspace(x0,x1,p+1)
    for j in range(0,len(xvals)):
        x += NBasis(deg=p, N_idx=j, t=xi) * xvals[j]
    return x




# input four points as a list of lists
# e.g. xpts = [[2,1],[5,-1],([8,0]),([4,4])]  
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