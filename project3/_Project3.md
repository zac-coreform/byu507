PROJECT 3

Complete the following two-dimensional finite element analysis code for diffusion/steady-state heat conduction. You will need to incorporate functionality here that was created in Homeworks 5, 6, and 7.

For your final submission, please send your completed code, in addition to all files necessary to run the code. Then, please run the code on the brick.obj file using thermal conductivity defined by the BrickKappa function. 
- Take the bottom boundary condition to be Dirichlet and g=10 (degrees celsius), 
- the top to be Neumann with flux h=0.3, 
- the left and right Neumann with h=0 (i.e. insulated or periodic). 
- Choose f=0 (no internal heat generation). Plot a surface contour plot.

Finally, describe what happens (using text and contour plots) when the thermal conductivity of the mortar (at the two "holes" in the mesh) is:
- significantly lower than that of the clay for the brick, 
- approximately equal to that of the brick, and 
- much greater than that of the brick.




