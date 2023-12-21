import math

# Bernstein polynomials
def b10(x): return 1 - x
def b11(x): return x

def b20(x): return (1 - x)**2
def b21(x): return 2*(1 - x)*x
def b22(x): return x**2

def b30(x): return (1-x)**3
def b31(x): return 3*x * (1-x)**2
def b32(x): return x**2 * (3 - 3*x)
def b33(x): return x**3

def b40(x): return (1-x)**4
def b41(x): return 4*x * (1-x)**3
def b42(x): return 6*x**2 * (1 - x)**2
def b43(x): return x**3 * (4 - 4*x)
def b44(x): return x**4

def b50(x): return (1-x)**5
def b51(x): return 5*x * (1-x)**4
def b52(x): return 10*x**2 * (1 - x)**3
def b53(x): return 10*x**3 * (1 - x)**2
def b54(x): return x**4 * (5 - 5*x)
def b55(x): return x**5

b1 = [b10, b11]
b2 = [b20, b21, b22]
b3 = [b30, b31, b32, b33]
b4 = [b40, b41, b42, b43, b44]
b5 = [b50, b51, b52, b53, b54, b55]

b_p = [b1, b2, b3, b4, b5]

def bern_p_testvals(p, a):
    p_idx = p - 1
    unit_pts = [0., 0.25, 0.67, 1.]
    fn = b_p[p_idx][a]
    vals = []
    for i in range(0, len(unit_pts)):
        pt = unit_pts[i]
        vals.append(fn(pt))
    return vals

bern_p1_a0_tvals = bern_p_testvals(1,0)
bern_p1_a1_tvals = bern_p_testvals(1,1)

b1_tvals_list = [bern_p1_a0_tvals, bern_p1_a1_tvals]

bern_p2_a0_tvals = bern_p_testvals(2,0)
bern_p2_a1_tvals = bern_p_testvals(2,1)
bern_p2_a2_tvals = bern_p_testvals(2,2)

b2_tvals_list = [bern_p2_a0_tvals, bern_p2_a1_tvals, bern_p2_a2_tvals]

bern_p3_a0_tvals = bern_p_testvals(3,0)
bern_p3_a1_tvals = bern_p_testvals(3,1)
bern_p3_a2_tvals = bern_p_testvals(3,2)
bern_p3_a3_tvals = bern_p_testvals(3,3)

b3_tvals_list = [bern_p3_a0_tvals, bern_p3_a1_tvals, bern_p3_a2_tvals, bern_p3_a3_tvals]

bern_p4_a0_tvals = bern_p_testvals(4,0)
bern_p4_a1_tvals = bern_p_testvals(4,1)
bern_p4_a2_tvals = bern_p_testvals(4,2)
bern_p4_a3_tvals = bern_p_testvals(4,3)
bern_p4_a4_tvals = bern_p_testvals(4,4)

b4_tvals_list = [bern_p4_a0_tvals, bern_p4_a1_tvals, bern_p4_a2_tvals, bern_p4_a3_tvals, bern_p4_a4_tvals]

bern_p5_a0_tvals = bern_p_testvals(5,0)
bern_p5_a1_tvals = bern_p_testvals(5,1)
bern_p5_a2_tvals = bern_p_testvals(5,2)
bern_p5_a3_tvals = bern_p_testvals(5,3)
bern_p5_a4_tvals = bern_p_testvals(5,4)
bern_p5_a5_tvals = bern_p_testvals(5,5)

b5_tvals_list = [bern_p5_a0_tvals, bern_p5_a1_tvals, bern_p5_a2_tvals, bern_p5_a3_tvals, bern_p5_a4_tvals, bern_p5_a5_tvals]






# for test values of bern derivs, wfa calcd deriv exprns

def b20derv(x): return 2*x - 2
def b21derv(x): return 2 - 4*x
def b22derv(x): return 2*x
b2d = [b20derv, b21derv, b22derv]

def bderv30(x): return -3 * (1 - x)**2
def bderv31(x): return 9*x**2 - 12*x + 3
def bderv32(x): return 6*x - 9*x**2
def bderv33(x): return 3*x**2
b3d = [bderv30, bderv31, bderv32, bderv33]

def bderv40(x): return -4 * (1-x)**3
def bderv41(x): return 4 - 24*x + 36*x**2 - 16*x**3
def bderv42(x): return 24*x**3 - 36*x**2 + 12*x
def bderv43(x): return 12*x**2 - 16*x**3
def bderv44(x): return 4*x**3
b4d = [bderv40, bderv41, bderv42, bderv43, bderv44]

bern_dervs = [b2d, b3d, b4d]

def bern2d_testvals(a):
    unit_pts = [0., 0.25, 0.67, 1.]
    fn = b2d[a]
    vals = []
    for i in range(0, len(unit_pts)):
        pt = unit_pts[i]
        vals.append(fn(pt))
    return vals

bern_p2_a0_d_tvals = bern2d_testvals(0)
bern_p2_a1_d_tvals = bern2d_testvals(1)
bern_p2_a2_d_tvals = bern2d_testvals(2)

b2d_tvals_list = [bern_p2_a0_d_tvals, bern_p2_a1_d_tvals, bern_p2_a2_d_tvals]

def bern3d_testvals(a):
    unit_pts = [0., 0.25, 0.67, 1.]
    fn = b3d[a]
    vals = []
    for i in range(0, len(unit_pts)):
        pt = unit_pts[i]
        vals.append(fn(pt))
    return vals

bern_p3_a0_d_tvals = bern3d_testvals(0)
bern_p3_a1_d_tvals = bern3d_testvals(1)
bern_p3_a2_d_tvals = bern3d_testvals(2)
bern_p3_a3_d_tvals = bern3d_testvals(3)

b3d_tvals_list = [bern_p3_a0_d_tvals, bern_p3_a1_d_tvals, bern_p3_a2_d_tvals, bern_p3_a3_d_tvals]

def bern4d_testvals(a):
    unit_pts = [0., 0.25, 0.67, 1.]
    fn = b4d[a]
    vals = []
    for i in range(0, len(unit_pts)):
        pt = unit_pts[i]
        vals.append(fn(pt))
    return vals

bern_p4_a0_d_tvals = bern4d_testvals(0)
bern_p4_a1_d_tvals = bern4d_testvals(1)
bern_p4_a2_d_tvals = bern4d_testvals(2)
bern_p4_a3_d_tvals = bern4d_testvals(3)
bern_p4_a4_d_tvals = bern4d_testvals(4)

b4d_tvals_list = [bern_p4_a0_d_tvals, bern_p4_a1_d_tvals, bern_p4_a2_d_tvals, bern_p4_a3_d_tvals, bern_p4_a4_d_tvals]



########################################
########################################



def x_of_xi(xi, x0, x1, xi0=0, xi1=1):
    x = ((xi - xi0)*(x1 - x0) / (xi1 - xi0)) + x0
    return x
def xi_of_x(x, x0, x1, xi0=0, xi1=1):
    xi = ((x - x0)*(xi1 - xi0) / (x1 - x0)) + xi0
    return xi

dom_u2u = [0,1]
dom_ugu = [0.25, 0.5]
dom_arb = [3., 7.]
doms = [dom_u2u, dom_ugu, dom_arb]

unit_pts = [0., 0.25, 0.67, 1.]

def xmap_p_dom_testvals(dom, pts=unit_pts):
    x0 = dom[0]
    x1 = dom[1]
    loop = len(pts)
    vals = []
    for i in range(0, loop):
        xi = unit_pts[i]
        x = x_of_xi(xi, x0, x1)
        vals.append(x)
    return vals

u2u_ = xmap_p_dom_testvals(dom_u2u)
ugu_ = xmap_p_dom_testvals(dom_ugu)
arb_ = xmap_p_dom_testvals(dom_arb)

# get values for XMapDerv
# equation for xmap is:
# ((xi - xi0)*(x1 - x0) / (xi1 - xi0)) + x0
# run through wfa to reduce in terms of x0=A, x1=B, hardcoding xi0,xi1 as 0,1
# ((y * (B - A))/(1 - 0)) + A --> derv = B - A
# so
def x_of_xi_derv(xi, x0, x1, xi0=0, xi1=1):
    # x = ((xi - xi0)*(x1 - x0) / (xi1 - xi0)) + x0
    B = x1
    A = x0
    # print(f"A={A}, B={B}")
    ddxi = B - A
    # print(ddxi)
    return ddxi

def xmap_derv_pdom_testvals(dom,pts=unit_pts):
    x0 = dom[0]
    x1 = dom[1]
    loop = len(pts)
    dx_out = 0
    ref_dx = x_of_xi_derv(pts[0], x0, x1)
    for i in range(1, loop):
        xi = pts[i]
        dx = x_of_xi_derv(xi, x0, x1)
        if math.isclose(dx, ref_dx):
            dx_out = dx
        else:
            dx_out = 99
    return dx_out

