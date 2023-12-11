import hw7Basis_Functions as bf
import hw7Poisson_1D as p1

def ErrorValues(fullD,xvals,p,quadrature,exact_u,exact_u_derv):
    l2_error_vec = []
    h1_error_vec = []
    n_elem = len(xvals)-1
    quad_wts = quadrature.quad_wts
    quad_pts = quadrature.quad_pts
    n_quad = quadrature.n_quad
    IEN = p1.CreateIENArray(p, n_elem)
    for e in range(0,n_elem):
        l2error = 0
        h1error = 0
        x0 = xvals[e]
        x1 = xvals[e+1]
        for q in range(0,n_quad):
            uh = 0
            uhx = 0
            ksi_q = quad_pts[q]
            # XMap(x0,x1,xi,p)
            x_q = bf.XMap(x0, x1, ksi_q, p)
            w_q = quad_wts[q]
            # XMapDerv(x0,x1,xi,p)
            x_inv = (bf.XMapDerv(x0, x1, ksi_q, p))**-1
            for a in range(0,p+1):
                A = IEN[a,e]
                # NBasis(deg, N_idx, t)
                uh += fullD[A,0] * bf.NBasis(p, a, ksi_q)
                uhx += fullD[A,0] * bf.NBasisDerv(p, a, ksi_q) * x_inv
        l2error += w_q * (uh - exact_u(x_q))**2 * (x1-x0)/2
        h1error += w_q * ((uh - exact_u(x_q))**2 + (uhx - exact_u_derv(x_q))**2) *(x1-x0)/2
    l2_error_vec.append(l2error)
    h1_error_vec.append(h1error)
    total_l2_error = (sum(l2_error_vec))**0.5
    total_h1_error = (sum(h1_error_vec))**0.5
    return [total_l2_error,total_h1_error]

# Plot logh compared to logei, with e0 being the error in L2 and e1 being the error H1 (i.e. the energy norm). What is the slope of these plots for p = 1, 2, 3, 4, 5, particularly between refinements with 32 and 64 elements?