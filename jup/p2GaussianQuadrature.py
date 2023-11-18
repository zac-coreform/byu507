import numpy.polynomial.legendre as lg

class GaussianQuadrature():
    def __init__(self, n_quad):
        self.n = n_quad
        self.pts = lg.leggauss(n_quad)[0]
        self.wts = lg.leggauss(n_quad)[1]

# usage:
# q = Quadrature(3)
# q.pts ->> array([-0.77459667,  0.        ,  0.77459667])
# q.wts ->> array([0.55555556, 0.88888889, 0.55555556])