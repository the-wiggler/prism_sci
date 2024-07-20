###############################################################################
# File  : notes/mc/pymc/integrator.py
# Author: Tom Evans
# Date  : Fri Jul 29 15:48:29 2016
#
# Finite element integration class
###############################################################################
from __future__ import (division, absolute_import, print_function, )
#-----------------------------------------------------------------------------#
import math
import numpy as np
###############################################################################

class Integrator(object):

    _X = 0
    _Y = 1
    _Z = 2

    def __init__(self, P = 3):
        self._P = P**3

    def set_cell(self, low, high):

        xlo = low[self._X]
        xhi = high[self._X]

        ylo = low[self._Y]
        yhi = high[self._Y]

        zlo = low[self._Z]
        zhi = high[self._Z]

        self._nodes = np.array([[xlo, xhi, xhi, xlo, xlo, xhi, xhi, xlo],
                                [ylo, ylo, yhi, yhi, ylo, ylo, yhi, yhi],
                                [zlo, zlo, zlo, zlo, zhi, zhi, zhi, zhi]])

        if (self._P == 8):

            q = math.sqrt(1.0 / 3.0)

            self._xi   = np.array([-q, q, -q, q, -q, q, -q, q])
            self._eta  = np.array([-q, -q, q, q, -q, -q, q, q])
            self._zeta = np.array([-q, -q, -q, -q, q, q, q, q])
            self._w    = np.ones(8)

        elif (self._P == 27):

            q1 = 0.0
            w1 = 8.0 / 9.0
            q2 = math.sqrt(3.0/5.0)
            w2 = 5.0 / 9.0

            self._xi   = np.array([-q2, q1, q2,
                                   -q2, q1, q2,
                                   -q2, q1, q2,
                                   -q2, q1, q2,
                                   -q2, q1, q2,
                                   -q2, q1, q2,
                                   -q2, q1, q2,
                                   -q2, q1, q2,
                                   -q2, q1, q2])
            self._eta  = np.array([-q2, -q2, -q2,
                                   q1, q1, q1,
                                   q2, q2, q2,
                                   -q2, -q2, -q2,
                                   q1, q1, q1,
                                   q2, q2, q2,
                                   -q2, -q2, -q2,
                                   q1, q1, q1,
                                   q2, q2, q2])
            self._zeta = np.array([-q2, -q2, -q2,
                                   -q2, -q2, -q2,
                                   -q2, -q2, -q2,
                                   q1, q1, q1,
                                   q1, q1, q1,
                                   q1, q1, q1,
                                   q2, q2, q2,
                                   q2, q2, q2,
                                   q2, q2, q2])
            self._w    = np.array([w2*w2*w2, w1*w2*w2, w2*w2*w2,
                                   w2*w1*w2, w1*w1*w2, w2*w1*w2,
                                   w2*w2*w2, w1*w2*w2, w2*w2*w2,
                                   w2*w2*w1, w1*w2*w1, w2*w2*w1,
                                   w2*w1*w1, w1*w1*w1, w2*w1*w1,
                                   w2*w2*w1, w1*w2*w1, w2*w2*w1,
                                   w2*w2*w2, w1*w2*w2, w2*w2*w2,
                                   w2*w1*w2, w1*w1*w2, w2*w1*w2,
                                   w2*w2*w2, w1*w2*w2, w2*w2*w2])

        elif (self._P == 64):

            q1 = 0.861136311594953
            w1 = 0.347854845137454
            q2 = 0.339981043584856
            w2 = 0.652145154862546

            self._xi = np.array([-q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1,
                                 -q1, -q2, q2, q1])

            self._eta = np.array([-q1, -q1, -q1, -q1,
                                  -q2, -q2, -q2, -q2,
                                  q2, q2, q2, q2,
                                  q1, q1, q1, q1,
                                  -q1, -q1, -q1, -q1,
                                  -q2, -q2, -q2, -q2,
                                  q2, q2, q2, q2,
                                  q1, q1, q1, q1,
                                  -q1, -q1, -q1, -q1,
                                  -q2, -q2, -q2, -q2,
                                  q2, q2, q2, q2,
                                  q1, q1, q1, q1,
                                  -q1, -q1, -q1, -q1,
                                  -q2, -q2, -q2, -q2,
                                  q2, q2, q2, q2,
                                  q1, q1, q1, q1])

            self._zeta = np.array([-q1, -q1, -q1, -q1,
                                   -q1, -q1, -q1, -q1,
                                   -q1, -q1, -q1, -q1,
                                   -q1, -q1, -q1, -q1,
                                   -q2, -q2, -q2, -q2,
                                   -q2, -q2, -q2, -q2,
                                   -q2, -q2, -q2, -q2,
                                   -q2, -q2, -q2, -q2,
                                   q2, q2, q2, q2,
                                   q2, q2, q2, q2,
                                   q2, q2, q2, q2,
                                   q2, q2, q2, q2,
                                   q1, q1, q1, q1,
                                   q1, q1, q1, q1,
                                   q1, q1, q1, q1,
                                   q1, q1, q1, q1])

            self._w = np.array([w1*w1*w1, w2*w1*w1, w2*w1*w1, w1*w1*w1,
                                w1*w2*w1, w2*w2*w1, w2*w2*w1, w1*w2*w1,
                                w1*w2*w1, w2*w2*w1, w2*w2*w1, w1*w2*w1,
                                w1*w1*w1, w2*w1*w1, w2*w1*w1, w1*w1*w1,
                                w1*w1*w2, w2*w1*w2, w2*w1*w2, w1*w1*w2,
                                w1*w2*w2, w2*w2*w2, w2*w2*w2, w1*w2*w2,
                                w1*w2*w2, w2*w2*w2, w2*w2*w2, w1*w2*w2,
                                w1*w1*w2, w2*w1*w2, w2*w1*w2, w1*w1*w2,
                                w1*w1*w2, w2*w1*w2, w2*w1*w2, w1*w1*w2,
                                w1*w2*w2, w2*w2*w2, w2*w2*w2, w1*w2*w2,
                                w1*w2*w2, w2*w2*w2, w2*w2*w2, w1*w2*w2,
                                w1*w1*w2, w2*w1*w2, w2*w1*w2, w1*w1*w2,
                                w1*w1*w1, w2*w1*w1, w2*w1*w1, w1*w1*w1,
                                w1*w2*w1, w2*w2*w1, w2*w2*w1, w1*w2*w1,
                                w1*w2*w1, w2*w2*w1, w2*w2*w1, w1*w2*w1,
                                w1*w1*w1, w2*w1*w1, w2*w1*w1, w1*w1*w1])

    def basis(self, xi, eta, zeta):
        N    = np.zeros(8)
        N[0] = 0.125 * (1-xi) * (1-eta) * (1-zeta)
        N[1] = 0.125 * (1+xi) * (1-eta) * (1-zeta)
        N[2] = 0.125 * (1+xi) * (1+eta) * (1-zeta)
        N[3] = 0.125 * (1-xi) * (1+eta) * (1-zeta)
        N[4] = 0.125 * (1-xi) * (1-eta) * (1+zeta)
        N[5] = 0.125 * (1+xi) * (1-eta) * (1+zeta)
        N[6] = 0.125 * (1+xi) * (1+eta) * (1+zeta)
        N[7] = 0.125 * (1-xi) * (1+eta) * (1+zeta)
        return N

    def der_basis(self, xi, eta, zeta):
        X = self._X
        Y = self._Y
        Z = self._Z

        N      = np.zeros((3, 8))
        N[X,0] = -0.125 * (1-eta) * (1-zeta)
        N[X,1] =  0.125 * (1-eta) * (1-zeta)
        N[X,2] =  0.125 * (1+eta) * (1-zeta)
        N[X,3] = -0.125 * (1+eta) * (1-zeta)
        N[X,4] = -0.125 * (1-eta) * (1+zeta)
        N[X,5] =  0.125 * (1-eta) * (1+zeta)
        N[X,6] =  0.125 * (1+eta) * (1+zeta)
        N[X,7] = -0.125 * (1+eta) * (1+zeta)

        N[Y,0] = -0.125 * (1-xi) * (1-zeta)
        N[Y,1] = -0.125 * (1+xi) * (1-zeta)
        N[Y,2] =  0.125 * (1+xi) * (1-zeta)
        N[Y,3] =  0.125 * (1-xi) * (1-zeta)
        N[Y,4] = -0.125 * (1-xi) * (1+zeta)
        N[Y,5] = -0.125 * (1+xi) * (1+zeta)
        N[Y,6] =  0.125 * (1+xi) * (1+zeta)
        N[Y,7] =  0.125 * (1-xi) * (1+zeta)

        N[Z,0] = -0.125 * (1-xi) * (1-eta)
        N[Z,1] = -0.125 * (1+xi) * (1-eta)
        N[Z,2] = -0.125 * (1+xi) * (1+eta)
        N[Z,3] = -0.125 * (1-xi) * (1+eta)
        N[Z,4] =  0.125 * (1-xi) * (1-eta)
        N[Z,5] =  0.125 * (1+xi) * (1-eta)
        N[Z,6] =  0.125 * (1+xi) * (1+eta)
        N[Z,7] =  0.125 * (1-xi) * (1+eta)

        return N

    def jacobian(self, xi, eta, zeta):

        J = np.zeros((3,3))
        D = self.der_basis(xi, eta, zeta)

        for i in xrange(3):
            for j in xrange(3):
                J[i,j] = np.sum(D[i,:] * self._nodes[j,:])

        return J

    def invert(self, J):
        invJ = np.linalg.inv(J)
        return invJ

    def det(self, J):
        return np.linalg.det(J)

    def point(self, pt):

        xi   = pt[0]
        eta  = pt[1]
        zeta = pt[2]

        x = np.zeros(3)
        v = self._nodes
        b = self.basis(xi, eta, zeta)

        for n in xrange(3):
            x[n] = np.sum(b[:] * v[n,:])

        return x

    def quad_point(self, n):
        return self.point((self._xi[n], self._eta[n], self._zeta[n]))

    def quad_points(self):

        pts = np.zeros((self._P, 3))

        for n in xrange(self._P):
            pts[n,:] = self.quad_point(n)

        return np.transpose(pts)

    def integrate(self, f):

        result = 0.0
        for p in xrange(self._P):
            J       = self.jacobian(self._xi[p], self._eta[p], self._zeta[p])
            detJ    = self.det(J)
            result += f[p] * detJ * self._w[p]

        return result

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Z(self):
        return self._Z

    @property
    def P(self):
        return self._P

#-----------------------------------------------------------------------------#

if __name__ == '__main__':

    i1 = Integrator(P = 2)
    i1.set_cell((-1, 0, 1.5), (3, 2.5, 2))

    for p in xrange(i1.P):
        print(i1.quad_point(p))

    print()

    i2 = Integrator(P = 2)
    i2.set_cell((-1, -1, -1), (1, 1, 1))

    for p in xrange(i2.P):
        print(i2.quad_point(p))

    print()

    print(i2.quad_points())

    print()

    # Integrate (x^2 - 1)(y^2 - 1)(z^2 - 1)

    i3   = Integrator(P = 2)
    i3.set_cell((1, -1, 0), (3, -0.5, 0.75))
    pts  = i3.quad_points()
    f    = np.zeros(i3.P)
    f[:] = (pts[0,:]**2 - 1.0)*(pts[1,:]**2 - 1.0)*(pts[2,:]**2 - 1.0)

    print(i3.integrate(f))

    print()

    # Integrate (x^2 - 1)(y^2 - 1)(z^2 - 1)

    i4   = Integrator(P = 3)
    i4.set_cell((1, -1, 0), (3, -0.5, 0.75))
    pts  = i4.quad_points()
    f    = np.zeros(i4.P)
    f[:] = (pts[0,:]**2 - 1.0)*(pts[1,:]**2 - 1.0)*(pts[2,:]**2 - 1.0)

    print(i4.integrate(f))

    print()

    # Integrate (x^2 - 1)(y^2 - 1)(z^2 - 1)

    i5   = Integrator(P = 4)
    i5.set_cell((1, -1, 0), (3, -0.5, 0.75))
    pts  = i5.quad_points()
    f    = np.zeros(i5.P)
    f[:] = (pts[0,:]**2 - 1.0)*(pts[1,:]**2 - 1.0)*(pts[2,:]**2 - 1.0)

    print(i5.integrate(f))


###############################################################################
# end of notes/mc/pymc/integrator.py
###############################################################################
