###############################################################################
# File  : notes/mc/pymc/uncf.py
# Author: Thomas Evans
# Date  : Thu Aug 04 15:55:34 2016
###############################################################################
from __future__ import (division, absolute_import, print_function, )
#-----------------------------------------------------------------------------#
import math
import itertools
import numpy as np

from . import integrator
###############################################################################

I = 2
J = 1
K = 0

##---------------------------------------------------------------------------##
## Mesh
##---------------------------------------------------------------------------##

class Mesh(object):

    def __init__(self, shape = (1,1,1), delta = (1,1,1), sigma = 1.0):
        self._grid = np.array([
            np.arange(0.0, shape[K]*delta[K]+1e-12, delta[K]),
            np.arange(0.0, shape[J]*delta[J]+1e-12, delta[J]),
            np.arange(0.0, shape[I]*delta[I]+1e-12, delta[I])
        ])

        self._N   = (self.z.shape[0] - 1,
                     self.y.shape[0] - 1,
                     self.x.shape[0] - 1)
        self._V   = np.product(delta)
        self._sig = sigma

    def num_cells_dim(self, dim):
        return self._N[dim]

    def bounding_box(self, i, j, k):

        low  = np.array([self._grid[I][i],
                         self._grid[J][j],
                         self._grid[K][k]])

        high = np.array([self._grid[I][i+1],
                         self._grid[J][j+1],
                         self._grid[K][k+1]])

        return low, high

    @property
    def shape(self):
        return self._N

    @property
    def sigma(self):
        return self._sig

    @property
    def num_cells(self):
        return np.product(self._N)

    @property
    def x(self):
        return self._grid[I]

    @property
    def y(self):
        return self._grid[J]

    @property
    def z(self):
        return self._grid[K]

    @property
    def volume(self):
        return self._V

##---------------------------------------------------------------------------##
## Source
##---------------------------------------------------------------------------##

class Source(object):

    def __init__(self, point = (0.0, 0.0, 0.0, ), strength = 1.0):

        self._P   = np.array(point)
        self._str = strength

    @property
    def point(self):
        return self._P

    @property
    def strength(self):
        return self._str

##---------------------------------------------------------------------------##
## Ray-Tracer
##---------------------------------------------------------------------------##

class Tracer(object):

    def __init__(self, source, mesh):

        self._mesh   = mesh
        self._source = source

        self._inv_4pi = 0.25 / np.pi

    def trace(self, target):

        r   = np.linalg.norm(target - self._source.point)
        phi = math.exp(-r * self._mesh.sigma) * self._inv_4pi / r**2 \
              * self._source.strength

        return phi

##---------------------------------------------------------------------------##
## Uncollided-flux
##---------------------------------------------------------------------------##

class UNCF(object):

    def __init__(self, source, mesh, P = 2):

        self._source = source
        self._mesh   = mesh

        self._integrator = integrator.Integrator(P)
        self._tracer     = Tracer(self._source, self._mesh)

        self._phi = np.zeros(self._mesh.shape)

    def calc_flux(self):

        # inverse volume
        inv_V = 1.0 / self._mesh.volume

        # Loop over cells
        for k,j,i in itertools.product(xrange(self._mesh.num_cells_dim(K)),
                                       xrange(self._mesh.num_cells_dim(J)),
                                       xrange(self._mesh.num_cells_dim(I))):

            # Get the bounding box for the cell
            low, high = self._mesh.bounding_box(i,j,k)

            # Set the integrator
            self._integrator.set_cell(low, high)

            # Get the quadrature points
            points = self._integrator.quad_points()

            # Get the flux at each point
            flux = np.apply_along_axis(self._tracer.trace, 0, points)

            # Integrate the flux
            self._phi[k,j,i] = self._integrator.integrate(flux)

        # Normalize by volume
        self._phi *= inv_V

    @property
    def flux(self):
        return self._phi

#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    main()

###############################################################################
# end of notes/mc/pymc/uncf.py
###############################################################################
