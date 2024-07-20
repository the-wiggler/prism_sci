###############################################################################
# File  : notes/mc/pymc/raytracer.py
# Author: Tom Evans
# Date  : Wed Aug 30 22:33:15 2017
###############################################################################
import numpy as np
import math

from . import uncf
###############################################################################

class Tally:
    """MC Tally class

    Instance Attributes
    -------------------
    - mesh
    - mean
    - var
    """

    def __init__(self, mesh, source):

        self._mesh = mesh

        # History statistics
        self._mean = np.zeros(mesh.shape)
        self._var  = np.zeros(mesh.shape)

        # Batch statistics
        self._batch = np.zeros(mesh.shape)
        self._batch_mean = np.zeros(mesh.shape)
        self._batch_var = np.zeros(mesh.shape)

        # Number of batches
        self._M = 0

        # Total number of particles (across all batches)
        self._N = 0

        # Balance variables
        self._K = source.strength
        self._L = 0.0

    def accumulate(self, kji, l):

        self._mean[tuple(kji)] += l
        self._var[tuple(kji)]  += (l*l)

        self._batch[tuple(kji)] += l

    def end_batch(self, Np):

        self._batch_mean += self._batch
        self._batch_var += self._batch * self._batch / Np
        self._batch[:,:,:] = 0.0

        self._M += 1

    def finalize(self):

        norm = self._K / self._mesh.volume

        # History statistics
        a = self._mean
        b = self._var

        a *= norm
        b *= norm*norm

        self._var   = 1 / (self._N*self._N * (self._N - 1)) * \
                      (self._N * b - a * a)
        self._mean *= 1 / self._N
        self._L    *= 1 / self._N

        # Batch statistics
        if self._M > 1:
            a = self.batch_mean
            b = self._batch_var

            a *= norm / self._N
            b *= norm*norm / self._N

            self._batch_var = 1.0 / (self._M - 1.0) * (b - a*a)

    def relative_error(self):
        return np.sqrt(self._var) / self._mean

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._var

    @property
    def batch_mean(self):
        return self._batch_mean

    @property
    def batch_var(self):
        return self._batch_var

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value

##---------------------------------------------------------------------------##

class Raytracer:
    """ MC Raytracing class


    Instance Attributes
    -------------------
    - mesh
    - sigma
    """

    # Class attributes
    _X = 0
    _Y = 1
    _Z = 2

    def __init__(self, mesh):

        self._mesh  = mesh
        self._sigma = mesh.sigma

    def trace(self, source, rays, tally):

        # set the point from the source
        point = source.point

        # find the indices of the point
        k,j,i = (np.searchsorted(self._mesh.z, point[2]) - 1,
                 np.searchsorted(self._mesh.y, point[1]) - 1,
                 np.searchsorted(self._mesh.x, point[0]) - 1)

        # steering vector
        sv = np.ones(3, dtype='int')

        # Update N
        tally.N += rays.shape[0]

        # track to boundaries
        for ray in rays:

            # Direction vectors
            dv = (ray / np.abs(ray)).astype('int')
            sv[np.where(dv < 0)] = 0

            srf = np.array([self._mesh.z[k+sv[self.Z]::dv[self.Z]],
                            self._mesh.y[j+sv[self.Y]::dv[self.Y]],
                            self._mesh.x[i+sv[self.X]::dv[self.X]]])

            dbx = (srf[uncf.I] - point[self.X]) / ray[self.X]
            dby = (srf[uncf.J] - point[self.Y]) / ray[self.Y]
            dbz = (srf[uncf.K] - point[self.Z]) / ray[self.Z]

            stx = np.full(dbx.shape, uncf.I, dtype='int')
            sty = np.full(dby.shape, uncf.J, dtype='int')
            stz = np.full(dbz.shape, uncf.K, dtype='int')

            # Determmine number of steps
            mind = np.min([dbx[-1], dby[-1], dbz[-1]])

            # Cell indices
            cell = [k,j,i]
            dirv = np.flip(dv, 0)

            # Sort
            st  = np.append(stx, np.append(sty, stz))
            db  = np.append(dbx, np.append(dby, dbz))
            idx = np.argsort(db)

            seg = np.append(np.asarray([db[idx[0]]]), np.diff(db[idx]))

            # Initialize values for transport loop
            w      = 1.0
            dist   = 0.0
            step   = 0
            inside = True

            while (inside):

                # Current step distance
                d = seg[step]

                # Exponent of tally
                tau      = self._sigma * d
                exponent = math.exp(-tau)

                # Tally the result
                tally.accumulate(cell, w * d / tau * (1.0 - exponent))

                # Update the weight
                w *= exponent

                # Update total distance traveled and new cell
                dist       += d
                surf        = st[idx[step]]
                cell[surf] += dirv[surf]

                if cell[surf] < 0 or cell[surf] > self._mesh.shape[surf] - 1:
                    inside = False

                step += 1

            # Add exiting weight to leakage
            tally.L += w

            # reset steering
            sv[:] = 1

    @property
    def mesh(self):
        return self._mesh

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Z(self):
        return self._Z

###############################################################################
# end of notes/mc/pymc/raytracer.py
###############################################################################
