###############################################################################
# File  : notes/mc/pymc/mcuncf.py
# Author: Tom Evans
# Date  : Sun Oct 01 23:50:24 2017
###############################################################################
import numpy as np
import pandas as pd
import math

from .raytracer import (Tally, Raytracer)
###############################################################################

class MCUNCF:
    """MC uncollided flux solver

    Instance Attributes
    -------------------
    - source
    - mesh
    - rt (raytracer)
    - tally
    """

    # Number of particle batches
    _Nb = 1

    # Number of particles per batch
    _Np = 1000

    # Construction
    def __init__(self, source, mesh):

        self._mesh   = mesh
        self._source = source

        # Make the tally
        self._tally = Tally(mesh, source)

        # Make the raytracer
        self._rt = Raytracer(mesh)

    def calc_flux(self):

        # Loop over batches
        for b in range(self._Nb):

            print(">>> Running batch {:5d} . . .".format(b), end='')

            # Set random numbers for this batch
            rngs = np.random.rand(self._Np, 2)

            # Build the rays for this batch
            rays = np.apply_along_axis(self._make_rays, 1, rngs)

            # Raytrace all of the particles for this batch
            self._rt.trace(self._source, rays, self._tally)

            # Do batch statistics
            self._tally.end_batch(self._Np)

            print(" {:10d}".format(self._tally.N))

        # Finalize all of the tallies
        self._tally.finalize()

        # Calculate terms in balance table
        self._balance = {}
        self._balance['L'] = self._tally.L

        # Calculate volume removal
        self._balance['T'] = np.sum(self.flux *
                                    self._mesh.sigma *
                                    self._mesh.volume)

        # Calculate total source
        self._balance['Q'] = -self._source.strength

    def balance_table(self):
        frame = pd.DataFrame(pd.Series(self._balance), columns=['Balance'])
        return frame

    # Build the rays
    def _make_rays(self, rng):
        costheta  = 1.0 - 2.0*rng[0]
        phi       = 2.0 * np.pi * rng[1]
        sintheta  = math.sqrt(1.0 - costheta*costheta)
        cosphi    = math.cos(phi)
        sinphi    = math.sin(phi)

        return (sintheta*cosphi, sintheta*sinphi, costheta)

    @property
    def Nb(self):
        return self._Nb

    @Nb.setter
    def Nb(self, value):
        self._Nb = value

    @property
    def Np(self):
        return self._Np

    @Np.setter
    def Np(self, value):
        self._Np = value

    @property
    def flux(self):
        return self._tally.mean

    @property
    def batch_flux(self):
        return self._tally.batch_mean

    @property
    def tally(self):
        return self._tally

    @property
    def balance(self):
        return self._balance

###############################################################################
# end of notes/mc/pymc/mcuncf.py
###############################################################################
