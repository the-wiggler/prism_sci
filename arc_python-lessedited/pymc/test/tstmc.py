###############################################################################
# File  : notes/mc/pymc/test/tstmc.py
# Author: Tom Evans
# Date  : Thu Aug 31 11:58:39 2017
###############################################################################

import sys
import math

import matplotlib.pyplot as plt
import numpy as np

import unittest

##---------------------------------------------------------------------------##

sys.path.append('../..')
import pymc.uncf as uf

###############################################################################

class TestMC(unittest.TestCase):

    def make_rays(self, rng):
        costheta  = 1.0 - 2.0*rng[0]
        phi       = 2.0 * np.pi * rng[1]
        sintheta  = math.sqrt(1.0 - costheta*costheta)
        cosphi    = math.cos(phi)
        sinphi    = math.sin(phi)

        return (sintheta*cosphi, sintheta*sinphi, costheta)

    def test_rays(self):

        mesh   = uf.Mesh(shape=(5,5,5), delta=(0.5,0.5,0.5))
        rt     = uf.Raytracer(mesh)
        source = uf.Source(point=(1.25, 1.25, 1.25), strength=1.0)
        tally  = uf.Tally(mesh, source)

        # Make rays
        rngs = np.random.rand(10,2)
        rays = np.apply_along_axis(self.make_rays, 1, rngs)

        rt.trace(source, rays, tally)

        tally.finalize()

    def test_ray(self):

        mesh   = uf.Mesh(shape=(5,5,5), delta=(0.5,0.5,0.5))
        rt     = uf.Raytracer(mesh)
        source = uf.Source(point=(1.25, 1.25, 1.25), strength=1.0)
        tally  = uf.Tally(mesh, source)

        # Make rays
        rays = np.array([[0.194633, 0.966159, -0.169275]])
        rt.trace(source, rays, tally)

    def test_mcuncf(self):

        print()

        mesh   = uf.Mesh(shape=(5,5,5), delta=(0.5,0.5,0.5))
        source = uf.Source(point=(1.25,1.25,1.25), strength=1.0)
        mcuncf = uf.MCUNCF(source, mesh)

        self.assertEqual(1, mcuncf.Nb)
        self.assertEqual(1000, mcuncf.Np)

        mcuncf.Nb = 2
        mcuncf.Np = 100

        self.assertEqual(2, mcuncf.Nb)
        self.assertEqual(100, mcuncf.Np)

        mcuncf.calc_flux()

        # Get the flux and relative error
        flux = mcuncf.flux[2,2,2]
        error = np.sqrt(mcuncf.tally.var)[2,2,2]
        bt = mcuncf.balance_table()

        # The fluxes should be equal
        diff_flux = np.abs(mcuncf.flux - mcuncf.batch_flux)
        self.assertAlmostEqual(np.max(diff_flux), 0.0)

        # The variance should not
        diff_var = np.abs(mcuncf.tally.var - mcuncf.tally.batch_var)
        self.assertNotAlmostEqual(np.max(diff_var), 0.0)

        self.assertEqual(200, mcuncf.tally.N)

        balance = bt.sum(axis=0)
        self.assertAlmostEqual(balance['Balance'], 0.0)

        self.assertAlmostEqual(2.1, flux, delta=0.1)
        self.assertAlmostEqual(0.015, error, delta=0.01)

        print(bt)
        print()

    def test_batch(self):

        print()

        mesh   = uf.Mesh(shape=(5,5,5), delta=(0.5,0.5,0.5))
        source = uf.Source(point=(1.25,1.25,1.25), strength=1.0)
        mcuncf = uf.MCUNCF(source, mesh)

        mcuncf.Nb = 100
        mcuncf.Np = 1

        self.assertEqual(100, mcuncf.Nb)
        self.assertEqual(1, mcuncf.Np)

        mcuncf.calc_flux()

        self.assertEqual(100, mcuncf.tally.N)

        bt = mcuncf.balance_table()
        balance = bt.sum(axis=0)
        self.assertAlmostEqual(balance['Balance'], 0.0)
        print(bt)

        # Get the flux and relative error
        flux = mcuncf.flux
        var = mcuncf.tally.var

        batch_flux = mcuncf.batch_flux
        batch_var = mcuncf.tally.batch_var

        diff_flux = np.abs(flux - batch_flux)
        diff_var = np.abs(var - batch_var)

        self.assertAlmostEqual(np.max(diff_flux), 0.0)
        self.assertAlmostEqual(np.max(diff_var), 0.0)

        print()

##---------------------------------------------------------------------------##
if __name__ == '__main__':
    unittest.main()

###############################################################################
# end of notes/mc/pymc/test/tstmc.py
###############################################################################
