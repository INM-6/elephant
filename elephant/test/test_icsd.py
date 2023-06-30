# -*- coding: utf-8 -*-
"""
iCSD testing suite
"""

import numpy as np
from numpy.testing import assert_array_almost_equal
import quantities as pq
import scipy.integrate as si
from scipy.interpolate import interp1d
from elephant.current_source_density import icsd
import unittest


def potential_of_plane(z_j, z_i=0. * pq.m,
                       C_i=1 * pq.A / pq.m**2,
                       sigma=0.3 * pq.S / pq.m):
    """
    Return potential of infinite horizontal plane with constant
    current source density at a vertical offset z_j.

    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to source layer
    z_i : float*pq.m
        z-position of source layer
    C_i : float*pq.A/pq.m**2
        current source density on circular disk in units of charge per area
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m

    Notes
    -----
    The potential is 0 at the plane, as the potential goes to infinity for
    large distances

    """
    try:
        assert z_j.units == z_i.units
    except AssertionError:
        raise AssertionError(f'units of z_j ({z_j.units}) and z_i ('
                             f'{z_i.units}) not equal')

    return -C_i / (2 * sigma) * abs(z_j - z_i).simplified


def potential_of_disk(z_j,
                      z_i=0. * pq.m,
                      C_i=1 * pq.A / pq.m**2,
                      R_i=1E-3 * pq.m,
                      sigma=0.3 * pq.S / pq.m):
    """
    Return potential of circular disk in horizontal plane with constant
    current source density at a vertical offset z_j.

    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to center of disk
    z_i : float*pq.m
        z_j-position of source disk
    C_i : float*pq.A/pq.m**2
        current source density on circular disk in units of charge per area
    R_i : float*pq.m
        radius of disk source
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m
    """
    try:
        assert z_j.units == z_i.units == R_i.units
    except AssertionError:
        raise AssertionError(f'units of z_j ({z_j.units}), z_i ({z_i.units}) '
                             f'and R_i ({R_i.units}) not equal')

    return C_i / (2 * sigma) * (
            np.sqrt((z_j - z_i) ** 2 + R_i**2) - abs(z_j - z_i)).simplified


def potential_of_cylinder(z_j,
                          z_i=0. * pq.m,
                          C_i=1 * pq.A / pq.m**3,
                          R_i=1E-3 * pq.m,
                          h_i=0.1 * pq.m,
                          sigma=0.3 * pq.S / pq.m,
                          ):
    """
    Return potential of cylinder in horizontal plane with constant homogeneous
    current source density at a vertical offset z_j.


    Arguments
    ---------
    z_j : float*pq.m
        distance perpendicular to center of disk
    z_i : float*pq.m
        z-position of center of source cylinder
    h_i : float*pq.m
        thickness of cylinder
    C_i : float*pq.A/pq.m**3
        current source density on circular disk in units of charge per area
    R_i : float*pq.m
        radius of disk source
    sigma : float*pq.S/pq.m
        conductivity of medium in units of S/m
    """
    try:
        assert z_j.units == z_i.units == R_i.units == h_i.units
    except AssertionError:
        raise AssertionError(f'units of z_j ({z_j.units}), z_i ({z_i.units}), '
                             f'R_i ({R_i.units}) and h ({h_i.units}) '
                             f'not equal')

    # speed up tests by stripping units
    _sigma = float(sigma)
    _R_i = float(R_i)
    _z_j = float(z_j)

    # evaluate integrand using quad
    def integrand(z):
        return 1 / (2 * _sigma) * \
            (np.sqrt((z - _z_j)**2 + _R_i**2) - abs(z - _z_j))

    phi_j = C_i * si.quad(integrand, z_i - h_i / 2, z_i + h_i / 2)[0]

    return phi_j * z_i.units**2 / sigma.units


def get_lfp_of_planes(z_j=np.arange(21) * 1E-4 * pq.m,
                      z_i=np.array([8E-4, 10E-4, 12E-4]) * pq.m,
                      C_i=np.array([-.5, 1., -.5]) * pq.A / pq.m**2,
                      sigma=0.3 * pq.S / pq.m,
                      plot=True):
    """
    Compute the lfp of spatially separated planes with given current source
    density
    """
    phi_j = np.zeros(z_j.size) * pq.V
    for i, (zi, Ci) in enumerate(zip(z_i, C_i)):
        phi_j += potential_of_plane(z_j, zi, Ci, sigma)

    # test plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        ax = plt.gca()
        ax.plot(np.zeros(z_j.size), z_j, 'r-o')
        for i, C in enumerate(C_i):
            ax.plot((0, C), (z_i[i], z_i[i]), 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_ylabel(f'z_j ({z_j.units})')
        ax.set_xlabel(f'C_i ({C_i.units})')
        ax.set_title('planar CSD')

        plt.subplot(122)
        ax = plt.gca()
        ax.plot(phi_j, z_j, 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_xlabel(f'phi_j ({phi_j.units})')
        ax.set_title('LFP')

    return phi_j, C_i


def get_lfp_of_disks(z_j=np.arange(21) * 1E-4 * pq.m,
                     z_i=np.array([8E-4, 10E-4, 12E-4]) * pq.m,
                     C_i=np.array([-.5, 1., -.5]) * pq.A / pq.m**2,
                     R_i=np.array([1, 1, 1]) * 1E-3 * pq.m,
                     sigma=0.3 * pq.S / pq.m,
                     plot=True):
    """
    Compute the lfp of spatially separated disks with a given
    current source density
    """
    phi_j = np.zeros(z_j.size) * pq.V
    for i, (zi, Ci, Ri) in enumerate(zip(z_i, C_i, R_i)):
        phi_j += potential_of_disk(z_j, zi, Ci, Ri, sigma)

    # test plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        ax = plt.gca()
        ax.plot(np.zeros(z_j.size), z_j, 'r-o')
        for i, C in enumerate(C_i):
            ax.plot((0, C), (z_i[i], z_i[i]), 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_ylabel(f'z_j ({z_j.units})')
        ax.set_xlabel(f'C_i ({C_i.units})')
        ax.set_title(f'disk CSD\nR={R_i}')

        plt.subplot(122)
        ax = plt.gca()
        ax.plot(phi_j, z_j, 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_xlabel(f'phi_j ({phi_j.units})')
        ax.set_title('LFP')

    return phi_j, C_i


def get_lfp_of_cylinders(z_j=np.arange(21) * 1E-4 * pq.m,
                         z_i=np.array([8E-4, 10E-4, 12E-4]) * pq.m,
                         C_i=np.array([-.5, 1., -.5]) * pq.A / pq.m**3,
                         R_i=np.array([1, 1, 1]) * 1E-3 * pq.m,
                         h_i=np.array([1, 1, 1]) * 1E-4 * pq.m,
                         sigma=0.3 * pq.S / pq.m,
                         plot=True):
    """
    Compute the lfp of spatially separated disks with a given
    current source density
    """
    phi_j = np.zeros(z_j.size) * pq.V
    for i, (zi, Ci, Ri, hi) in enumerate(zip(z_i, C_i, R_i, h_i)):
        for j, zj in enumerate(z_j):
            phi_j[j] += potential_of_cylinder(zj, zi, Ci, Ri, hi, sigma)

    # test plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        ax = plt.gca()
        ax.plot(np.zeros(z_j.size), z_j, 'r-o')
        ax.barh(np.asarray(z_i - h_i / 2),
                np.asarray(C_i),
                np.asarray(h_i), color='r')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_ylabel(f'z_j ({z_j.units})')
        ax.set_xlabel(f'C_i ({C_i.units})')
        ax.set_title(f'cylinder CSD\nR={R_i}')

        plt.subplot(122)
        ax = plt.gca()
        ax.plot(phi_j, z_j, 'r-o')
        ax.set_ylim(z_j.min(), z_j.max())
        ax.set_xlabel(f'phi_j ({phi_j.units})')
        ax.set_title('LFP')

    return phi_j, C_i


class TestICSD(unittest.TestCase):
    """
    Set of test functions for each CSD estimation method comparing
    estimate to LFPs calculated with known ground truth CSD
    """
    @classmethod
    def setUpClass(cls) -> None:
        # contact point coordinates
        z_j = np.arange(21) * 1E-4 * pq.m
        z_i = z_j
        cls.z_j = z_j
        cls.z_i = z_i
        # current source density magnitude
        C_i = np.zeros(z_j.size) * pq.A / pq.m**2
        C_i[7:12:2] += np.array([-.5, 1., -.5]) * pq.A / pq.m**2

        # current source density magnitude
        C_i_1 = np.zeros(z_i.size) * pq.A / pq.m**3
        C_i_1[7:12:2] += np.array([-.5, 1., -.5]) * pq.A / pq.m**3
        cls.C_i_1 = C_i_1

        # uniform conductivity
        sigma = 0.3 * pq.S / pq.m
        cls.sigma = sigma
        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_planes(z_j, z_i, C_i,
                                       sigma, plot)
        cls.C_i = C_i
        cls.phi_j = phi_j

        cls.R_i = np.ones(z_i.size) * 1E-3 * pq.m

    def test_StandardCSD_00(self):
        """test using standard SI units"""
        # set some parameters for ground truth csd and csd estimates.

        std_input = {
            'lfp': self.phi_j,
            'coord_electrode': self.z_j,
            'sigma': self.sigma,
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        std_csd = icsd.StandardCSD(**std_input)
        csd = std_csd.get_csd()

        self.assertEqual(self.C_i.units, csd.units)
        assert_array_almost_equal(self.C_i, csd)

    def test_StandardCSD_01(self):
        """test using non-standard SI units 1"""
        # set some parameters for ground truth csd and csd estimates.

        std_input = {
            'lfp': self.phi_j * 1E3 * pq.mV / pq.V,
            'coord_electrode': self.z_j,
            'sigma': self.sigma,
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        std_csd = icsd.StandardCSD(**std_input)
        csd = std_csd.get_csd()

        self.assertEqual(self.C_i.units, csd.units)
        assert_array_almost_equal(self.C_i, csd)

    def test_StandardCSD_02(self):
        """test using non-standard SI units 2"""
        # set some parameters for ground truth csd and csd estimates.

        std_input = {
            'lfp': self.phi_j,
            'coord_electrode': self.z_j * 1E3 * pq.mm / pq.m,
            'sigma': self.sigma,
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        std_csd = icsd.StandardCSD(**std_input)
        csd = std_csd.get_csd()

        self.assertEqual(self.C_i.units, csd.units)
        assert_array_almost_equal(self.C_i, csd)

    def test_StandardCSD_03(self):
        """test using non-standard SI units 3"""
        # set some parameters for ground truth csd and csd estimates.

        std_input = {
            'lfp': self.phi_j,
            'coord_electrode': self.z_j,
            'sigma': self.sigma * 1E3 * pq.mS / pq.S,
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        std_csd = icsd.StandardCSD(**std_input)
        csd = std_csd.get_csd()

        self.assertEqual(self.C_i.units, csd.units)
        assert_array_almost_equal(self.C_i, csd)

    def test_DeltaiCSD_00(self):
        """test using standard SI units"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        sigma_top = self.sigma

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(self.z_j, self.z_i, self.C_i, self.R_i,
                                      self.sigma, plot)
        delta_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j,
            'diam': self.R_i.mean() * 2,        # source diameter
            'sigma': self.sigma,           # extracellular conductivity
            'sigma_top': sigma_top,       # conductivity on top of cortex
            'f_type': 'gaussian',  # gaussian filter
            'f_order': (3, 1),     # 3-point filter, sigma = 1.
        }

        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd)

    def test_DeltaiCSD_01(self):
        """test using non-standard SI units 1"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        sigma_top = self.sigma

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(self.z_j, self.z_i, self.C_i, self.R_i,
                                      self.sigma, plot)
        delta_input = {
            'lfp': phi_j * 1E3 * pq.mV / pq.V,
            'coord_electrode': self.z_j,
            'diam': self.R_i.mean() * 2,        # source diameter
            'sigma': self.sigma,           # extracellular conductivity
            'sigma_top': sigma_top,       # conductivity on top of cortex
            'f_type': 'gaussian',  # gaussian filter
            'f_order': (3, 1),     # 3-point filter, sigma = 1.
        }

        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd)

    def test_DeltaiCSD_02(self):
        """test using non-standard SI units 2"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        sigma_top = self.sigma

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(self.z_j, self.z_i, self.C_i, self.R_i,
                                      self.sigma, plot)
        delta_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j * 1E3 * pq.mm / pq.m,
            'diam': self.R_i.mean() * 2 * 1E3 * pq.mm / pq.m,    # source diameter
            'sigma': self.sigma,           # extracellular conductivity
            'sigma_top': sigma_top,       # conductivity on top of cortex
            'f_type': 'gaussian',  # gaussian filter
            'f_order': (3, 1),     # 3-point filter, sigma = 1.
        }

        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd)

    def test_DeltaiCSD_03(self):
        """test using non-standard SI units 3"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        sigma_top = self.sigma

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(self.z_j, self.z_i, self.C_i, self.R_i,
                                      self.sigma, plot)
        delta_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j,
            'diam': self.R_i.mean() * 2,        # source diameter
            # extracellular conductivity
            'sigma': self.sigma * 1E3 * pq.mS / pq.S,
            'sigma_top': sigma_top * 1E3 * pq.mS / pq.S,  # conductivity on
                                                          # top of cortex
            'f_type': 'gaussian',  # gaussian filter
            'f_order': (3, 1),     # 3-point filter, sigma = 1.
        }

        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd)

    def test_DeltaiCSD_04(self):
        """test non-continous z_j array"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # source radius (delta, step)
        R_i = np.ones(self.z_j.size) * 1E-3 * pq.m

        sigma_top = self.sigma

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_disks(self.z_j, self.z_i, self.C_i, R_i,
                                      self.sigma, plot)
        inds = np.delete(np.arange(21), 5)
        delta_input = {
            'lfp': phi_j[inds],
            'coord_electrode': self.z_j[inds],
            'diam': R_i[inds] * 2,        # source diameter
            'sigma': self.sigma,           # extracellular conductivity
            'sigma_top': sigma_top,       # conductivity on top of cortex
            'f_type': 'gaussian',  # gaussian filter
            'f_order': (3, 1),     # 3-point filter, sigma = 1.
        }

        delta_icsd = icsd.DeltaiCSD(**delta_input)
        csd = delta_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i[inds], csd)

    def test_StepiCSD_units_00(self):
        """test using standard SI units"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # source height (cylinder)
        h_i = np.ones(self.z_i.size) * 1E-4 * pq.m

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, self.z_i, self.C_i_1,
                                          self.R_i, h_i, self.sigma, plot)

        step_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j,
            'diam': self.R_i.mean() * 2,
            'sigma': self.sigma,
            'sigma_top': self.sigma,
            'h': h_i,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd)

    def test_StepiCSD_01(self):
        """test using non-standard SI units 1"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # source height (cylinder)
        h_i = np.ones(self.z_i.size) * 1E-4 * pq.m

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, self.z_i, self.C_i_1,
                                          self.R_i, h_i, self.sigma, plot)

        step_input = {
            'lfp': phi_j * 1E3 * pq.mV / pq.V,
            'coord_electrode': self.z_j,
            'diam': self.R_i.mean() * 2,
            'sigma': self.sigma,
            'sigma_top': self.sigma,
            'h': h_i,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd)

    def test_StepiCSD_02(self):
        """test using non-standard SI units 2"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # source height (cylinder)
        h_i = np.ones(self.z_i.size) * 1E-4 * pq.m

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, self.z_i, self.C_i_1,
                                          self.R_i, h_i, self.sigma, plot)

        step_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j * 1E3 * pq.mm / pq.m,
            'diam': self.R_i.mean() * 2 * 1E3 * pq.mm / pq.m,
            'sigma': self.sigma,
            'sigma_top': self.sigma,
            'h': h_i * 1E3 * pq.mm / pq.m,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd)

    def test_StepiCSD_03(self):
        """test using non-standard SI units 3"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # source height (cylinder)
        h_i = np.ones(self.z_i.size) * 1E-4 * pq.m

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, self.z_i, self.C_i_1,
                                          self.R_i, h_i, self.sigma, plot)

        step_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j,
            'diam': self.R_i.mean() * 2,
            'sigma': self.sigma * 1E3 * pq.mS / pq.S,
            'sigma_top': self.sigma * 1E3 * pq.mS / pq.S,
            'h': h_i,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd)

    def test_StepiCSD_units_04(self):
        """test non-continuous z_j array"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # source height (cylinder)
        h_i = np.ones(self.z_i.size) * 1E-4 * pq.m

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, self.z_i, self.C_i_1,
                                          self.R_i, h_i, self.sigma, plot)
        inds = np.delete(np.arange(21), 5)
        step_input = {
            'lfp': phi_j[inds],
            'coord_electrode': self.z_j[inds],
            'diam': self.R_i[inds] * 2,
            'sigma': self.sigma,
            'sigma_top': self.sigma,
            'h': h_i[inds],
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        step_icsd = icsd.StepiCSD(**step_input)
        csd = step_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i[inds], csd)

    def test_SplineiCSD_00(self):
        """test using standard SI units"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # construct interpolators, spline method assume underlying source
        # pattern generating LFPs that are cubic spline interpolates between
        # contacts, so we generate CSD data relying on the same assumption
        f_C = interp1d(self.z_i, self.C_i_1, kind='cubic')
        f_R = interp1d(self.z_i, self.R_i)
        num_steps = 201
        z_i_i = np.linspace(float(self.z_i[0]), float(
            self.z_i[-1]), num_steps) * self.z_i.units
        C_i_i = f_C(np.asarray(z_i_i)) * self.C_i_1.units
        R_i_i = f_R(z_i_i) * self.R_i.units

        h_i_i = np.ones(z_i_i.size) * np.diff(z_i_i).min()

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          self.sigma, plot)

        spline_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j,
            'diam': self.R_i * 2,
            'sigma': self.sigma,
            'sigma_top': self.sigma,
            'num_steps': num_steps,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd, decimal=3)

    def test_SplineiCSD_01(self):
        """test using standard SI units, deep electrode coordinates"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # construct interpolators, spline method assume underlying source
        # pattern generating LFPs that are cubic spline interpolates between
        # contacts, so we generate CSD data relying on the same assumption
        f_C = interp1d(self.z_i, self.C_i_1, kind='cubic')
        f_R = interp1d(self.z_i, self.R_i)
        num_steps = 201
        z_i_i = np.linspace(float(self.z_i[0]), float(
            self.z_i[-1]), num_steps) * self.z_i.units
        C_i_i = f_C(np.asarray(z_i_i)) * self.C_i_1.units
        R_i_i = f_R(z_i_i) * self.R_i.units

        h_i_i = np.ones(z_i_i.size) * np.diff(z_i_i).min()

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          self.sigma, plot)

        spline_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j,
            'diam': self.R_i * 2,
            'sigma': self.sigma,
            'sigma_top': self.sigma,
            'num_steps': num_steps,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd, decimal=3)

    def test_SplineiCSD_02(self):
        """test using non-standard SI units"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # construct interpolators, spline method assume underlying source
        # pattern generating LFPs that are cubic spline interpolates between
        # contacts, so we generate CSD data relying on the same assumption
        f_C = interp1d(self.z_i, self.C_i_1, kind='cubic')
        f_R = interp1d(self.z_i, self.R_i)
        num_steps = 201
        z_i_i = np.linspace(float(self.z_i[0]), float(
            self.z_i[-1]), num_steps) * self.z_i.units
        C_i_i = f_C(np.asarray(z_i_i)) * self.C_i_1.units
        R_i_i = f_R(z_i_i) * self.R_i.units

        h_i_i = np.ones(z_i_i.size) * np.diff(z_i_i).min()

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          self.sigma, plot)

        spline_input = {
            'lfp': phi_j * 1E3 * pq.mV / pq.V,
            'coord_electrode': self.z_j,
            'diam': self.R_i * 2,
            'sigma': self.sigma,
            'sigma_top': self.sigma,
            'num_steps': num_steps,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd, decimal=3)

    def test_SplineiCSD_03(self):
        """test using standard SI units"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # construct interpolators, spline method assume underlying source
        # pattern generating LFPs that are cubic spline interpolates between
        # contacts, so we generate CSD data relying on the same assumption
        f_C = interp1d(self.z_i, self.C_i_1, kind='cubic')
        f_R = interp1d(self.z_i, self.R_i)
        num_steps = 201
        z_i_i = np.linspace(float(self.z_i[0]), float(
            self.z_i[-1]), num_steps) * self.z_i.units
        C_i_i = f_C(np.asarray(z_i_i)) * self.C_i_1.units
        R_i_i = f_R(z_i_i) * self.R_i.units

        h_i_i = np.ones(z_i_i.size) * np.diff(z_i_i).min()

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          self.sigma, plot)

        spline_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j * 1E3 * pq.mm / pq.m,
            'diam': self.R_i * 2 * 1E3 * pq.mm / pq.m,
            'sigma': self.sigma,
            'sigma_top': self.sigma,
            'num_steps': num_steps,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd, decimal=3)

    def test_SplineiCSD_04(self):
        """test using standard SI units"""
        # set some parameters for ground truth csd and csd estimates., e.g.,
        # we will use same source diameter as in ground truth

        # construct interpolators, spline method assume underlying source
        # pattern generating LFPs that are cubic spline interpolates between
        # contacts, so we generate CSD data relying on the same assumption
        f_C = interp1d(self.z_i, self.C_i_1, kind='cubic')
        f_R = interp1d(self.z_i, self.R_i)
        num_steps = 201
        z_i_i = np.linspace(float(self.z_i[0]), float(
            self.z_i[-1]), num_steps) * self.z_i.units
        C_i_i = f_C(np.asarray(z_i_i)) * self.C_i_1.units
        R_i_i = f_R(z_i_i) * self.R_i.units

        h_i_i = np.ones(z_i_i.size) * np.diff(z_i_i).min()

        # flag for debug plots
        plot = False

        # get LFP and CSD at contacts
        phi_j, C_i = get_lfp_of_cylinders(self.z_j, z_i_i, C_i_i, R_i_i, h_i_i,
                                          self.sigma, plot)

        spline_input = {
            'lfp': phi_j,
            'coord_electrode': self.z_j,
            'diam': self.R_i * 2,
            'sigma': self.sigma * 1E3 * pq.mS / pq.S,
            'sigma_top': self.sigma * 1E3 * pq.mS / pq.S,
            'num_steps': num_steps,
            'tol': 1E-12,          # Tolerance in numerical integration
            'f_type': 'gaussian',
            'f_order': (3, 1),
        }
        spline_icsd = icsd.SplineiCSD(**spline_input)
        csd = spline_icsd.get_csd()

        self.assertEqual(C_i.units, csd.units)
        assert_array_almost_equal(C_i, csd, decimal=3)
