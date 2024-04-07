import math

import numpy as np
import pytest
from sklearn import mixture

from explainitall import (stat_helpers)


def test_rayleigh_el():
    el = 1.0
    disp = 2.0
    expected = 0.303
    result = stat_helpers.rayleigh_el(el, disp)
    assert math.isclose(result, expected, rel_tol=1e-3)


@pytest.mark.parametrize("arr, dispersion, expected", [
    (np.array([1, 2, 3]), 2.0, np.array([0.303, 0.135, 0.016])),
])
def test_rayleigh(arr, dispersion, expected):
    result = stat_helpers.rayleigh(arr, dispersion)
    np.testing.assert_almost_equal(result, expected, decimal=3)


@pytest.mark.parametrize("el, disp, expected", [
    (1.0, 2.0, 0.3934),
    (2.0, 1.5, 0.9305),
    (0.5, 3.0, 0.0799)
])
def test_rayleigh_el_int(el, disp, expected):
    result = stat_helpers.rayleigh_el_integral(el, disp)
    assert math.isclose(result, expected, rel_tol=1e-3)


@pytest.mark.parametrize("arr, dispersion, expected", [
    (np.array([1, 2, 3]), 1.0, np.array([0.6321, 0.9816, 0.99987])),
    (np.array([0, 0.5, 1]), 0.5, np.array([0.0, 0.39346, 0.8646]))])
def test_rayleigh_int(arr, dispersion, expected):
    result = stat_helpers.rayleigh_integral(arr, dispersion)
    np.testing.assert_almost_equal(result, expected, decimal=4)


@pytest.mark.parametrize("el, mu, std, sqrt2, expected", [
    (1.0, 2.0, 0.3934, math.sqrt(2), 0.0055119),
    (2.0, 1.5, 0.9305, math.sqrt(2), 0.7044855),
    (0.5, 3.0, 0.0799, math.sqrt(2), 0.0)
])
def test_gauss_integral_element(el, mu, std, sqrt2, expected):
    result = stat_helpers.gaussian_integral_single(el, mu, std, sqrt2)
    assert math.isclose(result, expected, rel_tol=1e-3)


def test_gauss_m_integral_element():
    arr = np.array([1, 2, 3, 4, 5])
    gmm = mixture.GaussianMixture(n_components=2, random_state=0)
    gmm.fit(arr.reshape(-1, 1), 1)
    el = 1
    sqrt2 = math.sqrt(2)
    result = stat_helpers.gaussian_mixture_integral_single(el, gmm, sqrt2)
    expected_result = 0.07186
    assert math.isclose(result, expected_result, rel_tol=1e-3)


def test_gauss_m_integral_1D():
    arr = np.array([1, 2, 3, 4, 5])
    gmm = mixture.GaussianMixture(n_components=2, random_state=0)
    gmm.fit(arr.reshape(-1, 1), 1)
    expected_result = np.array([0.0719, 0.2886, 0.5261, 0.6753, 0.9324])
    result = stat_helpers.gaussian_mixture_integral(arr=arr, gmm=gmm)
    np.testing.assert_almost_equal(result, expected_result, decimal=4)


def test_gauss_m_integral_2D():
    arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    gmm = mixture.GaussianMixture(n_components=2, random_state=0)
    gmm.fit(arr.reshape(-1, 1), 1)
    expected_result = np.array([[0.1676, 0.4391, 0.8942],
                                [0.1676, 0.4391, 0.8942],
                                [0.1676, 0.4391, 0.8942]])
    result = stat_helpers.gaussian_mixture_integral(arr=arr, gmm=gmm)
    np.testing.assert_almost_equal(result, expected_result, decimal=4)


def test_gauss_integral_1D():
    arr = np.array([1, 2, 3, 4, 5])
    mu = 3
    std = 1
    expected_result = np.array([0.0227, 0.1586, 0.5000, 0.8413, 0.9772])
    result = stat_helpers.compute_gaussian_integral(arr, mu, std)
    np.testing.assert_almost_equal(result, expected_result, decimal=4)

def test_gauss_integral_2D():
    arr = np.array([[1, 2], [3, 4]])
    mu = 2
    std = 1
    expected_result = np.array([[0.15865525, 0.5], [0.84134475, 0.97724987]])
    result = stat_helpers.compute_gaussian_integral(arr, mu, std)
    np.testing.assert_almost_equal(result, expected_result, decimal=4)


def test_de_normalizy():
    # Test case 1: Normalized array with non-zero values
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = np.array([[2., 4., 6.], [8., 10., 12.]])
    np.testing.assert_almost_equal(stat_helpers.denormalize_array(arr), expected, decimal=1)


def test_calc_gauss_mixture_stat_params():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    result = stat_helpers.calc_gauss_mixture_stat_params(arr, num_components=1, seed=0)
    expected = np.array([[0.061, 0.123, 0.219], [0.349, 0.5, 0.651], [0.781, 0.877, 0.939]])
    np.testing.assert_almost_equal(result, expected, decimal=3)
