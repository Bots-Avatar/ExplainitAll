import math
from typing import Optional, Dict

import numpy as np
from sklearn import mixture
from sklearn.mixture import GaussianMixture


def rayleigh_el(el: float, disp: float) -> float:
    return (el / disp) * math.exp(-el ** 2 / disp)


def rayleigh(arr: np.ndarray, dispersion: float) -> np.ndarray:
    return np.array(list(map(lambda x: rayleigh_el(x, dispersion), arr)))


def rayleigh_el_integral(el: float, disp: float) -> float:
    return 1 - math.exp(-el ** 2 / disp)


def rayleigh_integral(arr: np.ndarray, dispersion: float) -> np.ndarray:
    return np.array(list(map(lambda x: rayleigh_el_integral(x, dispersion), arr)))


def gaussian_integral_single(element, mean, std, sqrt2):
    return 0.5 + 0.5 * math.erf((element - mean) / (sqrt2 * std))


def gaussian_mixture_integral_single(element: float, gaussian_mixture_model: GaussianMixture, sqrt2: float) -> float:
    means = gaussian_mixture_model.means_
    weights = gaussian_mixture_model.weights_
    variances = gaussian_mixture_model.covariances_[:, 0]

    integral_element = np.sum([
        weight * gaussian_integral_single(mean=mean[0], std=np.sqrt(variance[0]), element=element, sqrt2=sqrt2)
        for weight, mean, variance in zip(weights, means, variances)
    ])
    return float(integral_element)


def gaussian_mixture_integral(arr: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    sqrt2 = math.sqrt(2)
    if len(arr.shape) == 1:
        return np.array([gaussian_mixture_integral_single(x, gmm, sqrt2) for x in arr])

    if len(arr.shape) == 2:
        dim_1, dim_2 = arr.shape
        reshaped_arr = arr.reshape((dim_1 * dim_2))
        reshaped_arr = np.array([gaussian_mixture_integral_single(x, gmm, sqrt2) for x in reshaped_arr])
        return reshaped_arr.reshape((dim_1, dim_2))


def denormalize_array(array: np.ndarray) -> np.ndarray:
    edited = array.copy()
    non_zero_counts = np.count_nonzero(edited, axis=0)
    edited *= non_zero_counts
    return edited


def calc_gauss_mixture_stat_params(array: np.ndarray,
                                   num_components: int = 3,
                                   seed: Optional[int] = None) -> np.ndarray:
    "Рассчет нового массива на базе гауссовой смеси"
    d_array_2d = denormalize_array(array)

    d_array_1d = d_array_2d[~np.isnan(array)]
    d_array_1d = d_array_1d.reshape(len(d_array_1d), 1)

    gmm = mixture.GaussianMixture(n_components=num_components, random_state=0)
    gmm.fit(d_array_1d)
    return gaussian_mixture_integral(d_array_2d, gmm)


def calc_gmm_stat_params(array: np.ndarray) -> Dict:
    array_1d = array[np.logical_not(np.isnan(array))]

    mean = float(np.mean(array_1d))
    std = float(np.std(array_1d))
    new_arr = calc_gauss_mixture_stat_params(array)

    return {'new_arr': new_arr, "mean": mean, "std": std}


def compute_gaussian_integral(array: np.ndarray, mean: float, std_dev: float) -> np.ndarray:
    """Интегральная Гауссова функция для 1д-2д массива"""
    sqrt2 = math.sqrt(2)
    vectorized_gaussian = np.vectorize(gaussian_integral_single)
    array_1d = array.flatten()
    result_array = vectorized_gaussian(array_1d, mean, std_dev, sqrt2)
    return result_array.reshape(array.shape)
