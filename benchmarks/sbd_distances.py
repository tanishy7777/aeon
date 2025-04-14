"""Shape-based distance (SBD) between two time series."""

__maintainer__ = ["SebastianSchmidl"]


import numpy as np
from numba import objmode
from scipy.signal import correlate


def sbd_distance_fix(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_sbd_distance_fix(x, y, standardize)
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == 1 and y.shape[0] == 1:
            _x = x.ravel()
            _y = y.ravel()
            return _univariate_sbd_distance_fix(_x, _y, standardize)
        else:
            nchannels = min(
                x.shape[0], y.shape[0]
            )  # both x and y have the same number of channels
            norm = np.linalg.norm(x.astype(np.float64)) * np.linalg.norm(
                y.astype(np.float64)
            )
            distance = np.zeros((2 * x.shape[1] - 1,))
            for i in range(nchannels):
                distance += _helper_sbd(x[i], y[i], standardize)
            return np.abs(1 - np.max(distance) / norm)

    raise ValueError("x and y must be 1D or 2D")


def _univariate_sbd_distance_fix(
    x: np.ndarray, y: np.ndarray, standardize: bool
) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if standardize:
        if x.size == 1 or y.size == 1:
            return 0.0

        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)

    a = _helper_sbd(x, y, standardize)

    b = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.abs(1.0 - np.max(a / b))


# @njit(cache=True, fastmath=True)
def _helper_sbd(x, y, standardize):

    with objmode(a="float64[:]"):
        a = correlate(x, y, method="fft")

    return a


# -------------------------------------------------
def sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool = True) -> float:
    if x.ndim == 1 and y.ndim == 1:
        return _univariate_sbd_distance(x, y, standardize)
    if x.ndim == 2 and y.ndim == 2:
        if x.shape[0] == 1 and y.shape[0] == 1:
            _x = x.ravel()
            _y = y.ravel()
            return _univariate_sbd_distance(_x, _y, standardize)
        else:
            # independent (time series should have the same number of channels!)
            nchannels = min(x.shape[0], y.shape[0])
            distance = 0.0
            for i in range(nchannels):
                distance += _univariate_sbd_distance(x[i], y[i], standardize)
            return distance / nchannels

    raise ValueError("x and y must be 1D or 2D")


def _univariate_sbd_distance(x: np.ndarray, y: np.ndarray, standardize: bool) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if standardize:
        if x.size == 1 or y.size == 1:
            return 0.0

        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)

    with objmode(a="float64[:]"):
        a = correlate(x, y, method="fft")

    b = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.abs(1.0 - np.max(a / b))


from numpy.fft import fft, ifft

# ---------------------
from numpy.linalg import norm


def sbd_original(x, y):
    den = norm(x, axis=(0, 1)) * norm(y, axis=(0, 1))

    if den < 1e-9:
        den = np.inf

    x_len = x.shape[0]
    fft_size = 1 << (2 * x_len - 1).bit_length()

    cc = ifft(fft(x, fft_size, axis=0) * np.conj(fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len - 1) :], cc[:x_len]), axis=0)

    cc = np.real(cc).sum(axis=-1) / den

    return 1 - np.max(cc)


# ---------------------------
from tslearn.metrics.cycc import normalized_cc


def sbd_tslearn(x, y):
    return 1 - normalized_cc(x, y, norm1=-1, norm2=-1).max()
