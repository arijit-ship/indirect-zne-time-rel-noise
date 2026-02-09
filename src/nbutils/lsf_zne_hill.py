# lsf_zne_hill.py
"""
ZNE fitting using a Hill function with an exponential noise tail.

This module implements a staged fitting strategy for Zero-Noise
Extrapolation (ZNE) based on the following models.

----------------------------------------------------------------------
1. Noiseless Hill model
----------------------------------------------------------------------

The noiseless model is given by the Hill function:

    f(t) = ZNE + A / (1 + (t / T0)^n)

where:
    t   : noise scaling parameter (e.g. T_max)
    ZNE : zero-noise extrapolated value
    A   : positive amplitude (f(0) = ZNE + A)
    T0  : characteristic noise scale
    n   : integer Hill exponent (fixed, n = 1, 2, 3, ...)

Limits:
    f(0)        = ZNE + A
    f(t → ∞)    → ZNE

----------------------------------------------------------------------
2. Exponential noise tail (large-t behavior)
----------------------------------------------------------------------

For large t, the data is assumed to decay exponentially toward a
mixed-state energy E_mix:

    f_tail(t) = E_mix + C * exp(-gamma * t)

where:
    gamma : decay rate
    C     : amplitude
    E_mix : energy of the completely mixed state

----------------------------------------------------------------------
3. Full noisy model
----------------------------------------------------------------------

The full model interpolates between the Hill behavior at small t
and the exponential decay at large t:

    f_noisy(t) = E_mix + exp(-gamma * t) * ( f(t) - E_mix )

Explicitly:

    f_noisy(t) = E_mix
                 + exp(-gamma * t)
                   * [ ZNE + A / (1 + (t / T0)^n) - E_mix ]

----------------------------------------------------------------------
4. Fitting strategy
----------------------------------------------------------------------

Step 1:
    Estimate E_mix from large-t data (or fix it from physics).

Step 2:
    Fit gamma using only large-t data and the exponential tail model.

Step 3:
    Fit (ZNE, A, T0) using small-t data and the noiseless Hill model
    with fixed integer n.

Step 4 (optional):
    Perform a full fit using all parameters except n and E_mix.

This staged approach is essential because the full model is highly
nonlinear and difficult to fit in one shot.
"""

# ============================================================
# Imports
# ============================================================

import numpy as np
from scipy.optimize import curve_fit


# ============================================================
# Model functions
# ============================================================

def hill(t, ZNE, A, T0, n):
    """
    Noiseless Hill function.

    f(t) = ZNE + A / (1 + (t / T0)^n)
    """
    return ZNE + A / (1.0 + (t / T0)**n)


def exp_decay_tail(t, C, gamma, E_mix):
    """
    Exponential decay toward mixed-state energy.

    f(t) = E_mix + C * exp(-gamma * t)
    """
    return E_mix + C * np.exp(-gamma * t)


def full_model(t, ZNE, A, T0, gamma, n, E_mix):
    """
    Full noisy ZNE model.

    f(t) = E_mix + exp(-gamma * t) * ( hill(t) - E_mix )
    """
    f0 = hill(t=t, ZNE=ZNE, A=A, T0=T0, n=n)
    return E_mix + np.exp(-gamma * t) * (f0 - E_mix)


# ============================================================
# Data selection helpers
# ============================================================

def select_small_T(t, T_small):
    """Boolean mask for small-t region."""
    return t <= T_small


def select_large_T(t, T_large):
    """Boolean mask for large-t region."""
    return t >= T_large


# ============================================================
# Parameter estimation helpers
# ============================================================

def estimate_E_mix(t, y, T_large):
    """
    Estimate E_mix from large-t data as a simple average.
    """
    mask = select_large_T(t=t, T_large=T_large)
    return np.mean(y[mask])


# ============================================================
# Fitting routines
# ============================================================

def fit_gamma(
    t,
    y,
    sigma,
    E_mix,
    ZNE,
    *,
    p0=None,
    bounds=None,
    method="trf",
    max_nfev=None,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    verbose=0,
):
    """
    Fit exponential decay rate gamma using only large-t data.

    Model:
        f_noisy(t) = E_mix
                     + exp(-gamma * t)
                       * [ (ZNE + A / (1 + (t / T0)**n)) - E_mix ]

    Parameters
    ----------
    t : array_like
        T_max data
    y : array_like
        Observed energy values
    sigma : array_like or None
        Standard deviations
    E_mix : float
        Mixed-state energy (fixed)
    ZNE : float
        Zero-noise extrapolated energy (fixed)

    Keyword-only parameters
    -----------------------
    p0 : list or None
        Initial guess [gamma, A, T0, n]
    bounds : 2-tuple or None
        Bounds ([gamma_min, A_min, T0_min, n_min],
                [gamma_max, A_max, T0_max, n_max])
    method : str
        Optimization method ("trf", "dogbox", "lm")
    max_nfev : int or None
        Maximum function evaluations
    ftol, xtol, gtol : float
        Convergence tolerances
    verbose : int
        Verbosity level

    Returns
    -------
    popt : ndarray
        [gamma, A, T0, n]
    pcov : ndarray
        Covariance matrix
    """

    def model(t, gamma, A, T0, n):
        return (
            E_mix
            + np.exp(-gamma * t)
            * ((ZNE + A / (1.0 + (t / T0) ** n)) - E_mix)
        )

    if p0 is None:
        p0 = [0.01, y[0] - ZNE, t[len(t) // 2], 2.0]

    if bounds is None:
        bounds = (
            [0.0, -np.inf, 1e-12, 0.0],   # gamma, A, T0, n
            [np.inf, np.inf, np.inf, np.inf],
        )

    popt, pcov = curve_fit(
        model,
        t,
        y,
        sigma=sigma,
        p0=p0,
        bounds=bounds,
        method=method,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        verbose=verbose,
        absolute_sigma=True,
    )

    return popt, pcov



def fit_hill_noiseless(
    t,
    y,
    sigma,
    n,
    *,
    p0=None,
    bounds=None,
    method="trf",
    max_nfev=None,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    verbose=0,
):
    """
    Fit the noiseless Hill model with fixed integer n.

    Model:
        f(t) = ZNE + A / (1 + (t / T0)^n)

    Parameters
    ----------
    t : array_like
        T_max data
    y : array_like
        Observed energy values
    sigma : array_like or None
        Standard deviations
    n : int
        Fixed Hill exponent

    Keyword-only parameters
    -----------------------
    p0 : list or None
        Initial guess [ZNE, A, T0]
    bounds : 2-tuple or None
        ([lower bounds], [upper bounds])
    method : str
        Optimization method ("trf", "dogbox", "lm")
    max_nfev : int or None
        Maximum function evaluations
    ftol, xtol, gtol : float
        Convergence tolerances
    verbose : int
        Verbosity level passed to optimizer

    Returns
    -------
    popt : ndarray
        Optimal parameters [ZNE, A, T0]
    pcov : ndarray
        Covariance matrix
    """

    def model(t, ZNE, A, T0):
        return hill(t=t, ZNE=ZNE, A=A, T0=T0, n=n)

    # Defaults (only if user does not provide them)
    # p0 = [ZNE_guess, A0_guess, T0_guess]
    if p0 is None:
        p0 = [-6.0, 10.0, 30.0]

    if bounds is None:
        bounds = (
            [-7.0, 0.0, 10.0],
            [-5.0, 20.0, 50.0],
        )

    popt, pcov = curve_fit(
        f=model,
        xdata=t,
        ydata=y,
        sigma=sigma,
        p0=p0,
        bounds=bounds,
        method=method,
        absolute_sigma=(sigma is not None),
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        verbose=verbose,
    )

    return popt, pcov

def fit_full(
    t,
    y,
    sigma,
    init_params,
    n,
    E_mix,
):
    """
    Full fit using the complete noisy model with fixed n and E_mix.

    init_params = [ZNE, A, T0, gamma]
    """
    def model(t, ZNE, A, T0, gamma):
        return full_model(
            t=t,
            ZNE=ZNE,
            A=A,
            T0=T0,
            gamma=gamma,
            n=n,
            E_mix=E_mix,
        )

    popt, pcov = curve_fit(
        f=model,
        xdata=t,
        ydata=y,
        sigma=sigma,
        p0=init_params,
        absolute_sigma=(sigma is not None),
    )

    return popt, pcov
