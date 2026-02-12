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


# ============================================================
# Fitting routines
# ============================================================

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

    output = {
        "input": {
            "t": t,
            "y": y,
            "sigma": sigma,
            "n": n,
            "p0": p0,
            "bounds": bounds,
            "method": method,
            "max_nfev": max_nfev,
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
            "verbose": verbose,
        },
        "output": {
            "popt": popt,
            "pcov": pcov,
        },
    }
    return output

def fit_gamma_largeT(
    t,
    y,
    sigma,
    q=1.0,
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
    Fit exponential decay rate gamma using large-t data only.

    Model:
        f_noisy_largeT(t) = B * exp(-gamma * t)

    Parameters
    ----------
    t : array_like
        T_max data (large-t region)
    y : array_like
        Observed values
    sigma : array_like or None
        Standard deviations

    Keyword-only parameters
    -----------------------
    p0 : list or None
        Initial guess [gamma, B]
    bounds : 2-tuple or None
        Bounds ([gamma_min, B_min],
                [gamma_max, B_max])
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
        [gamma, B]
    pcov : ndarray
        Covariance matrix
    """

    def model(t, gamma, B, q=q):
        return B * np.exp(-gamma * (t**q))

    if p0 is None:
        p0 = [
            0.01,     # gamma
            y[0],     # B ~ first large-t value
        ]

    if bounds is None:
        # Parameter order: [gamma, B]
        bounds = (
            [0.0, -np.inf],
            [np.inf, np.inf],
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

    output = {
        "input": {
            "t": t,
            "y": y,
            "sigma": sigma,
            "p0": p0,
            "bounds": bounds,
            "method": method,
            "max_nfev": max_nfev,
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
            "verbose": verbose,
        },
        "output": {
            "popt": popt,
            "pcov": pcov,
        },
    }

    return output

def fit_smallT_ZNE(
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
    Fit (ZNE, A, T0) using small-t data.

    Model:
        f_noisy_smallT(t) = ZNE + A / (1 + (t / T0)**n)

    Parameters
    ----------
    t : array_like
        T_max data (small-t window)
    y : array_like
        Observed energy values
    sigma : array_like or None
        Standard deviations
    n : float
        Fixed power-law exponent

    Keyword-only parameters
    -----------------------
    p0 : list or None
        Initial guess [ZNE, A, T0]
    bounds : 2-tuple or None
        Bounds ([ZNE_min, A_min, T0_min],
                [ZNE_max, A_max, T0_max])
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
    output : dict
        Dictionary with input metadata and fit results
    """

    def model(t, ZNE, A, T0):
        return ZNE + A / (1.0 + (t / T0) ** n)

    if p0 is None:
        p0 = [
            y[-1],             # ZNE ~ largest-t value in small window
            y[0] - y[-1],      # A   ~ curvature at t ~ 0
            np.median(t),      # T0  ~ middle of small-t window
        ]

    if bounds is None:
        bounds = (
            [-20.0,   0.0,  1e-6],   # ZNE, A, T0
            [ -4.0, 200.0, 10.0],
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

    output = {
        "input": {
            "t": t,
            "y": y,
            "sigma": sigma,
            "n": n,
            "p0": p0,
            "bounds": bounds,
            "method": method,
            "max_nfev": max_nfev,
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
            "verbose": verbose,
        },
        "output": {
            "popt": popt,
            "pcov": pcov,
        },
    }

    return output

def fit_f_noisy(
    t,
    y,
    sigma,
    n,
    q=1.0,
    *,
    p0,
    bounds,
    method="trf",
    max_nfev=None,
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    verbose=0,
):
    """
    Global fit of noisy model using all T_max data.

    Model:
        f_noisy(t) = exp(-gamma * t)
                     * [ ZNE + A / (1 + (t / T0)**n) ]

    Parameters
    ----------
    t : array_like
        T_max data
    y : array_like
        Observed energies
    sigma : array_like or None
        Standard deviations
    n : float
        Fixed exponent

    Keyword-only parameters
    -----------------------
    p0 : list
        Initial guess [gamma, ZNE, A, T0]
    bounds : 2-tuple
        Parameter bounds
    method : str
        Optimization method
    max_nfev : int or None
        Maximum function evaluations
    ftol, xtol, gtol : float
        Convergence tolerances
    verbose : int
        Verbosity level

    Returns
    -------
    dict with keys:
        popt : ndarray
            [gamma, ZNE, A, T0]
        pcov : ndarray
            Covariance matrix
    """

    def model(t, gamma, ZNE, A, T0,q=q):
        return np.exp(-gamma * (t**q)) * (
            ZNE + A / (1.0 + (t / T0) ** n)
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

    output = {
        "input": {
            "t": t,
            "y": y,
            "sigma": sigma,
            "n": n,
            "p0": p0,
            "bounds": bounds,
            "method": method,
            "max_nfev": max_nfev,
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
            "verbose": verbose,
        },
        "output": {
            "popt": popt,
            "pcov": pcov,
        },
    }

    return output

def fit_fq_noisy(
    t,
    y,
    sigma,
    n,
    *,
    p0,
    bounds,
    method="trf",
    max_nfev=None,
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    verbose=0,
):
    """
    Global fit of noisy model using all T_max data.

    Model:
        f_noisy(t) = exp(-gamma * (t**q))
                     * [ ZNE + A / (1 + (t / T0)**n) ]

    Parameters
    ----------
    t : array_like
        T_max data
    y : array_like
        Observed energies
    sigma : array_like or None
        Standard deviations
    n : float
        Fixed exponent for the Hill-like term

    Keyword-only parameters
    -----------------------
    p0 : list
        Initial guess [gamma, ZNE, A, T0, q]
    bounds : 2-tuple
        Parameter bounds ([low_gamma, low_ZNE, ...], [high_gamma, high_ZNE, ...])
    method : str
        Optimization method
    max_nfev : int or None
        Maximum function evaluations
    ftol, xtol, gtol : float
        Convergence tolerances
    verbose : int
        Verbosity level

    Returns
    -------
    dict with keys:
        popt : ndarray
            Optimized [gamma, ZNE, A, T0, q]
        pcov : ndarray
            Covariance matrix
    """

    # n is passed from the outer scope and remains fixed.
    # q is now an argument for the optimizer to vary.
    def model(t, gamma, ZNE, A, T0, q):
        return np.exp(-gamma * (t**q)) * (
            ZNE + A / (1.0 + (t / T0) ** n)
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

    output = {
        "input": {
            "t": t,
            "y": y,
            "sigma": sigma,
            "n": n,
            "p0": p0,
            "bounds": bounds,
            "method": method,
            "max_nfev": max_nfev,
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
            "verbose": verbose,
        },
        "output": {
            "popt": popt,
            "pcov": pcov,
        },
    }

    return output

def fit_fq_noisy_gamma_q_only(
    t,
    y,
    sigma,
    *,
    ZNE,
    A,
    T0,
    n,
    p0,
    bounds,
    method="trf",
    max_nfev=None,
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    verbose=0,
):
    """
    Fit noisy model with fixed (ZNE, A, T0, n),
    optimizing only (gamma, q).

    Model:
        f_noisy(t) = exp(-gamma * (t**q))
                     * [ ZNE + A / (1 + (t / T0)**n) ]

    Parameters
    ----------
    t : array_like
        T_max data
    y : array_like
        Observed energies
    sigma : array_like or None
        Standard deviations

    Fixed parameters
    ----------------
    ZNE : float
    A   : float
    T0  : float
    n   : float

    Keyword-only parameters
    -----------------------
    p0 : list
        Initial guess [gamma, q]
    bounds : 2-tuple
        Parameter bounds ([low_gamma, low_q], [high_gamma, high_q])
    method : str
        Optimization method
    max_nfev : int or None
        Maximum function evaluations
    ftol, xtol, gtol : float
        Convergence tolerances
    verbose : int
        Verbosity level

    Returns
    -------
    dict with keys:
        popt : ndarray
            Optimized [gamma, q]
        pcov : ndarray
            Covariance matrix
    """

    def model(t, gamma, q):
        return np.exp(-gamma * (t ** q)) * (
            ZNE + A / (1.0 + (t / T0) ** n)
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

    output = {
        "input": {
            "t": t,
            "y": y,
            "sigma": sigma,
            "ZNE": ZNE,
            "A": A,
            "T0": T0,
            "n": n,
            "p0": p0,
            "bounds": bounds,
            "method": method,
            "max_nfev": max_nfev,
            "ftol": ftol,
            "xtol": xtol,
            "gtol": gtol,
            "verbose": verbose,
        },
        "output": {
            "popt": popt,   # [gamma, q]
            "pcov": pcov,
        },
    }

    return output
