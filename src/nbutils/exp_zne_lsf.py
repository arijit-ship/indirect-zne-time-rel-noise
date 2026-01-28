import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================================
# 1. Models
# ==========================================================

def exp_model(t, O0, gamma):
    """
    Simple exponential decay model 1.
    Formula: O(t) = O0 * exp(-gamma * t)
    
    Parameters:
        t (ndarray): Noise scaling parameters (T_max).
        O0 (float): Expectation value at zero noise.
        gamma (float): Effective decay rate.
    """
    return O0 * np.exp(-gamma * t)

def zne_exp_model(t, alpha, O0, gamma):
    """
    Exponential decay model 2.
    Formula: O(t) = alpha + O0 * exp(-gamma * t)
    """
    return alpha + O0 * np.exp(-gamma * t)

def biexp_model(t, O1, g1, O2, g2):
    """
    Bi-exponential decay model for multiple noise scales.
    Formula: O(t) = O1*exp(-g1*t) + O2*exp(-g2*t)
    """
    return O1 * np.exp(-g1 * t) + O2 * np.exp(-g2 * t)

# ==========================================================
# 2. Unified Fitting and Plotting
# ==========================================================

def fit_exponential(t, y, sigma, model=exp_model, p0=None):
    """
    Perform nonlinear least-squares fitting using a specified model.
    
    This function is designed to be backward compatible with notebooks expecting 
    a return signature of (O0, gamma, gamma_err, pcov).
    
    Parameters:
        t (ndarray): Noise scaling parameters.
        y (ndarray): Observable mean values.
        sigma (ndarray): Standard deviations for weighting.
        model (callable): The physics model to fit. Defaults to exp_model.
        p0 (tuple, optional): Initial parameter guesses.
        
    Returns:
        val0 (float): The extrapolated zero-noise value O(0).
        decay (float): The primary fitted decay rate (gamma).
        decay_err (float): The 1-sigma uncertainty of the decay rate.
        pcov (ndarray): The full covariance matrix from curve_fit.
    """

    # Safety: zne_exp_model has 3 parameters, so it needs 3 data points
    if len(t) < 3 and model == zne_exp_model:
        raise ValueError("Window too small! 3-param model needs 3 points.")

    # Heuristic initial guesses based on model selection
    if p0 is None:
        if model == exp_model:
            p0 = (y[0], 0.05)
        elif model == zne_exp_model:
            p0 = (y[-1], y[0] - y[-1], 0.05)
        elif model == biexp_model:
            p0 = (y[0]*0.7, 0.02, y[0]*0.3, 0.1)

    popt, pcov = curve_fit(
        model, t, y, p0=p0, sigma=sigma, absolute_sigma=True, maxfev=10000
    )
    
    errs = np.sqrt(np.diag(pcov))
    
    # Extract O(0) and primary decay for backward compatibility
    if model == exp_model:
        val0, decay, decay_err = popt[0], popt[1], errs[1]
    elif model == zne_exp_model:
        val0 = popt[0] + popt[1] # alpha + O0
        decay, decay_err = popt[2], errs[2]
    elif model == biexp_model:
        val0 = popt[0] + popt[2] # O1 + O2
        decay, decay_err = popt[1], errs[1]
        
    return val0, decay, decay_err, pcov

def plot_fit(t, y, sigma, O0, gamma, ax=None, label=None):
    """
    Visualize data and the resulting fit, extrapolated to t=0.
    
    Parameters:
        t, y, sigma: Data arrays.
        O0 (float): Extrapolated zero-noise value.
        gamma (float): Fitted decay rate.
        ax (Axes): Matplotlib axes object.
        label (str): Label for the plot legend.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Create extrapolation line starting from t=0
    t_plot = np.linspace(0, t.max() * 1.1, 500)
    y_plot = exp_model(t_plot, O0, gamma)

    ax.errorbar(t, y, yerr=sigma, fmt='o', label='Data', alpha=0.6)
    ax.plot(t_plot, y_plot, label=label or 'Fit')
    ax.set_xlim(left=0)
    ax.set_xlabel(r"$T_{max}$")
    ax.set_ylabel(r"$\langle O \rangle$")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    return ax

# ==========================================================
# 3. Data Utilities (Cleaned)
# ==========================================================

def extract_data(data):
    """
    Extract t, y, sigma arrays from raw data.

    Parameters
    ----------
    data : list
        [[ [mean, std], T_max ], ...]

    Returns
    -------
    t : ndarray
        T_max values
    y : ndarray
        Observable means (can be negative)
    sigma : ndarray
        Standard deviations
    """
    t = np.array([d[1] for d in data])
    y = np.array([d[0][0] for d in data])
    sigma = np.array([d[0][1] for d in data])
    return t, y, sigma

def select_window(t, y, sigma, tmin, tmax):
    """Filters data for a specific fitting window."""
    mask = (t >= tmin) & (t <= tmax)
    return t[mask], y[mask], sigma[mask]
