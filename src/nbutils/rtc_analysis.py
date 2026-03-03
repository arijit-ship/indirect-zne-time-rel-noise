import nbutils.rtc_fitting as rtc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys
from typing import List, Dict
from pypalettes import load_palette
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



def perform_rtc_analysis(
    R_data,
    T_data,
    target_T,
    target_C,
    T_by_C,
    degree_s1=2,
    degree_s2=[2, 2],
    figsize=(14, 16),
    s1_alpha=0.6,
    pltxt_loc=(0.5, 0.1),
):
    """
    Perform a two-stage polynomial fitting of the response function R(T, C)
    under the physical constraint R(T, C=0) = 0.

    The model assumes a polynomial expansion in the control parameter C
    with -dependent coefficients:

        R(T, C) = C·F₁(T) + C²·F₂(T) + C³·F₃(T) + ...

    This function implements a two-stage fitting procedure:

    Stage 1 (local in T):
        For each target T in `target_T`, R(T, C) is fitted as a
        polynomial in C of degree `degree_s1`, with no intercept. This yields
        estimates of F₁(T), F₂(T), ..., F_degree_s1(T).

    Stage 2 (global in T):
        Each coefficient Fᵢ(T) obtained from Stage 1 is then fitted as a
        polynomial function of T, with degree specified independently by
        `degree_s2[i]`.

    The function also generates diagnostic plots:
        - R vs C curves at fixed T (Stage 1)
        - Fᵢ(T) vs T with fitted curves (Stage 2)
        - A text summary of fitted coefficients embedded in the figure

    Parameters
    ----------
    R_data : dict[float, array-like]
        Dictionary mapping control parameter values C to arrays of measured
        R(T, C) values. Each array corresponds to the same ordering as
        `T_data[C]`.

    T_data : dict[float, array-like]
        Dictionary mapping control parameter values C to arrays of 
        values T at which R(T, C) was evaluated.

    target_T : array-like
        List of target T values at which Stage-1 fits are performed.
        For each T_target, the nearest available T in `T_data[C]` is used.

    target_C : array-like
        List of control parameter values C to include in the Stage-1 fits.
    
    T_by_C : array-like
        To validate target T, C.


    degree_s1 : int, default=2
        Polynomial degree in C for Stage-1 fitting. The model enforces
        zero intercept, so the fitted basis is {C, C², ..., C^degree_s1}.

    degree_s2 : int or list of int, default=[2, 2]
        Polynomial degrees in T for Stage-2 fitting of each coefficient Fᵢ(T).
        If an integer is provided, the same degree is used for all Fᵢ.
        If a list is provided, its length must be at least `degree_s1`.

    figsize : tuple, default=(14, 16)
        Figure size passed to matplotlib.

    s1_alpha : float, default=0.6
        Transparency used for Stage-1 fitted R(T, C) curves.

    pltxt_loc : tuple, default=(0.5, 0.1)
        (x, y) location in figure coordinates for the summary text box.

    Returns
    -------
    df_s1 : pandas.DataFrame
        DataFrame containing Stage-1 fit results with columns:
        ['T_target', 'F1', 'F2', ..., 'F_degree_s1'].

    global_results : list of ndarray
        List of polynomial coefficient arrays for the Stage-2 fits.
        Each array corresponds to one Fᵢ(T), ordered by increasing
        power of T (constant term first).

    Notes
    -----
    - The function assumes nearest-neighbor selection in T when extracting
      R(T, C) values for Stage-1 fitting.
    - Over-parameterization can lead to unstable fits; lower-degree models
      should be preferred unless justified by the data.
    - The physical constraint R(T, C=0)=0 is enforced by construction.

    """
    # VALIDATION
    if isinstance(degree_s2, int):
        degree_s2 = [degree_s2] * degree_s1
    if len(degree_s2) < degree_s1:
        # If user provides [1, 2] for degree_s1=3, pad with the last value
        degree_s2 = degree_s2 + [degree_s2[-1]] * (degree_s1 - len(degree_s2))

    for c in target_C:
        if c not in R_data:
            raise KeyError(f"STRICT ERROR: Noise level C={c} not found in R_data.")
        if c not in T_by_C:
            raise KeyError(f"STRICT ERROR: Noise level C={c} not found in T_by_C reference.")

    for T_target in target_T:
        for c in target_C:
            # We convert to list for a direct search. 
            # Note: If floating point precision is an issue (e.g. 1.00000000001), 
            # this will correctly fail, forcing you to align your target_T.
            available_times = T_by_C[c].tolist() 
            
            if T_target not in available_times:
                raise ValueError(
                    f"STRICT ERROR: Target T={T_target} does not exist for noise level C={c}. "
                    f"Check your simulation logs for this specific configuration."
                )
                
    print(f"Validation Successful: All {len(target_T)} time points found for all {len(target_C)} noise levels.")
##################################################################################################
# 2. STAGE 1: R(T, C) fitting
    fit_results = []
    exact_points_by_T = {} 
    
    for T_target in target_T:
        C_slice, R_slice = [], []
        
        for c in target_C:
            # We already validated these keys and T values exist.
            # We find the index once and use it directly.
            T_vals_list = T_by_C[c].tolist()
            idx = T_vals_list.index(T_target)
            
            C_slice.append(c)
            R_slice.append(R_data[c][idx])

        # Perform the Stage 1 fit (R vs C)
        # Note: We checked C_slice length during validation logic implicitly 
        # by ensuring all target_C are present.
        coeffs = rtc.fit_stage1(C_slice, R_slice, degree=degree_s1, include_intercept=False)
        
        # Store results: [T, F1, F2, ...]
        fit_results.append([T_target] + list(coeffs))
        
        # For visualization purposes
        exact_points_by_T[T_target] = (np.array(C_slice), np.array(R_slice))

    # Create Stage 1 DataFrame
    cols = ['T_target'] + [f'F{i+1}' for i in range(degree_s1)]
    df_s1 = pd.DataFrame(fit_results, columns=cols).sort_values('T_target').reset_index(drop=True)

    # 3. STAGE 2: F(T) fitting
    T_final = df_s1['T_target'].values
    global_results = []
    for i in range(degree_s1):
        Fi_vals = df_s1[f'F{i+1}'].values
        # Fit Fi vs T with specific degree
        g_coeffs = rtc.fit_stage2(T_final, Fi_vals, degree=degree_s2[i], include_intercept=True)
        global_results.append(g_coeffs)

    # 4. CONSTRUCT TEXT OUTPUT
    txt = "FIT 1:  R(T,C) (R vs C)\n" + "-"*90 + "\n"
    txt += rf"R(T,C) = C * F_1(T) + C^2 * F_2(T) + C^3 * F_3(T)... with order of C: {degree_s1}" + "\n\n"
    txt += df_s1.to_string(index=False, formatters={c: "{:.4e}".format for c in df_s1.columns if 'F' in c})
    txt += "\n\nFIT 2: Fi  (Fi vs T)\n" + "-"*90 + "\n"
    for i, g in enumerate(global_results):
        terms = [f"({v:.3e})*T^{j}" for j, v in enumerate(g)]
        txt += f"F{i+1}(T) [deg {degree_s2[i]}] = {' + '.join(terms)}\n"
##################################################################################################
    # 5. VISUALIZATION
    #fig = plt.figure(figsize=figsize)
    # 5. VISUALIZATION (Compact Layout)

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # 3 rows:
    # row 0 → Stage 1
    # row 1 → Stage 2
    # row 2 → Text box
    gs = fig.add_gridspec(
        nrows=3,
        ncols=degree_s1,
        height_ratios=[0.45, 0.45, 0.10]  # compact vertical balance
    )

    # ======================
    # Top Plot (Stage 1)
    # ======================
    ax_s1 = fig.add_subplot(gs[0, :])

    C_range = np.linspace(0, max(target_C) * 1.05, 200)

    for T_t, (cp, rp) in exact_points_by_T.items():
        sc = ax_s1.scatter(cp, rp, s=40, edgecolors='k', linewidth=0.5)
        color = sc.get_facecolor()[0]

        row_coeffs = df_s1[df_s1['T_target'] == T_t].iloc[0, 1:].values
        R_line = sum(c_val * (C_range**(idx+1))
                    for idx, c_val in enumerate(row_coeffs))

        ax_s1.plot(C_range, R_line, color=color, alpha=s1_alpha, lw=2)

    ax_s1.set_title(f"R(T,C) fitting (degree={degree_s1})", pad=6)
    ax_s1.set_xlabel("C")
    ax_s1.set_ylabel("R(T,C)")
    ax_s1.legend([f"T={T}" for T in exact_points_by_T.keys()],
                ncol=min(4, len(exact_points_by_T)),
                fontsize=9,
                frameon=False)

    # ======================
    # Bottom Plots (Stage 2)
    # ======================
    T_range = np.linspace(min(target_T)-5, max(target_T)+5, 200)

    for i in range(degree_s1):
        ax = fig.add_subplot(gs[1, i])

        ax.scatter(T_final,
                df_s1[f'F{i+1}'],
                edgecolors='k',
                linewidth=0.5)

        Fi_line = sum(val * (T_range**j)
                    for j, val in enumerate(global_results[i]))

        ax.plot(T_range, Fi_line, 'k--', lw=1.5, label=f"deg {degree_s2[i]} Fit")
        ax.set_xlabel(r"$T_{max}$", fontsize=9)
        ax.set_ylabel(fr"$F_{i+1}(T)$", fontsize=9)
        ax.legend(fontsize="x-small")
    # ======================
    # Dedicated Text Panel
    # ======================
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")

    ax_txt.text(
        0.5, 0.5,
        txt,
        fontsize=8,
        family='monospace',
        ha='center',
        va='center',
        bbox=dict(boxstyle='round', facecolor='white',
                edgecolor='black', linewidth=0.8)
    )

    plt.show()
    return df_s1, global_results