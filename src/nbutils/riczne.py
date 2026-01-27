from typing import List, Dict


def RichardsonZne(costs: List[float], noise_param: List[float]) -> Dict:
    """
    Perform single-variable Richardson zero-noise extrapolation.

    Args:
        costs (List[float]):
            Measured expectation values at different noise levels.
        noise_param (List[float]):
            Corresponding noise scale factors (must be same length as costs).

    Returns:
        Dict containing:
            - extrapolated_val: zero-noise estimate
            - betas: Richardson coefficients
            - sorted_noise: sorted noise parameters
            - sorted_costs: costs reordered to match sorted noise
            - cost_richardson_zne: sum(beta_k^2), mitigation cost metric
    """
    if len(costs) != len(noise_param):
        raise ValueError("costs and noise_param must have the same length.")

    if len(costs) < 2:
        raise ValueError("At least two noise points are required for Richardson extrapolation.")

    # Sort by noise parameter
    sorted_pairs = sorted(zip(noise_param, costs), key=lambda x: x[0])
    sorted_noise, sorted_costs = map(list, zip(*sorted_pairs))

    n = len(sorted_noise)
    betas = []

    # Compute Richardson (Lagrange) coefficients
    for k in range(n):
        alpha_k = sorted_noise[k]
        beta_k = 1.0
        for i in range(n):
            if i == k:
                continue
            alpha_i = sorted_noise[i]
            if alpha_k == alpha_i:
                raise ValueError("Noise parameters must be distinct.")
            beta_k *= alpha_i / (alpha_k - alpha_i)
        betas.append(beta_k)

    # Normalize betas so sum(beta) = 1
    beta_sum = sum(betas)
    betas = [b / beta_sum for b in betas]

    # Zero-noise extrapolated value
    zne_value = sum(betas[i] * sorted_costs[i] for i in range(n))

    # Cost of error mitigation (variance amplification proxy)
    cost_error_mitigation = sum(b * b for b in betas)

    return {
        "extrapolated_val": zne_value,
        "sorted_noise": sorted_noise,
        "sorted_costs": sorted_costs,
        "betas": betas,
        "cost_richardson_zne": cost_error_mitigation,
    }
