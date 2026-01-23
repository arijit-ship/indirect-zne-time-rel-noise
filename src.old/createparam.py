import numpy as np
from typing import Dict, Tuple, List


def create_param(
    layers: int,
    gateset: int,
    ti: float,
    tf: float,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Create initial parameters for Big-T VQE.

    Parameter structure:
    --------------------
    Time parameters:
        - t0 is fixed to 0 (NOT optimized)
        - We optimize (layers - 1) intermediate times
        - Final time is enforced externally as T_max

    Rotation parameters:
        - For each layer: 2 * gateset parameters (RX, RY)

    Total parameters:
        (layers - 1) + layers * (2 * gateset)

    Returns
    -------
    param : np.ndarray
        Flat parameter array for optimizer

    param_dict : dict
        Human-readable breakdown
    """

    if layers < 1:
        raise ValueError("layers must be >= 1")

    # ---- Time params (excluding t0 and T_max) ----
    if layers > 1:
        time_params = np.sort(np.random.uniform(ti, tf, layers - 1))
    else:
        time_params = np.array([])

    # ---- Rotation params ----
    rot_params = np.random.uniform(
        low=-np.pi,
        high=np.pi,
        size=layers * 2 * gateset
    )

    param = np.concatenate([time_params, rot_params])

    param_dict = {
        "time": time_params.tolist(),
        "rotation": rot_params.tolist(),
    }

    return param, param_dict
