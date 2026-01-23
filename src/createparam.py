import numpy as np


def create_param(layer: int, gateset: int, ti: float, tf: float) -> np.ndarray:
    """
    Creates initial parameters for the citcuit. Parameters are time, and theta: angle for rotation gates.

    time: 0 - max time
    theta: 0 - 1

    Args:
        layer (int): number of layer.
        gateset (int): Number of rotation gate set. Each set has 4 gates.
        ti (float): initial time
        tf (float): final time

    Returns:
        numpy.ndarray: An nd-array [
        t1, t2, ... td, theta1, ..., theatd * 4
    ]

    """
    param = np.array([])
    # Time param
    time = np.random.uniform(ti, tf, layer)
    time = np.sort(time)  # Time must be in incresing order
    for i in time:
        param = np.append(param, i)
    #param= np.append(param, tf)  # Last time is always tf

    # Theta param
    theta_len = layer * gateset * 2
    theta = np.random.random(theta_len) * 1e-1  # Each layer has 4 rotation gates
    for i in theta:
        param = np.append(param, i)
    print("Initial parameter created.", time, theta)
    return param, {"time-len": layer, "time": time.tolist(), "theta-len": theta_len,"theta": theta.tolist()}
