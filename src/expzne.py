import numpy as np


def expzne(costs: List[float], noise_param: List[float])-> Dict:
    """
    Zero noise extrapolation using exonential fitting.
    
    Args:
        costs (List[float]): List of costs measured at different noise levels.
        noise_param (List[float]): List of noise parameters corresponding to the costs.
        
        Returns:
            Dict: A dictionary containing the extrapolated zero-noise cost and fitting parameters.
    """

    length: int = len(costs)
    if length != len(noise_param):
        raise ValueError("Length of costs and noise_param must be the same.")
    
    