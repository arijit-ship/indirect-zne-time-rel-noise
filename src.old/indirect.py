import os
from datetime import datetime
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from qulacs import DensityMatrix, Observable, QuantumCircuit, QuantumState
from qulacsvis import circuit_drawer
from scipy.optimize import minimize

from src.ansatz import create_noisy_ansatz, noiseless_ansatz
from src.constraint import create_time_constraints, create_tf_fixed_constraint
from src.createparam import create_param
from src.hamiltonian import (
    create_heisenberg_hamiltonian,
    create_ising_hamiltonian,
    create_xy_hamiltonian,
    create_xy_iss_hamiltonian,
)
from src.modules import calculate_noise_levels

class IndiectVQE:
    def __init__(
        self,
        hamiltonian: Observable,
        ansatz_config: Dict,
        optimization_config: Dict,
        noise_config: Dict,
        iteration: int,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.ansatz_config = ansatz_config
        self.optimization_config = optimization_config
        self.noise_config = noise_config
        self.iteration = iteration
        self.noise_levels = calculate_noise_levels(noise_config)
        self.cost_history: List[float] = []
        self.optimized_parameters: np.ndarray = np.array([])