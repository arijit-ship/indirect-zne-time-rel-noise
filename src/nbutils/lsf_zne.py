import numpy as np
import numpy.typing as npt
from typing import List, Union, Literal, Optional, Dict, Any
from scipy.optimize import curve_fit

NumericData = Union[List[float], npt.NDArray[np.float64]]

class LstZne:
    def __init__(
        self, 
        vqe_estimations: NumericData, 
        noise_levels: NumericData, 
        stds: Optional[NumericData] = None,
        model_str: Literal['exp_single', 'exp_double', 'exp_offset', 'logistic_decay', 'logistic_shifted', 'logistic_offset_shifted'] = 'exp_single',
        xtol: float = 1e-8,
        ftol: float = 1e-8,
        gtol: float = 1e-8
    ) -> None:
        self.y = np.array(vqe_estimations, dtype=np.float64)
        self.x = np.array(noise_levels, dtype=np.float64)
        self.sigma = np.array(stds, dtype=np.float64) if stds is not None else None
        self.model_str = model_str.lower()
        
        # Default tolerances
        self.xtol, self.ftol, self.gtol = xtol, ftol, gtol
        
        # Tracking results
        self.params: Optional[npt.NDArray[np.float64]] = None
        self.coefficients: Dict[str, float] = {}
        self.zne_value: Optional[float] = None
        self.history: Dict[str, Dict[str, Any]] = {}

    def _stable_exp(self, exponent: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.exp(np.clip(exponent, -500, 500))

    def _models(self, t: npt.NDArray[np.float64], *args: float) -> npt.NDArray[np.float64]:
        if self.model_str == 'exp_single':
            O0, gamma = args
            return O0 * self._stable_exp(-gamma * t)
        elif self.model_str == 'exp_double':
            O1, g1, O2, g2 = args
            return O1 * self._stable_exp(-g1 * t) + O2 * self._stable_exp(-g2 * t)
        elif self.model_str == 'exp_offset':
            alpha, beta, gamma = args
            return alpha + (beta * self._stable_exp(-gamma * t))
        elif self.model_str == 'logistic_decay':
            K, N, r = args
            return (-N * K) / ((K - N) * self._stable_exp(-r * t) + N)
        elif self.model_str == 'logistic_shifted':
            K, N, r, t0 = args
            return (-N * K) / ((K - N) * self._stable_exp(-r * (t - t0)) + N)
        elif self.model_str == 'logistic_offset_shifted':
            K, N, r, t0, C = args
            return C - (N * K / ((K - N) * self._stable_exp(-r * (t - t0)) + N))
        raise ValueError(f"Model {self.model_str} not supported.")

    def fit(self, p0: Optional[List[float]] = None, diff_step: Optional[List[float]] = None, **kwargs) -> float:
        idx = np.argsort(self.x)
        x_s, y_s = self.x[idx], self.y[idx]
        sig_s = self.sigma[idx] if self.sigma is not None else None

        # Determine Names and Bounds
        if 'logistic' in self.model_str:
            if self.model_str == 'logistic_decay':
                p_def, names = [-y_s[-1], -y_s[0], 0.1], ['K', 'N', 'r']
                b = ([-np.inf, -np.inf, 0], [np.inf, np.inf, 10])
            elif self.model_str == 'logistic_shifted':
                p_def, names = [-y_s[-1], -y_s[0], 0.1, 0.0], ['K', 'N', 'r', 't0']
                b = ([-np.inf, -np.inf, 0, -100], [np.inf, np.inf, 10, 100])
            else:
                p_def, names = [-y_s[-1], -y_s[0], 0.1, 0.0, 0.0], ['K', 'N', 'r', 't0', 'C']
                b = ([-np.inf, -np.inf, 0, -100, -np.inf], [np.inf, np.inf, 10, 100, np.inf])
        elif self.model_str == 'exp_offset':
            p_def, names = [y_s[-1], y_s[0]-y_s[-1], 0.1], ['alpha', 'beta', 'gamma']
            b = ([-np.inf, -np.inf, 0], [np.inf, np.inf, 10])
        elif self.model_str == 'exp_double':
            p_def, names = [y_s[0]/2, 0.1, y_s[0]/2, 0.01], ['O1', 'g1', 'O2', 'g2']
            b = ([-np.inf, 0, -np.inf, 0], [np.inf, 10, np.inf, 10])
        else: # exp_single
            p_def, names = [y_s[0], 0.1], ['O0', 'gamma']
            b = ([-np.inf, 0], [np.inf, 10])

        # Merge solver configs
        conf = {
            'xtol': kwargs.get('xtol', self.xtol),
            'ftol': kwargs.get('ftol', self.ftol),
            'gtol': kwargs.get('gtol', self.gtol),
            'diff_step': diff_step,
            'maxfev': kwargs.get('maxfev', 50000)
        }

        popt, _ = curve_fit(self._models, x_s, y_s, p0=(p0 or p_def), 
                            sigma=sig_s, absolute_sigma=(sig_s is not None), 
                            bounds=b, **conf)
        
        self.params = popt
        self.coefficients = dict(zip(names, popt))
        
        # ZNE Calculation Logic
        if self.model_str in ['logistic_decay', 'logistic_shifted']:
            self.zne_value = -float(self.coefficients['K'])
        elif self.model_str == 'logistic_offset_shifted':
            self.zne_value = float(self.coefficients['C'] - self.coefficients['K'])
        else:
            self.zne_value = self.predict(0.0)

        # Store in history for later access
        self.history[self.model_str] = {'params': popt, 'zne': self.zne_value, 'coeffs': self.coefficients}
        return self.zne_value

    def predict(self, t_val: float) -> float:
        return float(self._models(np.array([t_val]), *self.params)[0])