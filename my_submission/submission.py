"""POD + Gaussian process on a pressure-coefficient-like target.

The wall density of an adiabatic wall is roughly the wall pressure divided by
a nearly uniform recovery temperature, ``rho / rho_inf ~ (p / p_inf) / tau``
with ``tau = 1 + r (gamma - 1) / 2 M^2``. We therefore regress the
pressure-coefficient-like quantity

    Cp = (rho * tau - 1) / (gamma / 2 M^2),

which varies far more regularly with Mach than the raw density, and map the
prediction back. The POD coefficients of Cp are regressed on the conditions
with a Matern 3/2 Gaussian process (rougher than RBF, it extrapolates less
wildly across Mach lines) on (Mach, AoA) only: the stagnation pressure Pi
changes rho / rho_inf by ~0.5 %, the level of the solver noise, so dropping it
lets the GP fit the two inputs that matter.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from benchmark_utils.base_solver import CRMSolver

GAMMA = 1.4
RECOVERY = 0.89


def _tau(mach):
    return 1 + RECOVERY * (GAMMA - 1) / 2 * mach ** 2


def to_cp(conditions, rho):
    m = conditions[:, :1]
    return (rho * _tau(m) - 1) / (GAMMA / 2 * m ** 2)


def from_cp(conditions, cp):
    m = conditions[:, :1]
    return (cp * (GAMMA / 2 * m ** 2) + 1) / _tau(m)


class CpPODGP:

    def __init__(self, n_modes=10, nu=1.5):
        self.n_modes, self.nu = n_modes, nu

    def _x(self, conditions):
        # (Mach, AoA) only: Pi is left out on purpose.
        return (np.asarray(conditions, np.float64)[:, :2] - self.mu) / self.sd

    def fit(self, conditions, fields):
        conditions = np.asarray(conditions, np.float64)
        self.mu, self.sd = conditions[:, :2].mean(0), conditions[:, :2].std(0)
        cp = to_cp(conditions, np.asarray(fields, np.float64))
        self.pod = PCA(self.n_modes, random_state=0)
        coefs = self.pod.fit_transform(cp)
        kernel = (ConstantKernel() * Matern([1.0] * 2, nu=self.nu)
                  + WhiteKernel(1e-4))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True,
                                           n_restarts_optimizer=2,
                                           random_state=0)
        self.gp.fit(self._x(conditions), coefs)
        return self

    def predict(self, conditions):
        conditions = np.asarray(conditions, np.float64)
        coefs = self.gp.predict(self._x(conditions)).reshape(len(conditions), -1)
        return from_cp(conditions, self.pod.inverse_transform(coefs))


class Solver(CRMSolver):

    name = "submission"

    def get_model(self, meta):
        return CpPODGP(n_modes=10, nu=1.5)

    def fit(self, model, conditions, fields):
        model.fit(conditions, fields)
