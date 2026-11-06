"""Pointwise neural field on a pressure-coefficient-like target.

Linear reduced-order models (POD + regression) cannot move a shock to a
position no training field has, which caps their error when extrapolating in
Mach. Here a network maps (point, flight conditions) -> wall value, so the
shock location is a smooth function of the conditions.

Target: rho / rho_inf of an adiabatic wall is roughly (p / p_inf) / tau, with
a nearly uniform recovery temperature tau = 1 + r (gamma - 1) / 2 M^2. We
learn Cp = (rho * tau - 1) / (gamma / 2 M^2), optionally as a residual over a
POD + Gaussian-process prediction, with an L1 loss weighted exactly like the
competition metric (component weights, per-run relative scale, half weight
at |AoA| >= 10).
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from benchmark_utils.base_solver import CRMSolver

GAMMA = 1.4
RECOVERY = 0.89
COMPONENT_WEIGHTS = {"wing": 0.3, "pylon": 0.3, "fuselage": 0.2,
                     "nacelle": 0.2}

CONFIG = dict(
    use_base=False,       # learn a residual over POD-GP
    steps=15000,
    batch=16384,
    lr=2e-3,
    n_fourier=64,
    sigma=8.0,
    width=256,
    depth=5,
    n_models=3,           # seed ensemble, predictions averaged
    time_budget=35 * 60,  # seconds of NN training in total, hard stop
    seed=0,
)


def _tau(mach):
    return 1 + RECOVERY * (GAMMA - 1) / 2 * mach ** 2


def to_cp(conditions, rho):
    m = conditions[:, :1]
    return (rho * _tau(m) - 1) / (GAMMA / 2 * m ** 2)


def from_cp(conditions, cp):
    m = conditions[:, :1]
    return (cp * (GAMMA / 2 * m ** 2) + 1) / _tau(m)


def point_weights(components, component_names):
    w = np.zeros(len(components))
    for cid, name in enumerate(component_names):
        mask = components == cid
        if name in COMPONENT_WEIGHTS and mask.any():
            w[mask] = COMPONENT_WEIGHTS[name] / mask.sum()
    return w / w.sum()


def cond_features(conditions):
    c = np.asarray(conditions, np.float64)
    return np.c_[(c[:, 0] - 0.75) / 0.2, c[:, 1] / 8, np.log(c[:, 2])]


class CpPODGP:

    def __init__(self, n_modes=10, nu=1.5):
        self.n_modes, self.nu = n_modes, nu

    def fit(self, conditions, cp):
        self.mu, self.sd = conditions.mean(0), conditions.std(0)
        self.pod = PCA(self.n_modes, random_state=0)
        coefs = self.pod.fit_transform(cp)
        kernel = (ConstantKernel() * Matern([1.0] * 3, nu=self.nu)
                  + WhiteKernel(1e-4))
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True,
                                           n_restarts_optimizer=2,
                                           random_state=0)
        self.gp.fit((conditions - self.mu) / self.sd, coefs)
        return self

    def predict_cp(self, conditions):
        z = self.gp.predict((conditions - self.mu) / self.sd)
        return self.pod.inverse_transform(z.reshape(len(conditions), -1))


class Net(nn.Module):

    def __init__(self, n_fourier, sigma, width, depth, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(3, n_fourier, generator=g)
                             * sigma)
        layers, d = [], 2 * n_fourier + 9
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, nrm, cond):
        proj = 2 * math.pi * xyz @ self.B
        h = torch.cat([torch.sin(proj), torch.cos(proj), xyz, nrm, cond], -1)
        return self.mlp(h).squeeze(-1)


class NeuralField:

    def __init__(self, meta, **cfg):
        self.cfg = {**CONFIG, **cfg}
        self.dev = torch.device("cuda" if torch.cuda.is_available() else
                                "mps" if torch.backends.mps.is_available()
                                else "cpu")
        geo = np.asarray(meta["geometry"], np.float64)
        lo, hi = geo[:, :3].min(0), geo[:, :3].max(0)
        self.xyz = self._t(2 * (geo[:, :3] - lo) / (hi - lo) - 1)
        self.nrm = self._t(geo[:, 3:])
        self.w = point_weights(np.asarray(meta["components"]),
                               meta["component_names"])
        self.base = None

    def _t(self, a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(
            self.dev)

    def fit(self, conditions, fields):
        cfg = self.cfg
        c = np.asarray(conditions, np.float64)
        f = np.asarray(fields, np.float64)
        y = to_cp(c, f)
        if cfg["use_base"]:
            self.base = CpPODGP().fit(c, y)
            y = y - self.base.predict_cp(c)
        y_t = self._t(y)
        del y

        # Metric-matched weights: |d rho| = |d Cp| * (gamma/2 M^2) / tau,
        # relative to the run's weighted mean density.
        m = c[:, 0]
        scale = (GAMMA / 2 * m ** 2) / _tau(m) / (f @ self.w)
        scale *= np.where(np.abs(c[:, 1]) >= 10, 0.5, 1.0)
        run_w = self._t(scale / scale.mean())
        pts_p = self._t(self.w)
        cond = self._t(cond_features(c))

        self.nets = []
        start = time.time()
        for k in range(cfg["n_models"]):
            budget = (cfg["time_budget"] - (time.time() - start)) / (
                cfg["n_models"] - k)
            self.nets.append(self._train_one(cfg["seed"] + k, y_t, cond,
                                             run_w, pts_p, budget))
        del y_t
        return self

    def _train_one(self, seed, y_t, cond, run_w, pts_p, budget):
        cfg = self.cfg
        torch.manual_seed(seed)
        net = Net(cfg["n_fourier"], cfg["sigma"], cfg["width"],
                  cfg["depth"], seed).to(self.dev)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, cfg["lr"], total_steps=cfg["steps"], pct_start=0.05)
        n_runs, start = len(cond), time.time()
        for it in range(cfg["steps"]):
            ri = torch.randint(0, n_runs, (cfg["batch"],), device=self.dev)
            pi = torch.multinomial(pts_p, cfg["batch"], replacement=True)
            out = net(self.xyz[pi], self.nrm[pi], cond[ri])
            loss = (run_w[ri] * (out - y_t[ri, pi]).abs()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if it % 1000 == 0:
                print(f"[nn seed {seed}] it {it} loss {loss.item():.4f} "
                      f"{time.time() - start:.0f}s", flush=True)
            if time.time() - start > budget:
                print(f"[nn seed {seed}] time budget hit at it {it}",
                      flush=True)
                break
        return net.eval()

    @torch.no_grad()
    def predict(self, conditions):
        c = np.asarray(conditions, np.float64)
        cond = self._t(cond_features(c))
        n, chunk = len(self.xyz), 131072
        out = np.empty((len(c), n))
        for k in range(len(c)):
            for i in range(0, n, chunk):
                xyz = self.xyz[i:i + chunk]
                ck = cond[k].expand(len(xyz), 3)
                out[k, i:i + chunk] = torch.stack(
                    [net(xyz, self.nrm[i:i + chunk], ck) for net in self.nets]
                ).mean(0).cpu().numpy()
        if self.base is not None:
            out += self.base.predict_cp(c)
        return from_cp(c, out)


class Solver(CRMSolver):

    name = "submission"

    def get_model(self, meta):
        return NeuralField(meta)

    def fit(self, model, conditions, fields):
        model.fit(conditions, fields)
