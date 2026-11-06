"""Graph neural network on the surface mesh (MeshGraphNet-style).

The mesh connectivity is not given, so we rebuild it: each point is linked to
its k nearest neighbours, made symmetric, and edges whose normals point in
opposite directions are dropped (otherwise the thin trailing edge would link
the upper and lower wing surfaces, which see very different flows).

Model, for one run:
    node input  = Fourier features of (x, y, z) + coords + normal
                  + local mesh spacing + flight conditions (Mach, AoA)
    edge input  = relative position / local spacing, distance, normal dot
    encoder MLP -> L message-passing blocks (edge MLP, mean aggregation,
    node MLP, residual + LayerNorm, conditions re-injected) -> decoder MLP
    output      = Cp-like target, mapped back to density.

Without the message passing this is the pointwise neural field that already
beat the POD-GP in cross-validation; the graph blocks let neighbouring points
exchange information (shock sharpness, consistency along the surface).

Target: rho / rho_inf of an adiabatic wall ~ (p / p_inf) / tau with
tau = 1 + r (gamma - 1) / 2 M^2, so we learn Cp = (rho tau - 1) / (gamma/2 M^2)
with an L1 loss weighted like the competition metric. Pi is not an input: it
changes the density by ~0.5 %, the level of the solver noise.

Training uses random spatial crops of the graph (the K points nearest to a
centre drawn with the metric's point weights), several runs per step, and the
loss on the crop interior only, whose receptive field is not cut. Prediction
runs on the full graph. Training stops on a wall-clock budget, with the
learning-rate schedule following elapsed time, so the 1 h limit is safe.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from benchmark_utils.base_solver import CRMSolver

GAMMA = 1.4
RECOVERY = 0.89
COMPONENT_WEIGHTS = {"wing": 0.3, "pylon": 0.3, "fuselage": 0.2,
                     "nacelle": 0.2}

CONFIG = dict(
    # graph
    k=8,                  # nearest neighbours per point
    min_normal_dot=-0.3,  # drop edges between opposite-facing points
    # model
    hidden=64,
    n_layers=6,
    n_fourier=48,
    sigma=8.0,            # Fourier feature scale on normalised coords
    # training
    crop=12000,           # points per training crop
    interior=0.6,         # loss on the nearest 60 % of the crop
    runs_per_step=4,
    steps=30000,          # upper bound; the time budget usually stops first
    lr=1e-3,
    warmup=0.03,
    time_budget=38 * 60,  # seconds of training, hard stop
    amp=False,            # fp16 autocast on CUDA (faster on T4, less tested)
    edge_chunk=1_000_000, # edges per chunk at full-graph prediction
    seed=0,
)


# --- physics transform -------------------------------------------------

def _tau(mach):
    return 1 + RECOVERY * (GAMMA - 1) / 2 * mach ** 2


def to_cp(conditions, rho):
    m = conditions[:, :1]
    return (rho * _tau(m) - 1) / (GAMMA / 2 * m ** 2)


def from_cp(conditions, cp):
    m = conditions[:, :1]
    return (cp * (GAMMA / 2 * m ** 2) + 1) / _tau(m)


def cond_features(conditions):
    c = np.asarray(conditions, np.float64)
    return np.c_[(c[:, 0] - 0.75) / 0.2, c[:, 1] / 8]


def point_weights(components, component_names):
    w = np.zeros(len(components))
    for cid, name in enumerate(component_names):
        mask = components == cid
        if name in COMPONENT_WEIGHTS and mask.any():
            w[mask] = COMPONENT_WEIGHTS[name] / mask.sum()
    return w / w.sum()


# --- graph -------------------------------------------------------------

def build_graph(xyz, nrm, k, min_normal_dot):
    """Symmetric kNN graph; returns (src, dst, local spacing per node)."""
    n = len(xyz)
    dist, idx = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    spacing = np.maximum(dist[:, 1:].mean(1), 1e-6)
    src = idx[:, 1:].ravel()
    dst = np.repeat(np.arange(n), k)
    src, dst = np.r_[src, dst], np.r_[dst, src]
    keep = (src != dst) & ((nrm[src] * nrm[dst]).sum(1) > min_normal_dot)
    key = np.unique(dst[keep].astype(np.int64) * n + src[keep])
    return key % n, key // n, spacing


# --- network -----------------------------------------------------------

def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(),
                         nn.Linear(d_hidden, d_hidden), nn.SiLU(),
                         nn.Linear(d_hidden, d_out))


class Block(nn.Module):

    def __init__(self, hidden, d_cond):
        super().__init__()
        self.msg = mlp(3 * hidden, hidden, hidden)
        self.upd = mlp(2 * hidden + d_cond, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, e, cond, src, dst, inv_deg, edge_chunk):
        # h (B, n, H), e (E, H), cond (B, C), src/dst (E,), inv_deg (n, 1)
        B, n, _ = h.shape
        agg = None
        for i in range(0, len(src), edge_chunk):
            s, d = src[i:i + edge_chunk], dst[i:i + edge_chunk]
            m = self.msg(torch.cat(
                [h[:, d], h[:, s], e[i:i + edge_chunk].expand(B, -1, -1)],
                -1))
            if agg is None:
                agg = m.new_zeros(B, n, m.shape[-1])
            agg = agg.index_add(1, d, m.to(agg.dtype))
        agg = agg * inv_deg
        c = cond[:, None, :].expand(B, n, -1)
        u = self.upd(torch.cat([h, agg.to(h.dtype), c.to(h.dtype)], -1))
        return h + self.norm(u)


class GraphNet(nn.Module):

    def __init__(self, d_node, d_cond, d_edge, hidden, n_layers):
        super().__init__()
        self.enc = mlp(d_node + d_cond, 2 * hidden, hidden)
        self.edge_enc = mlp(d_edge, hidden, hidden)
        self.blocks = nn.ModuleList(Block(hidden, d_cond)
                                    for _ in range(n_layers))
        self.dec = mlp(hidden, hidden, 1)

    def forward(self, node, cond, src, dst, edge, inv_deg, edge_chunk):
        # node (n, D) static features, cond (B, C) one row per run
        B, n = cond.shape[0], node.shape[0]
        h = self.enc(torch.cat([node.expand(B, n, -1),
                                cond[:, None, :].expand(B, n, -1)], -1))
        e = self.edge_enc(edge)
        for block in self.blocks:
            h = block(h, e, cond, src, dst, inv_deg, edge_chunk)
        return self.dec(h).squeeze(-1)


# --- model -------------------------------------------------------------

class GraphModel:

    def __init__(self, meta, **cfg):
        self.cfg = cfg = {**CONFIG, **cfg}
        self.dev = torch.device("cuda" if torch.cuda.is_available() else
                                "mps" if torch.backends.mps.is_available()
                                else "cpu")
        t0 = time.time()
        geo = np.asarray(meta["geometry"], np.float64)
        xyz, nrm = geo[:, :3], geo[:, 3:]
        self.n = len(xyz)
        self.w = point_weights(np.asarray(meta["components"]),
                               meta["component_names"])

        src, dst, spacing = build_graph(xyz, nrm, cfg["k"],
                                        cfg["min_normal_dot"])
        rel = (xyz[src] - xyz[dst]) / spacing[dst, None]
        edge = np.c_[rel, np.linalg.norm(rel, axis=1),
                     (nrm[src] * nrm[dst]).sum(1)]

        lo, hi = xyz.min(0), xyz.max(0)
        xyz_n = 2 * (xyz - lo) / (hi - lo) - 1
        g = np.random.default_rng(cfg["seed"])
        B = g.normal(size=(3, cfg["n_fourier"])) * cfg["sigma"]
        proj = 2 * math.pi * xyz_n @ B
        node = np.c_[np.sin(proj), np.cos(proj), xyz_n, nrm,
                     np.log(spacing / np.median(spacing))]

        self.xyz = self._t(xyz)
        self.node = self._t(node)
        self.src = torch.from_numpy(src).to(self.dev)
        self.dst = torch.from_numpy(dst).to(self.dev)
        self.edge = self._t(edge)
        deg = np.bincount(dst, minlength=self.n)
        self.inv_deg = self._t(1 / np.maximum(deg, 1))[:, None]
        print(f"[gnn] graph: {self.n} nodes, {len(src)} edges, "
              f"mean degree {len(src) / self.n:.1f}, "
              f"{(deg == 0).sum()} isolated, {time.time() - t0:.1f}s",
              flush=True)

    def _t(self, a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(
            self.dev)

    def _crop(self, pts_p):
        """Nodes of a random crop (nearest first) and its local subgraph."""
        K = self.cfg["crop"]
        centre = torch.multinomial(pts_p, 1)
        d2 = ((self.xyz - self.xyz[centre]) ** 2).sum(1)
        nodes = torch.topk(d2, K, largest=False, sorted=True).indices
        lut = torch.full((self.n,), -1, dtype=torch.long, device=self.dev)
        lut[nodes] = torch.arange(K, device=self.dev)
        ls, ld = lut[self.src], lut[self.dst]
        keep = (ls >= 0) & (ld >= 0)
        src, dst = ls[keep], ld[keep]
        deg = torch.bincount(dst, minlength=K).clamp(min=1)
        inv_deg = (1.0 / deg.float())[:, None]
        return nodes, src, dst, self.edge[keep], inv_deg

    def fit(self, conditions, fields):
        cfg = self.cfg
        torch.manual_seed(cfg["seed"])
        c = np.asarray(conditions, np.float64)
        f = np.asarray(fields, np.float64)
        y_t = self._t(to_cp(c, f))

        # Metric-matched run weights: |d rho| = |d Cp| (gamma/2 M^2) / tau,
        # relative to the run's weighted mean density, half at |AoA| >= 10.
        m = c[:, 0]
        scale = (GAMMA / 2 * m ** 2) / _tau(m) / (f @ self.w)
        scale *= np.where(np.abs(c[:, 1]) >= 10, 0.5, 1.0)
        run_w = self._t(scale / scale.mean())
        cond = self._t(cond_features(c))
        pts_p = self._t(self.w)
        del f

        self.net = GraphNet(self.node.shape[1], cond.shape[1],
                            self.edge.shape[1], cfg["hidden"],
                            cfg["n_layers"]).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg["lr"],
                                weight_decay=1e-5)
        amp = cfg["amp"] and self.dev.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        n_int = int(cfg["crop"] * cfg["interior"])
        n_runs, R = len(c), cfg["runs_per_step"]
        start = time.time()
        for it in range(cfg["steps"]):
            progress = max(it / cfg["steps"],
                           (time.time() - start) / cfg["time_budget"])
            if progress >= 1:
                print(f"[gnn] stop at it {it} ({time.time() - start:.0f}s)",
                      flush=True)
                break
            if progress < cfg["warmup"]:
                lr = cfg["lr"] * progress / cfg["warmup"]
            else:
                p = (progress - cfg["warmup"]) / (1 - cfg["warmup"])
                lr = cfg["lr"] * (0.02 + 0.98 * 0.5 * (1 + math.cos(
                    math.pi * p)))
            for group in opt.param_groups:
                group["lr"] = lr

            nodes, src, dst, edge, inv_deg = self._crop(pts_p)
            ri = torch.randint(0, n_runs, (R,), device=self.dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                pred = self.net(self.node[nodes], cond[ri], src, dst, edge,
                                inv_deg, cfg["edge_chunk"])
            pred = pred[:, :n_int].float()
            target = y_t[ri[:, None], nodes[None, :n_int]]
            loss = (run_w[ri, None] * (pred - target).abs()).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if it % 500 == 0:
                print(f"[gnn] it {it} loss {loss.item():.4f} lr {lr:.1e} "
                      f"{time.time() - start:.0f}s", flush=True)
        self.net.eval()
        del y_t
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def predict(self, conditions):
        c = np.asarray(conditions, np.float64)
        cond = self._t(cond_features(c))
        out = np.empty((len(c), self.n))
        for k in range(len(c)):
            out[k] = self.net(self.node, cond[k:k + 1], self.src, self.dst,
                              self.edge, self.inv_deg,
                              self.cfg["edge_chunk"])[0].float().cpu().numpy()
        return from_cp(c, out)


class Solver(CRMSolver):

    name = "submission"

    def get_model(self, meta):
        return GraphModel(meta)

    def fit(self, model, conditions, fields):
        model.fit(conditions, fields)
