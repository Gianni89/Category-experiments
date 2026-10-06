"""
Experiment 1R -- How architecture-dependent is demonstration one?
================================================================

A reviewer asked whether the boundary/prototype dissociation survives
materially different learning rules, kernel shapes, capacity penalties and
category representations, or whether it depends on learned precision being a
GLOBAL property of a unit.

Same environment as exp1 (positive class at 0, contrast fixed at -2.5, second
contrast swept on the right; positive stream byte-identical across conditions).
Seven learners:

  baseline        the report's learner (Gaussian kernel, one precision per unit,
                  K = sum beta)
  laplace         L1 kernel instead of Gaussian (heavier tails)
  cost-quadratic  K = sum beta^2
  cost-log        K = sum log beta (constant pressure; saturating equilibrium)
  side-precision  a SEPARATE precision for each side of the centre on each
                  channel.  This removes exactly the coupling the reviewer
                  suspected: precision is no longer global to the unit.
  alcove          Kruschke's ALCOVE: exemplar nodes on a covering grid, global
                  sensitivity, learned per-dimension attention, humble teacher.
                  No prototype, no capacity cost.
  rbf-logistic    fixed-width RBF features and a logistic readout with weight
                  decay.  Purely discriminative; nothing learned about widths.

All effects are measured against the true positive mean (0), which is identical
across conditions, so the measurements are architecture-neutral.

Effects scored:
  E1  right boundary monotone in contrast distance
  E2  zero-crossing of asymmetry near the symmetric point (2.5)
  E3  asymmetry non-monotone (interior maximum)
  E4  opposite-flank effect: left reach varies with the RIGHT contrast
  E5  centre of the classificatory region shifts away from the near contrast
      (the peak-shift analogue)
"""

from __future__ import annotations

import json
import sys
import numpy as np

from jars_model import Params, Learner, softmax

SIGMA = 0.40
LEFT = -2.50
DELTAS = [1.00, 1.50, 2.00, 2.50, 3.00, 3.50]
GRID = np.arange(-5.0, 6.0 + 1e-9, 0.01)

QUICK = "--quick" in sys.argv
N_TRIALS = 2500 if QUICK else 7000
N_SEEDS = 2 if QUICK else 8
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]


def make_streams(seed, n):
    rng = np.random.default_rng(10_000 + seed)
    u = rng.random(n)
    kind = np.where(u < 0.50, 0, np.where(u < 0.75, 1, 2))
    p_items = rng.normal(0.0, SIGMA, size=(n, 2))
    l_items = rng.normal(0.0, SIGMA, size=(n, 2)) + np.array([LEFT, 0.0])
    r_shape = rng.normal(0.0, SIGMA, size=(n, 2))
    return kind, p_items, l_items, r_shape


# ---------------------------------------------------------------------------
# learners
# ---------------------------------------------------------------------------

class JarsWrap:
    """The report's learner, optionally with kernel / cost-form switched."""

    def __init__(self, seed, **kw):
        p = Params(n_app=2, n_ctx=0, N=6, C=0, n_act=2, **kw)
        self.L = Learner(p, np.random.default_rng(seed), n_init_units=3)

    def step(self, s, correct):
        self.L.step(s, correct)

    def dv(self, s):
        self.L.frozen = True
        x, _ = self.L.settle(s)
        z = self.L.A @ x
        return z[1] - z[0]

    def proto(self, positives):
        acts = np.array([self.L.settle(s)[0] for s in positives]).mean(0)
        return float(self.L.w[int(np.argmax(acts)), 0])


class SidePrecision(Learner):
    """Precision is held separately for each side of the centre on each channel.

    If the opposite-flank effect exists only because precision is a single
    global property of the unit, it should disappear here.
    """

    def __init__(self, p, rng, n_init_units=3):
        super().__init__(p, rng, n_init_units)
        self.bs = np.full((p.N, p.n_app, 2), p.b0)

    def _u(self, app, alive):
        d = app[None, :] - self.w
        side = (d > 0).astype(int)
        beta = np.exp(np.take_along_axis(self.bs, side[:, :, None], axis=2)[:, :, 0])
        u = -np.sum(beta * d ** 2, axis=1) + self.a
        return np.where(alive, u, -1e9)

    def settle(self, s, g=None, R=None, L=None, alive=None):
        p = self.p
        L = self.L if L is None else L
        alive = self.alive if alive is None else alive
        u = self._u(s[:p.n_app], alive)
        x = softmax(u)
        for _ in range(p.T_fast):
            h = np.where(alive, u + p.gamma * (L @ x), -1e9)
            x = (1 - p.damp) * x + p.damp * softmax(h)
        return x / x.sum(), np.zeros(0)

    def _learn(self, s, x, y, a_hat, correct_action, r, cons_obs):
        p = self.p
        app = s[:p.n_app]
        err = 1.0 - r
        d = app[None, :] - self.w
        resp = self.responsibility(app)
        dw = p.eta_w * resp[:, None] * d
        if err > 0:
            nrm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            dw -= p.eta_w_err * x[:, None] * (d / nrm)
        self.w += dw * self.alive[:, None]

        # credit each (unit, channel, side) for errors made on inputs on that side,
        # in proportion to how much of the input's deviation lies on that channel
        side = (d > 0).astype(int)
        share = d ** 2 / (np.sum(d ** 2, axis=1, keepdims=True) + 1e-9)
        credit = np.zeros_like(self.bs)
        N, D = d.shape
        credit[np.arange(N)[:, None], np.arange(D)[None, :], side] = x[:, None] * err * share
        self.bs += p.eta_b * (credit - p.lam_b * np.exp(self.bs)) * self.alive[:, None, None]
        np.clip(self.bs, p.b_min, p.b_max, out=self.bs)

        if err > 0:
            outer = np.outer(x, x); np.fill_diagonal(outer, 0.0)
            self.L -= p.eta_L * outer
        self.L *= (1.0 - p.lam_L)
        np.fill_diagonal(self.L, 0.0)
        np.clip(self.L, p.L_min, p.L_max, out=self.L)

        self.A[a_hat] -= p.eta_A * err * x
        self.A[correct_action] += p.eta_A * err * x


class SideWrap(JarsWrap):
    def __init__(self, seed):
        p = Params(n_app=2, n_ctx=0, N=6, C=0, n_act=2)
        self.L = SidePrecision(p, np.random.default_rng(seed), n_init_units=3)


class SideFixedReadout(SidePrecision):
    """Side-specific precision AND a readout whose magnitude errors cannot erode.

    Each unit votes with the running proportion of 'avoid' among the cases it
    claims, so errors on one flank no longer weaken the unit's vote on the other.
    If the opposite-flank effect survives side-specific precision because the
    unit's single error-driven readout weight couples the flanks, it should
    vanish here.
    """

    def __init__(self, p, rng, n_init_units=3):
        super().__init__(p, rng, n_init_units)
        self.m = np.full(p.N, 0.5)

    def _learn(self, s, x, y, a_hat, correct_action, r, cons_obs):
        eta_A = self.p.eta_A
        self.p.eta_A = 0.0
        super()._learn(s, x, y, a_hat, correct_action, r, cons_obs)
        self.p.eta_A = eta_A
        self.m += 0.02 * x * (correct_action - self.m)
        self.A[1] = self.m
        self.A[0] = 1.0 - self.m


class SideFixedWrap(JarsWrap):
    def __init__(self, seed):
        p = Params(n_app=2, n_ctx=0, N=6, C=0, n_act=2)
        self.L = SideFixedReadout(p, np.random.default_rng(seed), n_init_units=3)


def covering_grid():
    gx = np.arange(-4.5, 5.5 + 1e-9, 0.25)
    gy = np.arange(-1.5, 1.5 + 1e-9, 0.5)
    return np.array([[a, b] for a in gx for b in gy])


class Alcove:
    """Kruschke (1992) ALCOVE, covering-map version, two response categories."""

    def __init__(self, seed, c=2.0, lw=0.03, la=0.003, phi=2.0):
        self.rng = np.random.default_rng(seed)
        self.H = covering_grid()
        self.alpha = np.array([0.5, 0.5])
        self.W = np.zeros((2, len(self.H)))
        self.c, self.lw, self.la, self.phi = c, lw, la, phi

    def _hid(self, s):
        dist = np.abs(self.H - s[None, :])
        return np.exp(-self.c * dist @ self.alpha), dist

    def step(self, s, correct):
        a, dist = self._hid(s)
        o = self.W @ a
        t = np.where(np.arange(2) == correct, np.maximum(o, 1.0), np.minimum(o, -1.0))
        e = t - o
        back = (e @ self.W) * a                          # per hidden node
        self.W += self.lw * np.outer(e, a)
        # dE/d(alpha_d) = sum_j back_j * c * |h_jd - x_d|  (Kruschke 1992, eq. 6)
        self.alpha += -self.la * (back @ (self.c * dist))
        self.alpha = np.maximum(self.alpha, 0.0)

    def dv(self, s):
        a, _ = self._hid(s)
        o = self.W @ a
        return o[1] - o[0]

    def proto(self, positives):
        return None


class RBFLogistic:
    """Fixed-width RBF features, logistic readout, weight decay.  No learned widths."""

    def __init__(self, seed, width=0.40, eta=0.05, wd=1e-4):
        self.H = covering_grid()
        self.v = np.zeros(len(self.H)); self.c0 = 0.0
        self.width, self.eta, self.wd = width, eta, wd

    def _phi(self, s):
        return np.exp(-np.sum((self.H - s[None, :]) ** 2, axis=1) / (2 * self.width ** 2))

    def step(self, s, correct):
        f = self._phi(s)
        z = self.v @ f + self.c0
        pr = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        e = correct - pr
        self.v += self.eta * (e * f - self.wd * self.v)
        self.c0 += self.eta * e

    def dv(self, s):
        return self.v @ self._phi(s) + self.c0

    def proto(self, positives):
        return None


LEARNERS = {
    "baseline":       lambda seed: JarsWrap(seed),
    "laplace":        lambda seed: JarsWrap(seed, kernel="laplace"),
    "cost-quadratic": lambda seed: JarsWrap(seed, cost_form="quadratic", lam_b=0.008),
    "cost-log":       lambda seed: JarsWrap(seed, cost_form="log", lam_b=0.02),
    "side-precision": lambda seed: SideWrap(seed),
    "side-fixed-readout": lambda seed: SideFixedWrap(seed),
    "alcove":         lambda seed: Alcove(seed),
    "rbf-logistic":   lambda seed: RBFLogistic(seed),
}


# ---------------------------------------------------------------------------

def run(name, seed, delta):
    m = LEARNERS[name](seed)
    kind, p_items, l_items, r_shape = make_streams(seed, N_TRIALS)
    pos = []
    for t in range(N_TRIALS):
        if kind[t] == 0:
            s, c = p_items[t], 0; pos.append(s)
        elif kind[t] == 1:
            s, c = l_items[t], 1
        else:
            s, c = r_shape[t] + np.array([delta, 0.0]), 1
        m.step(s, c)

    dv = np.array([m.dv(np.array([g, 0.0])) for g in GRID])
    zero = int(np.argmin(np.abs(GRID)))

    def cross(idxs, rising):
        seg = np.sign(dv)
        d = np.diff(seg)
        cand = [i for i in idxs if (d[i] > 0 if rising else d[i] < 0)]
        if not cand:
            return np.nan
        i = cand[0] if rising else cand[-1]
        den = dv[i + 1] - dv[i]
        fr = -dv[i] / den if abs(den) > 1e-12 else 0.0
        return float(GRID[i] + fr * (GRID[i + 1] - GRID[i]))

    br = cross(range(zero, len(GRID) - 1), True)
    bl = cross(range(0, zero), False)
    proto = m.proto(np.array(pos[-300:]))
    return dict(delta=delta, b_right=br, b_left=bl,
                reach_right=br, reach_left=-bl,
                asym=(-bl) - br, centre=(bl + br) / 2.0,
                proto=proto if proto is not None else np.nan)


def main():
    names = ONLY or list(LEARNERS)
    out = {}
    for name in names:
        rows = []
        for seed in range(N_SEEDS):
            for d in DELTAS:
                rows.append(run(name, seed, d))
        agg = {}
        for k in ["b_right", "b_left", "reach_left", "reach_right", "asym", "centre", "proto"]:
            M = np.array([[r[k] for r in rows if r["delta"] == d] for d in DELTAS], float)
            agg[k] = dict(mean=np.nanmean(M, 1).tolist(),
                          sem=(np.nanstd(M, 1) / np.sqrt(np.sum(~np.isnan(M), 1).clip(1))).tolist())
        br = np.array(agg["b_right"]["mean"]); asym = np.array(agg["asym"]["mean"])
        rl = np.array(agg["reach_left"]["mean"]); ce = np.array(agg["centre"]["mean"])
        rl_sem = np.array(agg["reach_left"]["sem"])
        # interpolated zero-crossing of asymmetry on the descending limb
        zc = np.nan
        for i in range(len(DELTAS) - 1):
            if asym[i] > 0 >= asym[i + 1]:
                zc = DELTAS[i] + (asym[i] / (asym[i] - asym[i + 1])) * (DELTAS[i + 1] - DELTAS[i])
        pr = np.array(agg["proto"]["mean"])
        score = dict(
            E1_boundary_monotone=bool(np.all(np.diff(br) > 0)),
            E2_zero_crossing=float(zc),
            E3_nonmonotone=bool(np.argmax(asym) not in (0,)),
            E4_opposite_flank_range=float(rl.max() - rl.min()),
            E4_opposite_flank_sem=float(rl_sem.mean()),
            E5_centre_shift=float(ce[0] - ce[-1]),
            boundary_travel=float(br.max() - br.min()),
            proto_travel=float(np.nanmax(pr) - np.nanmin(pr)) if not np.all(np.isnan(pr)) else None,
        )
        out[name] = dict(agg=agg, score=score)
        print(f"{name:15s} E1 {score['E1_boundary_monotone']!s:5s} "
              f"zc {score['E2_zero_crossing']:.2f}  E3 {score['E3_nonmonotone']!s:5s} "
              f"opp-flank {score['E4_opposite_flank_range']:.3f}±{score['E4_opposite_flank_sem']:.3f} "
              f"centre-shift {score['E5_centre_shift']:+.3f}  "
              f"bnd {score['boundary_travel']:.2f} proto {score['proto_travel']}", flush=True)
    tag = "_".join(names) if ONLY else "all"
    with open(f"results_exp1R_{tag}.json", "w") as f:
        json.dump(dict(deltas=DELTAS, n_seeds=N_SEEDS, n_trials=N_TRIALS, results=out), f)


if __name__ == "__main__":
    main()
