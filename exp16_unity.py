"""
Experiment 16 -- Is far-edge coupling the same thing as unity?
==============================================================

Reviewer round 3, question 1.  v4 makes UNITY essential to jarhood and uses
far-edge coupling as its evidence.  Two ways that could go wrong:

  too strong  one reusable capacity realised WITHOUT any parameter spanning its
              edges shows no coupling.  (Already in hand: the RBF-logistic and
              ALCOVE learners of the robustness run -- a single readout, a single
              error signal, one capacity -- show coupling <= 0.01.)
  too weak    two DIFFERENT capacities that happen to share a parameter show
              coupling ACROSS them, so coupling would fuse two jars into one.

This experiment builds the second case.

WORLD (one dimension).  Category A at 0 with contrasts at -2.5 (fixed) and at
+delta (swept).  Category B at 10 with contrasts at 7.5 and 12.5 (fixed).
Two targets: is-A, is-B.  Within a replicate every stream is identical across
delta except the position of A's right-hand contrast.

LEARNERS.  Each category has its own jar -- one unit, centre w, precision beta,
readout v, bias c -- trained by error with a capacity cost on precision, as in
demonstration one.
  separate   A and B each have their own precision
  shared     A and B share ONE precision parameter (a global gain on the
             dimension, like a shared attention weight)

MEASURES: reach of A's LEFT edge (within-jar far edge) and reach of B's LEFT
edge (the other jar), as delta varies.

PREDICTIONS, stated before running.
  1. A's left edge moves with delta in both learners (within-jar coupling).
  2. B's left edge is flat under 'separate' and moves with delta under
     'shared': far-edge coupling ACROSS two jars.
"""

from __future__ import annotations

import json
import sys
import numpy as np

QUICK = "--quick" in sys.argv
T = 12000 if QUICK else 24000
N_SEEDS = 3 if QUICK else 8
DELTAS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
SIG = 0.4


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def stream(seed):
    rng = np.random.default_rng(16_000 + seed)
    kind = rng.integers(0, 6, T)       # 0 A, 1 A-left, 2 A-right(delta), 3 B, 4 B-left, 5 B-right
    noise = rng.normal(0, SIG, T)
    return kind, noise


def run(seed, delta, shared, lam=0.02, eta=0.05, eta_w=0.02, eta_b=0.02):
    kind, noise = stream(seed)
    centres = [0.0, -2.5, delta, 10.0, 7.5, 12.5]
    w = np.array([0.0, 10.0]); b = np.log(np.array([0.8, 0.8]))
    v = np.array([2.0, 2.0]); c = np.array([-1.0, -1.0])
    for t in range(T):
        x = centres[kind[t]] + noise[t]
        y = np.array([float(kind[t] == 0), float(kind[t] == 3)])
        beta = np.exp(b); a = np.exp(-beta * (x - w) ** 2)
        e = y - sig(v * a + c)
        g = e * v * a
        v += eta * e * a; c += eta * e
        w += eta_w * g * 2 * beta * (x - w)
        gb = beta * (g * (-(x - w) ** 2)) - lam * beta
        if shared:
            gb = np.full(2, gb.sum())
        b += eta_b * gb
        b = np.clip(b, -6, 4)
        if shared:
            b[1] = b[0]

    def reach_left(j):
        xs = np.arange(w[j] - 4, w[j] + 1e-9, 0.005)
        p = sig(v[j] * np.exp(-np.exp(b[j]) * (xs - w[j]) ** 2) + c[j])
        above = np.where(p > 0.5)[0]
        return float(w[j] - xs[above[0]]) if len(above) else 0.0

    return dict(seed=seed, delta=delta, shared=shared, reach_A_left=reach_left(0), reach_B_left=reach_left(1),
                beta=np.exp(b).tolist(), w=w.tolist())


def main():
    out = []
    for shared in [False, True]:
        for seed in range(N_SEEDS):
            for d in DELTAS:
                out.append(run(seed, d, shared))
        rs = [r for r in out if r["shared"] == shared]
        for key in ["reach_A_left", "reach_B_left"]:
            m = [np.mean([r[key] for r in rs if r["delta"] == d]) for d in DELTAS]
            print(f"  {'shared  ' if shared else 'separate'} {key}: " + " ".join(f"{x:.3f}" for x in m)
                  + f"   range {max(m) - min(m):.3f}", flush=True)
    json.dump(dict(T=T, n_seeds=N_SEEDS, deltas=DELTAS, runs=out), open("results_exp16.json", "w"))


if __name__ == "__main__":
    main()
