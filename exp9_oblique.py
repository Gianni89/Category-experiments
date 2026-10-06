"""
Experiment 9 -- Overlap when the relevant dimensions are oblique to the channels.
================================================================================

Reviewer question 19.  Demonstration four used features aligned with the input
channels, the easiest case for overlap.  Here the three behaviourally relevant
features are ROTATED relative to the channel basis, so every feature is spread
across all three channels and per-channel precision cannot isolate any of them.

Minimal extra machinery: each unit may learn ONE preferred direction in input
space (a rank-1 learned metric).  It is a parameter of the unit itself, updated
from the unit's own error signal, so learning stays as local as in exp4.

LEARNERS (identical except as stated)
  aligned-axis   features aligned with channels, per-channel precision (= exp4 local)
  oblique-axis   features rotated, per-channel precision only
  oblique-dir    features rotated, per-unit learned direction
  oblique-dir-G  as oblique-dir, but global (softmax) competition

The two cost variants (and aligned-axis-cost) were ADDED AFTER A PILOT RUN, in
which oblique-dir recovered only part of the aligned result.  lam_b = 0.02 was
the one value tried; it is declared here rather than presented as prespecified.

PREDICTIONS, stated before running.
  1. oblique-axis loses what aligned-axis had: units become conjunctive and the
     never-seen combination is handled much worse.
  2. oblique-dir recovers it: units align with the true feature directions,
     inputs are claimed by several jars at once, and the never-seen combination is
     handled well.
  3. oblique-dir-G does not: global competition still forces conjunctive cells.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from pool import Pool2

QUICK = "--quick" in sys.argv
T = 10000 if QUICK else 24000
N_SEEDS = 3 if QUICK else 8
D, M, AMP, NOISE = 4, 3, 1.2, 0.45
HELD_OUT = (0, 0, 1)


def rotation():
    def rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rz(np.radians(40)) @ ry(np.radians(35)) @ rx(np.radians(50))


ROT = rotation()


def item(rng, rotate, allow_held_out=False):
    while True:
        f = tuple(int(v) for v in rng.integers(0, 2, 3))
        if allow_held_out or f != HELD_OUT:
            break
    z = (2 * np.array(f) - 1) * AMP + rng.normal(0, NOISE, 3)
    s = np.zeros(D)
    s[:3] = ROT @ z if rotate else z
    s[3] = rng.normal(0, 1)
    return s, f


CONFIGS = {
    "aligned-axis":  dict(rotate=False, directions=False, mode="local"),
    "oblique-axis":  dict(rotate=True,  directions=False, mode="local"),
    "oblique-dir":   dict(rotate=True,  directions=True,  mode="local"),
    "oblique-dir-G": dict(rotate=True,  directions=True,  mode="global"),
    # added after the pilot: in the aligned case the capacity cost did no work;
    # does it matter once dimensions are oblique?
    "oblique-dir-cost": dict(rotate=True, directions=True, mode="local", lam_b=0.02),
    "oblique-dir-nocost": dict(rotate=True, directions=True, mode="local", lam_b=0.0),
    "aligned-axis-cost": dict(rotate=False, directions=False, mode="local", lam_b=0.02),
}


def run(name, seed):
    cfg = CONFIGS[name]
    P = Pool2(D, M, seed, mode=cfg["mode"], directions=cfg["directions"],
              lam_b=cfg.get("lam_b", 0.004))
    env = np.random.default_rng(1_000 + seed)
    on = np.ones(M, bool)
    for _ in range(T):
        s, f = item(env, cfg["rotate"])
        P.step(s, np.array(f, float), on)
    P.frozen = True

    rng = np.random.default_rng(77_000 + seed)
    trained, novel = [], []
    while len(trained) < 400 or len(novel) < 200:
        s, f = item(rng, cfg["rotate"], allow_held_out=True)
        (novel if f == HELD_OUT else trained).append((s, np.array(f, float), on))
    trained, novel = trained[:400], novel[:200]
    acc_tr = P.acc_by_target(trained).tolist()
    acc_nv = P.acc_by_target(novel).tolist()
    X = np.array([P.predict(s)[1] for s, _, _ in trained])
    multi = float(np.mean(np.sum(X > 0.5, axis=1) >= 2))

    # tuning measured along the TRUE feature directions in input space
    basis = np.zeros((3, D))
    basis[:, :3] = (ROT if cfg["rotate"] else np.eye(3)).T
    sel = P.selectivity(basis=basis)
    return dict(name=name, seed=seed, n_units=int(P.alive.sum()), acc_trained=acc_tr,
                acc_novel=acc_nv, multi_claimed=multi,
                selectivity=[s_ for _, s_, _ in sel], top_feature=[c for _, _, c in sel])


def main():
    out = []
    for name in CONFIGS:
        for seed in range(N_SEEDS):
            r = run(name, seed)
            out.append(r)
            print(f"  {name:18s} seed {seed}: units {r['n_units']:2d} multi {r['multi_claimed']:.2f} "
                  f"sel {np.mean(r['selectivity']):.2f} trained {np.round(r['acc_trained'],2)} "
                  f"novel {np.round(r['acc_novel'],2)}", flush=True)
    json.dump(dict(T=T, n_seeds=N_SEEDS, rotation=ROT.tolist(), runs=out), open("results_exp9.json", "w"))
    print("\nSUMMARY")
    for name in CONFIGS:
        rs = [r for r in out if r["name"] == name]
        f = lambda k: np.mean([r[k] for r in rs], 0)
        print(f"  {name:18s} units {f('n_units'):.1f}  claimed by >=2 {f('multi_claimed'):.2f}  "
              f"selectivity {np.mean([np.mean(r['selectivity']) for r in rs]):.2f}  "
              f"trained {np.round(f('acc_trained'),3)}  NEVER-SEEN {np.round(f('acc_novel'),3)}")


if __name__ == "__main__":
    main()
