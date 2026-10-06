"""
Experiment 8 -- Redundant realisation and the load-bearing measure.
===================================================================

Reviewer question 5: build a case where two components redundantly realise the
same classificatory contribution.  Does single-component ablation wrongly assign
each of them little or no load, although the pair is indispensable?  What
measure does better?

WORLD.  Same factorial world as exp4 (hue, shape, size, noise), with two
targets: RIPE (hue) and ROLLS (shape).

TWO CONDITIONS.
  reliable    units never fail.  Expect one hue jar and one shape jar.
  unreliable  each unit fails independently with probability 0.25 on every
              trial during learning, and a unit's worth is judged under those
              same operating conditions.  Redundancy now pays, so the learner
              should come to maintain more than one hue jar.

MEASURES, all with learning frozen and no failures at test time:
  single ablation    accuracy lost when one unit is removed (the report's measure)
  joint ablation     accuracy lost when a set of units is removed together
  Shapley value      each unit's average marginal contribution over all coalitions
                     of the other units -- multi-perturbation Shapley value
                     analysis (Keinan et al. 2004) -- which divides a jointly
                     realised contribution among the units that share it

PREDICTIONS, stated before running.
  1. Reliable: the hue jar's single-ablation load ~ 0.5 (chance), Shapley ~ 0.5.
  2. Unreliable: two or more hue jars, each with single-ablation load near 0,
     joint ablation ~ 0.5, Shapley values summing to ~ 0.5.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from pool import Pool2, shapley

QUICK = "--quick" in sys.argv
T = 8000 if QUICK else 16000
N_SEEDS = 3 if QUICK else 8
D, M, AMP, NOISE = 4, 2, 1.2, 0.45


def item(rng):
    f = rng.integers(0, 2, 3)
    s = np.zeros(D)
    s[:3] = (2 * f - 1) * AMP + rng.normal(0, NOISE, 3)
    s[3] = rng.normal(0, 1)
    return s, np.array([float(f[0]), float(f[1])])


def run(seed, unreliable):
    P = Pool2(D, M, seed, mode="local", unit_dropout=0.25 if unreliable else 0.0,
              prune_under_dropout=unreliable)
    env = np.random.default_rng(2_000 + seed)
    on = np.array([True, True])
    for _ in range(T):
        s, tg = item(env)
        P.step(s, tg, on)
    P.frozen = True
    P.unit_dropout = 0.0

    rng = np.random.default_rng(8_000 + seed)
    cases = [(s, tg, on) for s, tg in (item(rng) for _ in range(300))]
    alive = [int(k) for k in np.where(P.alive)[0]]

    def acc_with(S, m):
        a = np.zeros_like(P.alive); a[list(S)] = True
        return float(P.acc_by_target(cases, a)[m])

    full = P.acc_by_target(cases)
    units = []
    for k in alive:
        beta = np.exp(P.b[k][:3]); ch = int(beta.argmax())
        units.append(dict(k=k, channel=ch, selectivity=float(beta.max() / beta.sum())))
    res = dict(seed=seed, unreliable=unreliable, n_units=len(alive), full=full.tolist(), units=units)
    for m, name in [(0, "ripe"), (1, "rolls")]:
        single = {k: float(full[m] - acc_with(set(alive) - {k}, m)) for k in alive}
        phi, v = shapley(lambda S, m=m: acc_with(S, m), alive)
        relevant = [u["k"] for u in units if u["channel"] == m]
        joint = float(full[m] - acc_with(set(alive) - set(relevant), m)) if relevant else 0.0
        res[name] = dict(single=single, shapley={k: float(x) for k, x in phi.items()},
                         relevant=relevant, joint=joint,
                         baseline_empty=float(v(frozenset())))
    return res


def main():
    out = []
    for unreliable in [False, True]:
        for seed in range(N_SEEDS):
            r = run(seed, unreliable)
            out.append(r)
            rp = r["ripe"]
            print(f"  {'unreliable' if unreliable else 'reliable  '} seed {seed}: units {r['n_units']} "
                  f"hue jars {rp['relevant']} single {[round(rp['single'][k],3) for k in rp['relevant']]} "
                  f"joint {rp['joint']:.3f} shapley {[round(rp['shapley'][k],3) for k in rp['relevant']]}",
                  flush=True)
    json.dump(dict(T=T, n_seeds=N_SEEDS, runs=out), open("results_exp8.json", "w"))

    print("\nSUMMARY (target RIPE; units tuned to hue)")
    for unreliable in [False, True]:
        rs = [r for r in out if r["unreliable"] == unreliable]
        nh = [len(r["ripe"]["relevant"]) for r in rs]
        singles = [r["ripe"]["single"][k] for r in rs for k in r["ripe"]["relevant"]]
        shap = [r["ripe"]["shapley"][k] for r in rs for k in r["ripe"]["relevant"]]
        shap_sum = [sum(r["ripe"]["shapley"][k] for k in r["ripe"]["relevant"]) for r in rs]
        joints = [r["ripe"]["joint"] for r in rs]
        print(f"  {'unreliable' if unreliable else 'reliable'}: hue jars per learner {np.mean(nh):.2f} "
              f"(range {min(nh)}-{max(nh)}); single-ablation load per hue jar {np.mean(singles):.3f}; "
              f"joint load {np.mean(joints):.3f}; Shapley per hue jar {np.mean(shap):.3f}, "
              f"summed {np.mean(shap_sum):.3f}")


if __name__ == "__main__":
    main()
