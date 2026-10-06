"""
Experiment 13 -- Can the realiser drift while the capacity stays put?
=====================================================================

Reviewer round 3, question 3.  Demonstration Two showed REPAIR after damage.
The stronger claim -- "stable capacity through changing realisation" -- needs
drift WITHOUT damage: at steady performance, with ongoing plasticity and small
noise, does the machinery that carries the capacity change progressively
while accuracy holds?

WORLD.  As exp8: channels hue, shape, size, noise; targets RIPE (hue) and
ROLLS (shape).

CONDITIONS (2 x 2)
  reliable / redundant   units never fail / each unit fails on 25% of learning
                         trials and is valued under those conditions (as exp8),
                         so several jars come to share each job
  plastic / noisy        after warm-up, learning simply continues / learning
                         continues AND every live parameter receives small
                         Gaussian noise each trial (synaptic turnover)

After a warm-up of T0 trials the realiser at t0 is recorded.  Over the drift
phase we track, every CHECK trials, with learning frozen for the measurement:
  accuracy          on RIPE and ROLLS
  old-unit load     accuracy lost on RIPE if every unit that was already alive
                    at t0 is removed together (how much of the capacity is
                    still carried by the original realiser)
  new-unit load     the same for units recruited after t0
  RSA similarity    correlation between the item-by-item similarity structure of
                    the population code at t and at t0 (identity-free; the
                    Laakso & Cottrell move)
  turnover          units recruited and pruned since t0

PREDICTIONS, stated before running.
  1. Accuracy stays high (> 0.95) throughout in all four conditions.
  2. plastic (no noise): the realiser is essentially static -- old-unit load
     stays near its t0 value, RSA near 1, little turnover.
  3. redundant + noisy: the old-unit load falls progressively while accuracy
     holds -- the capacity migrates onto newly recruited units.  This is drift
     in the strong sense.
  4. reliable + noisy: drift is smaller than in 3; a single load-bearing jar is
     held in place by its own error signal.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from pool import Pool2

QUICK = "--quick" in sys.argv
T0 = 8000 if QUICK else 16000
T_DRIFT = 20000 if QUICK else 60000
CHECK = 2000 if QUICK else 3000
N_SEEDS = 2 if QUICK else 6
D, M, AMP, NOISE = 4, 2, 1.2, 0.45
SIG_PARAM = 0.01


def item(rng):
    f = rng.integers(0, 2, 3)
    s = np.zeros(D)
    s[:3] = (2 * f - 1) * AMP + rng.normal(0, NOISE, 3)
    s[3] = rng.normal(0, 1)
    return s, np.array([float(f[0]), float(f[1])])


def uids(P):
    return {int(k): (int(k), int(P.born[k])) for k in np.where(P.alive)[0]}


def rsa_matrix(P, probes):
    X = np.array([P.act(s) for s in probes])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T
    iu = np.triu_indices(len(probes), 1)
    return S[iu]


def run(seed, redundant, noisy):
    P = Pool2(D, M, seed, mode="local", unit_dropout=0.25 if redundant else 0.0,
              prune_under_dropout=redundant)
    env = np.random.default_rng(2_000 + seed)
    noise_rng = np.random.default_rng(3_000 + seed)
    on = np.array([True, True])
    for _ in range(T0):
        s, tg = item(env); P.step(s, tg, on)

    test_rng = np.random.default_rng(8_000 + seed)
    cases = [(s, tg, on) for s, tg in (item(test_rng) for _ in range(300))]
    probes = [s for s, _, _ in cases[:60]]
    old = set(uids(P).values())
    rsa0 = rsa_matrix(P, probes)
    recruited, pruned = 0, 0
    trace = []

    def measure(t):
        drop = P.unit_dropout; P.unit_dropout = 0.0; P.frozen = True
        full = P.acc_by_target(cases)
        cur = uids(P)
        old_slots = [k for k, u in cur.items() if u in old]
        new_slots = [k for k, u in cur.items() if u not in old]
        a = P.alive.copy(); a[old_slots] = False
        old_load = float(full[0] - P.acc_by_target(cases, a)[0]) if old_slots else 0.0
        a = P.alive.copy(); a[new_slots] = False
        new_load = float(full[0] - P.acc_by_target(cases, a)[0]) if new_slots else 0.0
        r = float(np.corrcoef(rsa0, rsa_matrix(P, probes))[0, 1])
        P.unit_dropout = drop; P.frozen = False
        trace.append(dict(t=t, acc=full.tolist(), old_load=old_load, new_load=new_load,
                          n_old=len(old_slots), n_new=len(new_slots), rsa=r,
                          recruited=recruited, pruned=pruned))

    measure(0)
    for t in range(1, T_DRIFT + 1):
        before = uids(P)
        s, tg = item(env); P.step(s, tg, on)
        if noisy:
            a = P.alive
            P.w[a] += noise_rng.normal(0, SIG_PARAM, P.w[a].shape)
            P.b[a] += noise_rng.normal(0, SIG_PARAM, P.b[a].shape)
            P.V[:, a] += noise_rng.normal(0, SIG_PARAM, P.V[:, a].shape)
        after = uids(P)
        recruited += len(set(after.values()) - set(before.values()))
        pruned += len(set(before.values()) - set(after.values()))
        if t % CHECK == 0:
            measure(t)
    return dict(seed=seed, redundant=redundant, noisy=noisy, trace=trace)


def main():
    out = []
    for redundant in [False, True]:
        for noisy in [False, True]:
            for seed in range(N_SEEDS):
                r = run(seed, redundant, noisy); out.append(r)
                a, z = r["trace"][0], r["trace"][-1]
                print(f"  {'redundant' if redundant else 'reliable '} {'noisy  ' if noisy else 'plastic'} seed {seed}: "
                      f"acc {np.round(a['acc'],3)} -> {np.round(z['acc'],3)}  old-unit load {a['old_load']:.2f} -> {z['old_load']:.2f}  "
                      f"new-unit load {z['new_load']:.2f}  RSA {z['rsa']:.2f}  recruited {z['recruited']} pruned {z['pruned']}",
                      flush=True)
    json.dump(dict(T0=T0, T_drift=T_DRIFT, check=CHECK, sigma=SIG_PARAM, n_seeds=N_SEEDS, runs=out),
              open("results_exp13.json", "w"))
    print("\nSUMMARY (end of drift phase; mean over seeds)")
    for redundant in [False, True]:
        for noisy in [False, True]:
            rs = [r for r in out if r["redundant"] == redundant and r["noisy"] == noisy]
            f = lambda k, i=-1: np.mean([r["trace"][i][k] for r in rs])
            minacc = np.mean([min(min(tr["acc"]) for tr in r["trace"]) for r in rs])
            print(f"  {'redundant' if redundant else 'reliable '} {'noisy  ' if noisy else 'plastic'}: "
                  f"min acc over drift {minacc:.3f}; old-unit load {f('old_load',0):.2f} -> {f('old_load'):.2f}; "
                  f"new-unit load {f('new_load'):.2f}; RSA {f('rsa'):.2f}; recruited {f('recruited'):.1f} pruned {f('pruned'):.1f}")


if __name__ == "__main__":
    main()
