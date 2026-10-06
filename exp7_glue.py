"""
Experiment 7 -- Can a toy model illuminate the glue problem?
=============================================================

Reviewer question 17: construct environments with the SAME internal classifier
but different selection histories, so that a teleosemantic account would assign
different targets.  What does that show, and what does it not show?

WORLD.  Four channels: hue, softness, smell, noise.  Three worldly conditions
could be "what the jar is for":
    RED   (hue > 0)      RIPE  (soft > 0)      SAFE  (smell > 0)
Phase 1: the three always co-occur (ripe fruit is red, soft and sweet-smelling),
so a learner rewarded for eating ripe fruit, one rewarded for avoiding unsafe
fruit, and one rewarded for sorting by colour receive IDENTICAL feedback on
IDENTICAL inputs.
Phase 2: the conditions come apart (independent signs per channel).  Each
learner is now corrected by the condition its success actually depended on.

LEARNERS.
  A  success depends on RIPE
  B  success depends on SAFE
  C  success depends on RED
  S  a "swamp copy": A's exact state at the end of phase 1, created from nothing,
     then placed in A's phase-2 environment.

PREDICTIONS, stated before running.
  1. A, B and C are bit-for-bit identical at the end of phase 1.
  2. In phase 2 each re-tunes toward a different channel: A to softness, B to
     smell, C to hue.
  3. S's phase-2 trajectory is identical to A's.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from pool import Pool2

QUICK = "--quick" in sys.argv
T1 = 3000 if QUICK else 6000
T2 = 3000 if QUICK else 6000
N_SEEDS = 3 if QUICK else 8
D, AMP, NOISE = 4, 1.2, 0.45
CH = {"RED": 0, "RIPE": 1, "SAFE": 2}


def phase1_item(rng):
    k = int(rng.integers(2))
    s = np.zeros(D)
    s[:3] = (2 * k - 1) * AMP + rng.normal(0, NOISE, 3)
    s[3] = rng.normal(0, 1)
    return s, k


def phase2_item(rng):
    f = rng.integers(0, 2, 3)
    s = np.zeros(D)
    s[:3] = (2 * f - 1) * AMP + rng.normal(0, NOISE, 3)
    s[3] = rng.normal(0, 1)
    return s, f


def train_phase1(seed):
    P = Pool2(D, 1, seed, mode="local")
    env = np.random.default_rng(3_000 + seed)
    on = np.array([True])
    for _ in range(T1):
        s, k = phase1_item(env)
        P.step(s, np.array([float(k)]), on)     # identical for A, B, C: all conditions = k
    return P


def state(P):
    return np.concatenate([P.w.ravel(), P.b.ravel(), P.V.ravel(), P.c, P.alive.astype(float)])


def channel_shares(P):
    shares = []
    for k in np.where(P.alive)[0]:
        beta = np.exp(P.b[k][:3]) * max(abs(P.V[0, k]), 1e-9)
        shares.append(beta)
    tot = np.sum(shares, axis=0)
    return (tot / tot.sum()).tolist()


def test_acc(P, seed):
    rng = np.random.default_rng(9_000 + seed)
    was = P.frozen; P.frozen = True
    acc = {n: 0 for n in CH}
    N = 400
    for _ in range(N):
        s, f = phase2_item(rng)
        p, _ = P.predict(s)
        for n, ch in CH.items():
            acc[n] += int((p[0] > 0.5) == bool(f[ch]))
    P.frozen = was
    return {n: v / N for n, v in acc.items()}


def main():
    import copy
    out = []
    for seed in range(N_SEEDS):
        base = train_phase1(seed)
        learners = {n: copy.deepcopy(base) for n in ["A", "B", "C"]}
        # three independent trainings in phase 1 would be identical by construction;
        # verify that claim directly instead of assuming it
        check = {n: train_phase1(seed) for n in ["A", "B", "C"]}
        diff_AB = float(np.max(np.abs(state(check["A"]) - state(check["B"]))))
        diff_AC = float(np.max(np.abs(state(check["A"]) - state(check["C"]))))
        swamp = copy.deepcopy(check["A"])          # same state, no history
        learners = {"A": check["A"], "B": check["B"], "C": check["C"], "S": swamp}
        target_of = {"A": "RIPE", "B": "SAFE", "C": "RED", "S": "RIPE"}

        rec = {n: dict(t=[], acc=[], shares=[]) for n in learners}
        start = {n: dict(acc=test_acc(L, seed), shares=channel_shares(L)) for n, L in learners.items()}
        for n, L in learners.items():
            env = np.random.default_rng(5_000 + seed)       # same phase-2 inputs for everyone
            on = np.array([True])
            for t in range(T2):
                s, f = phase2_item(env)
                L.step(s, np.array([float(f[CH[target_of[n]]])]), on)
                if t % 500 == 0 or t == T2 - 1:
                    rec[n]["t"].append(t)
                    rec[n]["acc"].append(test_acc(L, seed))
                    rec[n]["shares"].append(channel_shares(L))
        diff_AS = float(np.max(np.abs(state(learners["A"]) - state(learners["S"]))))
        out.append(dict(seed=seed, diff_AB_phase1=diff_AB, diff_AC_phase1=diff_AC,
                        diff_AS_end=diff_AS, start=start, rec=rec))
        fin = {n: rec[n]["shares"][-1] for n in learners}
        print(f"  seed {seed}: |A-B|={diff_AB:.1e} |A-C|={diff_AC:.1e} |A-S| end={diff_AS:.1e} "
              f"start shares {np.round(start['A']['shares'],2)} -> "
              f"A {np.round(fin['A'],2)} B {np.round(fin['B'],2)} C {np.round(fin['C'],2)}", flush=True)

    json.dump(dict(T1=T1, T2=T2, n_seeds=N_SEEDS, runs=out), open("results_exp7.json", "w"))

    print("\nSUMMARY")
    print(f"  max |A-B| at end of phase 1: {max(r['diff_AB_phase1'] for r in out):.1e}")
    print(f"  max |A-C| at end of phase 1: {max(r['diff_AC_phase1'] for r in out):.1e}")
    print(f"  max |A-S| at end of phase 2: {max(r['diff_AS_end'] for r in out):.1e}")
    for n in ["A", "B", "C"]:
        s0 = np.mean([r["start"][n]["shares"] for r in out], 0)
        s1 = np.mean([r["rec"][n]["shares"][-1] for r in out], 0)
        a0 = {k: np.mean([r["start"][n]["acc"][k] for r in out]) for k in CH}
        a1 = {k: np.mean([r["rec"][n]["acc"][-1][k] for r in out]) for k in CH}
        print(f"  {n}: channel shares hue/soft/smell {np.round(s0,2)} -> {np.round(s1,2)}; "
              f"acc RED/RIPE/SAFE {[round(a0[k],2) for k in CH]} -> {[round(a1[k],2) for k in CH]}")


if __name__ == "__main__":
    main()
