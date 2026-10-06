"""
Experiment 10 -- When does a compound get its own jar?
======================================================

Reviewer question 20.  In demonstration four EDIBLE (red AND round) stayed a
readout over the RED and ROUND jars.  Two harder cases:

  xor        ODD = red XOR round.  Not recoverable by any threshold on a
             weighted sum of the constituent jars' outputs.
  own-edge   CHOICE = very red (hue > 1.4) AND round.  The compound needs an
             edge on the hue channel at a DIFFERENT place from RED's edge, which
             RIPE still needs where it is.  This is a compound with its own
             contrast history: moderately red round things are its near misses.

plus the replication case
  linear     EDIBLE = red AND round (as in demonstration four).

Base jars (RIPE/hue, ROLLS/shape, PORTION/size) are trained from the start; the
compound target comes online halfway.  Each case is also run with recruitment
switched OFF at the compound's onset, which shows what the existing jars plus a
new readout can do on their own.

PREDICTIONS, stated before running.
  1. linear: readout suffices; no new jar is recruited for the compound.
  2. xor: readout alone fails (~75%); with recruitment, new units form that are
     tuned to two channels at once -- genuinely conjunctive jars.
  3. own-edge: readout alone falls short; with recruitment, a new unit forms that
     is tuned to hue at a new place.  Whether it is conjunctive (hue AND shape)
     or unary ("very red") is open.
  4. In every case the base jars keep carrying the base targets.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from pool import Pool2

QUICK = "--quick" in sys.argv
T = 10000 if QUICK else 24000
N_SEEDS = 3 if QUICK else 8
D, AMP, NOISE = 4, 1.2, 0.45
ONSET = T // 2
M = 4
NAMES = ["ripe", "rolls", "portion", "compound"]


def item(rng):
    f = rng.integers(0, 2, 3)
    s = np.zeros(D)
    s[:3] = (2 * f - 1) * AMP + rng.normal(0, NOISE, 3)
    s[3] = rng.normal(0, 1)
    return s, f


def compound(case, s, f):
    if case == "linear":
        return float(f[0] and f[1])
    if case == "xor":
        return float(f[0] != f[1])
    if case == "own-edge":
        return float(s[0] > 1.4 and f[1] == 1)
    raise ValueError(case)


def targets(case, s, f):
    return np.array([float(f[0]), float(f[1]), float(f[2]), compound(case, s, f)])


def run(case, seed, recruit_after_onset=True):
    P = Pool2(D, M, seed, mode="local")
    env = np.random.default_rng(1_000 + seed)
    for t in range(T):
        if t == ONSET and not recruit_after_onset:
            P.do_recruit = False
        s, f = item(env)
        on = np.array([True, True, True, t >= ONSET])
        P.step(s, targets(case, s, f), on)
    P.frozen = True

    rng = np.random.default_rng(66_000 + seed)
    on = np.ones(M, bool)
    cases = [(s, targets(case, s, f), on) for s, f in (item(rng) for _ in range(500))]
    full = P.acc_by_target(cases)
    units = []
    for k in np.where(P.alive)[0]:
        a = P.alive.copy(); a[k] = False
        drop = (full - P.acc_by_target(cases, a)).tolist()
        beta = np.exp(P.b[k][:3]); share = beta / beta.sum()
        top2 = np.sort(share)[::-1][:2]
        units.append(dict(k=int(k), born=int(P.born[k]), after_onset=bool(P.born[k] >= ONSET),
                          share=share.tolist(), top_channel=int(share.argmax()),
                          conjunctive=bool(top2[1] > 0.30), drop=drop,
                          centre=P.w[k][:3].tolist()))
    carrier = max(units, key=lambda u: u["drop"][3]) if units else None
    base_ok = all(any(u["top_channel"] == m and u["drop"][m] > 0.2 for u in units) for m in range(3))
    return dict(case=case, seed=seed, recruit_after_onset=recruit_after_onset,
                acc=full.tolist(), n_units=len(units),
                new_units=sum(u["after_onset"] for u in units), units=units,
                carrier=carrier, base_jars_intact=base_ok)


def main():
    out = []
    for case in ["linear", "xor", "own-edge"]:
        for recruit in [False, True]:
            for seed in range(N_SEEDS):
                r = run(case, seed, recruit)
                out.append(r)
                c = r["carrier"]
                print(f"  {case:8s} {'recruit' if recruit else 'readout'} seed {seed}: "
                      f"acc {np.round(r['acc'],3)} new units {r['new_units']} "
                      f"carrier: new={c['after_onset']} conj={c['conjunctive']} "
                      f"share={np.round(c['share'],2)} drop={c['drop'][3]:.2f} "
                      f"base intact={r['base_jars_intact']}", flush=True)
    json.dump(dict(T=T, onset=ONSET, n_seeds=N_SEEDS, runs=out), open("results_exp10.json", "w"))

    print("\nSUMMARY")
    for case in ["linear", "xor", "own-edge"]:
        for recruit in [False, True]:
            rs = [r for r in out if r["case"] == case and r["recruit_after_onset"] == recruit]
            acc = np.mean([r["acc"][3] for r in rs])
            newu = np.mean([r["new_units"] for r in rs])
            car_new = np.mean([r["carrier"]["after_onset"] for r in rs])
            car_conj = np.mean([r["carrier"]["conjunctive"] for r in rs])
            intact = np.mean([r["base_jars_intact"] for r in rs])
            print(f"  {case:8s} {'with recruitment' if recruit else 'readout only   '}: compound acc {acc:.3f}; "
                  f"new units {newu:.1f}; compound carried by a NEW unit in {car_new:.0%}, "
                  f"by a CONJUNCTIVE unit in {car_conj:.0%}; base jars intact {intact:.0%}")


if __name__ == "__main__":
    main()
