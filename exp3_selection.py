"""
Experiment 3 -- Selection under a capacity cost: split, merge, and Pareto.
=========================================================================

Post 1: "we keep the distinctions that earn their place."  Section 6 of the
brief writes this as U(r) = V(r) - lambda*K(r) and then worries (Section 24.C)
that it is circular: retained because valuable, valuable because retained.

The circularity is avoidable, and the avoidance is mechanical rather than
verbal.  Here V is measured COUNTERFACTUALLY against feedback the environment
generates regardless of what the learner has retained: every prune_period the
learner asks, of each unit, "if I removed you, how much accuracy would I lose
on the cases I have just seen?"  A unit survives iff that loss exceeds
lambda*K.  Nothing in that test consults whether the unit was previously kept.

PART A -- the distinction has to earn its place, and can lose it.
  Three phases over one continuous life:
    Phase 1: sub-regions S1 and S2 require the SAME action.  The feature
             structure that would support a split is fully present, but the
             split buys nothing.
    Phase 2: S1 and S2 require DIFFERENT actions.  Now it buys ~1/3 accuracy.
    Phase 3: they require the same action again.  It stops paying.
  Prediction: unit count goes 2 -> 3 -> 2, and the split happens in phase 2 and
  NOT in phase 1 -- even though the bimodal feature structure was there all
  along.  That is the non-circular claim: retention tracks independently
  generated feedback, not the shape of the input.

PART B -- the phase boundary sits where the theory says it does.
  Sweep lambda.  The split should occur iff the measured accuracy gain exceeds
  lambda.  Since the gain here is ~1/3 (a merged unit must give one action to
  two sub-regions that need different ones), the critical lambda should be
  near 0.33 -- a quantitative prediction, not a curve-fit.

PART C -- scalar utility vs a Pareto front.
  The critique claimed that a vector of values with a partial order predicts
  something a scalarised utility does not.  Here is the concrete difference:
  a distinction lying on a CONCAVE stretch of the Pareto front is optimal
  under no linear weighting of the value dimensions, so a scalarising learner
  drops it for every weighting, while a Pareto-front learner keeps it.  That
  is a real, testable divergence between the two readings of Section 6.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from jars_model import Params, Learner


QUICK = "--quick" in sys.argv
N_SEEDS = 3 if QUICK else 10
PHASE = 4000 if QUICK else 7000          # trials per phase
SIGMA = 0.35
S1, S2, D = 0.0, 1.3, 4.0                # sub-regions and the distinct category


def phase_of(t):
    return 0 if t < PHASE else (1 if t < 2 * PHASE else 2)


def sample(rng, t):
    which = int(rng.integers(3))
    ph = phase_of(t)
    if which == 0:
        x0, correct = S1, 0
    elif which == 1:
        x0, correct = S2, (2 if ph == 1 else 0)     # diverges only in phase 2
    else:
        x0, correct = D, 1
    app = np.array([x0, 0.0]) + rng.normal(0, SIGMA, 2)
    return app, correct, which


def params(lam_unit):
    return Params(n_app=2, n_ctx=0, N=6, C=0, n_act=3,
                  allow_structure_change=True, lam_unit=lam_unit,
                  prune_period=500, prune_buffer=240,
                  recruit_err_thresh=0.22, recruit_err_window=150)


def run(seed, lam_unit, n_phases=3, record=False):
    p = params(lam_unit)
    rng = np.random.default_rng(seed)
    L = Learner(p, rng, n_init_units=2)
    L.init_centres_from(np.array([sample(np.random.default_rng(seed + 1), 0)[0]
                                  for _ in range(200)]))

    env = np.random.default_rng(1000 + seed)
    total = n_phases * PHASE
    trace = []
    for t in range(total):
        app, correct, which = sample(env, t)
        L.step(app, correct)
        if record and t % 200 == 0:
            trace.append(dict(t=t, n_units=int(L.alive.sum()),
                              err=float(np.mean(L._err_hist[-300:]))))
    L.frozen = True
    return L, trace


def separates(L, seed):
    """Do S1 and S2 recruit DIFFERENT units?  (Not merely: are there 3 units.)"""
    rng = np.random.default_rng(4242 + seed)
    win = {}
    for name, x0 in [("S1", S1), ("S2", S2)]:
        ks = [int(np.argmax(L.settle(np.array([x0, 0.0]) + rng.normal(0, SIGMA, 2))[0]))
              for _ in range(120)]
        win[name] = max(set(ks), key=ks.count)
    return win["S1"] != win["S2"], win


def measured_gain(seed):
    """The accuracy a split is actually worth here, measured, not assumed.

    Train to the end of phase 2 with a generous capacity budget, then compare
    accuracy with the split structure against the best single-unit-for-S1-and-S2
    alternative (ablate the S2 unit and let the readout do its best).
    """
    L, _ = run(seed, lam_unit=0.0, n_phases=2)
    sep, win = separates(L, seed)
    if not sep:
        return np.nan
    rng = np.random.default_rng(9000 + seed)
    test = []
    for _ in range(600):
        which = int(rng.integers(3))
        x0, correct = [(S1, 0), (S2, 2), (D, 1)][which]
        test.append((np.array([x0, 0.0]) + rng.normal(0, SIGMA, 2), correct))
    full = np.mean([int(np.argmax(L.A @ L.settle(s)[0]) == c) for s, c in test])
    alive2 = L.alive.copy(); alive2[win["S2"]] = False
    merged = np.mean([int(np.argmax(L.A @ L.settle(s, alive=alive2)[0]) == c) for s, c in test])
    return float(full - merged)


# --------------------------------------------------------------------------
# Part C: scalarisation cannot reach a concave Pareto front
# --------------------------------------------------------------------------

def pareto_demo():
    """Candidate distinctions with two independent value dimensions.

    V1 = predictive value, V2 = value for action.  A scalarising learner keeps
    the top `budget` candidates by w*V1 + (1-w)*V2 for its own weighting w; a
    Pareto learner keeps whatever is undominated.  Sweeping the budget shows
    WHEN the two readings of Section 6 actually differ.
    """
    cands = {
        "specialist-1": (1.00, 0.15),
        "mixed-1":      (0.85, 0.35),
        "generalist":   (0.55, 0.55),   # undominated, but in a CONCAVE notch:
        "mixed-2":      (0.35, 0.85),   # it lies below the chord joining its
        "specialist-2": (0.15, 1.00),   # neighbours, so no linear weighting
        "dominated-1":  (0.30, 0.30),   # ever makes it the best candidate
        "dominated-2":  (0.50, 0.20),
        "dominated-3":  (0.20, 0.45),
    }
    names = list(cands)
    V = np.array([cands[n] for n in names])

    dominated = np.zeros(len(names), bool)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and np.all(V[j] >= V[i]) and np.any(V[j] > V[i]):
                dominated[i] = True
    front = [n for n, d in zip(names, dominated) if not d]

    ws = np.linspace(0, 1, 2001)
    by_budget = {}
    for budget in [1, 2, 3, 4]:
        kept = {n: 0 for n in names}
        for w in ws:
            u = w * V[:, 0] + (1 - w) * V[:, 1]
            for k in np.argsort(-u)[:budget]:
                kept[names[k]] += 1
        never = [n for n in names if kept[n] == 0]
        by_budget[budget] = dict(
            kept_fraction={n: kept[n] / len(ws) for n in names},
            never_selected=never,
            on_front_but_never=[n for n in front if n in never])

    return dict(cands=cands, front=front, by_budget=by_budget,
                concave_point="generalist")


# --------------------------------------------------------------------------

def main():
    # ---- Part A: the three-phase life --------------------------------------
    traces, seps = [], []
    for seed in range(N_SEEDS):
        L, tr = run(seed, lam_unit=0.10, record=True)
        traces.append(tr)
        seps.append(separates(L, seed)[0])
        print(f"  A seed {seed}: final units={int(L.alive.sum())}", flush=True)

    ts = [p["t"] for p in traces[0]]
    n_mean = np.mean([[p["n_units"] for p in tr] for tr in traces], axis=0)
    n_sem = np.std([[p["n_units"] for p in tr] for tr in traces], axis=0) / np.sqrt(N_SEEDS)
    err_mean = np.mean([[p["err"] for p in tr] for tr in traces], axis=0)

    # split status at the end of each phase, per seed
    phase_sep = {}
    for ph, n_ph in [("phase1", 1), ("phase2", 2), ("phase3", 3)]:
        vals = []
        for seed in range(N_SEEDS):
            L, _ = run(seed, lam_unit=0.10, n_phases=n_ph)
            vals.append(bool(separates(L, seed)[0]))
        phase_sep[ph] = vals
        print(f"  A {ph}: split in {sum(vals)}/{N_SEEDS} learners", flush=True)

    # ---- Part B: the lambda sweep ------------------------------------------
    lams = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    sweep = {}
    for lam in lams:
        vals = []
        for seed in range(N_SEEDS):
            L, _ = run(seed, lam_unit=lam, n_phases=2)
            vals.append(bool(separates(L, seed)[0]))
        sweep[str(lam)] = vals
        print(f"  B lam={lam:.2f}: split in {sum(vals)}/{N_SEEDS}", flush=True)

    gains = [measured_gain(s) for s in range(min(N_SEEDS, 6))]
    gain = float(np.nanmean(gains))

    # ---- Part C -------------------------------------------------------------
    par = pareto_demo()

    out = dict(t=ts, n_units_mean=n_mean.tolist(), n_units_sem=n_sem.tolist(),
               err_mean=err_mean.tolist(), phase=PHASE, n_seeds=N_SEEDS,
               phase_sep=phase_sep, sweep=sweep, lams=lams,
               measured_gain=gain, gains=gains, pareto=par)
    with open("results_exp3.json", "w") as f:
        json.dump(out, f)

    print("\nPART A -- split appears only when it pays")
    for ph in ["phase1", "phase2", "phase3"]:
        v = phase_sep[ph]
        print(f"  {ph}: S1/S2 separated in {sum(v)}/{len(v)} learners")

    print("\nPART B -- phase boundary vs the measured value of the split")
    print(f"  measured accuracy gain from splitting: {gain:.3f}")
    for lam in lams:
        v = sweep[str(lam)]
        bar = "#" * sum(v)
        print(f"  lambda={lam:.2f} {'(< gain)' if lam < gain else '(> gain)':9s} "
              f"split {sum(v)}/{len(v)}  {bar}")

    print("\nPART C -- scalar utility cannot reach a concave Pareto front")
    print(f"  Pareto front: {par['front']}")
    for budget, d in par["by_budget"].items():
        print(f"  budget {budget}: on the front yet never selected under ANY "
              f"linear weighting -> {d['on_front_but_never'] or 'none'}")


if __name__ == "__main__":
    main()
