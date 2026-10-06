"""
Experiment 2 -- Constitution vs plasticity: the two-timescale dissociation.
==========================================================================

THE PROBLEM.  The brief's Section 11 proposes to test whether a relation is
constitutive of the present jar by asking whether "perturbing it predictably
reorganises application and characteristic cognitive consequences".  Section
24.I then asks why strong causal leverage should count as evidence of
constitution rather than merely entrenched belief, and gets no answer.

THE PROPOSAL.  Split the timescales and the question becomes two questions:

  CONSTITUTION  = fast-timescale counterfactual dependence with the slow
                  dynamics HELD FIXED.  Does ablating this relation change how
                  the classifier settles on the very next case?
  PLASTICITY    = slow-timescale drift.  How much does this relation change
                  under ordinary learning?

These are two axes, not one.  A relation can be highly modifiable and not
load-bearing, or load-bearing and rarely modified.

THE MECHANISM (Section 14's permeability, made concrete).  A relation R(J,c)
starts as a pure readout: computed FROM the settled classification, unable to
affect it.  A gate g_c in [0,1] controls whether consequence c feeds BACK into
the settling loop.  The slow dynamics raise g_c only when doing so improves
action reward, judged against feedback generated independently of the relation.
Migration from readout to loop is a datable dynamical event -- a belief
becoming part of a classificatory capacity.

THE ENVIRONMENT, built to separate entrenchment from constitution.
  Appearance:  A at 0.0, C at 0.75 (heavily confusable with A), B at 2.8.
  Task:        avoid iff DANGEROUS.  Danger groups {A,B} vs {C}: it CROSSCUTS
               appearance, so it can resolve the A/C ambiguity.
  Control:     a "parallel" consequence grouping {A,C} vs {B} -- exactly the
               grouping appearance already gives, so feeding it back resolves
               nothing.
  Both are equally frequent, equally predictable, equally cued, equally well
  learned and equally often used in inference.  If only the crosscutting one
  migrates, entrenchment is NOT what makes a relation concept-constitutive.

Context cues are available while a case is settled, but the action is read from
the settled classification alone -- context at encoding, decision afterwards.
So a cue can only help by shaping classification.

PART A: a population of relations, scored on both axes.
PART B: the SAME relation at three plasticity levels, to test the Section 11
        test itself with load-bearing held constant.

PREDICTIONS, stated before running.
  1. g rises for the crosscutting relation and stays near zero for the parallel
     one, despite identical entrenchment.
  2. Load-bearing and plasticity are close to uncorrelated across the
     population, and all four cells of the 2x2 are occupied.
  3. Part B: with load-bearing held constant, the Section 11 score falls as
     plasticity rises -- i.e. the Section 11 test reads unmodifiability as
     constitution.
"""

from __future__ import annotations

import copy
import json
import sys
import numpy as np

from jars_model import Params, Learner, js_divergence


QUICK = "--quick" in sys.argv
N_TRIALS = 6000 if QUICK else 16000
WARMUP = 3000
N_SEEDS = 3 if QUICK else 10
SNAP_EVERY = 500
S11_WINDOW = 500 if QUICK else 1500
S11_MARKS = [0, 25, 50, 100, 200, 400, 800, 1500]

SIGMA = 0.50
CENTRES = {"A": np.array([0.00, 0.0]),
           "C": np.array([1.00, 0.0]),      # confusable with A by appearance
           "B": np.array([3.00, 0.0])}
DANGER = {"A": 1, "B": 1, "C": 0}           # crosscuts appearance
PARALLEL = {"A": 1, "C": 1, "B": 0}         # parallels appearance
CUE_NOISE = 0.80

# Part A population.  One crosscutting relation only: several redundant copies
# would compete for the same job and none would migrate cleanly.
CONS = [
    ("danger",        "coherent", 0.100),
    ("parallel-fast", "parallel", 0.100),
    ("parallel-slow", "parallel", 0.008),
    ("random-fast",   "random",   0.100),
    ("flat-slow",     "constant", 0.008),
]
C = len(CONS)
IDX_DANGER, IDX_PARALLEL = 0, 1


def sample_item(rng, cons=CONS):
    cat = "ABC"[int(rng.integers(3))]
    app = CENTRES[cat] + rng.normal(0, SIGMA, 2)
    obs = np.zeros(len(cons))
    for i, (_, kind, _) in enumerate(cons):
        if kind == "coherent":
            obs[i] = DANGER[cat]
        elif kind == "parallel":
            obs[i] = PARALLEL[cat]
        elif kind == "random":
            obs[i] = float(rng.integers(2))
        else:
            obs[i] = 0.5
    cue = (2 * obs - 1) + rng.normal(0, CUE_NOISE, len(cons))
    s = np.concatenate([app, cue])
    correct = 1 if DANGER[cat] == 1 else 0      # avoid iff dangerous
    return s, correct, obs, cat


def make_stream(seed, n, cons=CONS):
    rng = np.random.default_rng(50_000 + seed)
    return [sample_item(rng, cons) for _ in range(n)]


def make_probes(seed, n_cons=C, n=90):
    """Probes: the three categories plus items straddling the A/C boundary."""
    rng = np.random.default_rng(90_000 + seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.45:
            app = np.array([rng.uniform(-0.4, 1.2), rng.normal(0, 0.2)])
        else:
            cat = "ABC"[int(rng.integers(3))]
            app = CENTRES[cat] + rng.normal(0, SIGMA, 2)
        out.append(np.concatenate([app, rng.normal(0, 1.0, n_cons)]))
    return np.array(out)


def base_params(n_cons=C):
    return Params(n_app=2, n_ctx=n_cons, N=6, C=n_cons, n_act=2,
                  eta_R=0.100, eta_g=0.20, cost_g=0.010)


def train(seed, cons=CONS, snapshots=True):
    p = base_params(len(cons))
    rng = np.random.default_rng(seed)
    L = Learner(p, rng, n_init_units=3)
    stream = make_stream(seed, N_TRIALS, cons)
    L.init_centres_from(np.array([s for s, _, _, _ in stream[:400]]))
    L.eta_R_c = np.full(len(cons), 0.100)

    snaps = []
    for t, (s, correct, obs, cat) in enumerate(stream):
        if t == WARMUP:
            L.eta_R_c = np.array([e for _, _, e in cons])
        L.step(s, correct, cons_obs=obs)
        if snapshots and t % SNAP_EVERY == 0:
            snaps.append(dict(t=t, g=L.g.copy(), R=L.R.copy(), L=L.L.copy()))
    if snapshots:
        snaps.append(dict(t=N_TRIALS, g=L.g.copy(), R=L.R.copy(), L=L.L.copy()))
    return L, snaps, stream


def lb_timecourse(seed):
    """Gate and load-bearing through training, for the two key relations."""
    p = base_params()
    rng = np.random.default_rng(seed)
    L = Learner(p, rng, n_init_units=3)
    stream = make_stream(seed, N_TRIALS)
    L.init_centres_from(np.array([s for s, _, _, _ in stream[:400]]))
    L.eta_R_c = np.full(C, 0.100)
    probes = make_probes(seed)[:50]

    rec = dict(t=[], g_danger=[], g_parallel=[], lb_danger=[], lb_parallel=[], acc=[])
    for t, (s, correct, obs, cat) in enumerate(stream):
        if t == WARMUP:
            L.eta_R_c = np.array([e for _, _, e in CONS])
        L.step(s, correct, cons_obs=obs)
        if t % SNAP_EVERY == 0:
            L.frozen = True
            rec["t"].append(t)
            rec["g_danger"].append(float(L.g[IDX_DANGER]))
            rec["g_parallel"].append(float(L.g[IDX_PARALLEL]))
            rec["lb_danger"].append(L.load_bearing(probes, "R", IDX_DANGER))
            rec["lb_parallel"].append(L.load_bearing(probes, "R", IDX_PARALLEL))
            rec["acc"].append(1.0 - float(np.mean(L._err_hist[-400:])))
            L.frozen = False
    return rec


def plasticity_from_snaps(snaps, kind, idx, tail=6):
    """Slow-timescale drift: mean change per trial over the last stretch."""
    use = snaps[-(tail + 1):]
    ds = []
    for a, b in zip(use[:-1], use[1:]):
        if kind == "R":
            d = float(np.linalg.norm(b["R"][:, idx] - a["R"][:, idx]))
        else:
            i, j = idx
            d = abs(b["L"][i, j] - a["L"][i, j]) + abs(b["L"][j, i] - a["L"][j, i])
        ds.append(d / (b["t"] - a["t"]))
    return float(np.mean(ds))


def s11_trajectory(L_trained, stream_tail, probes, kind, idx, marks):
    """The Section 11 test as written: perturb the relation, let the system go
    on LEARNING, then ask whether application has reorganised.  Returns the
    divergence at several points so repair is visible."""
    ctrl = copy.deepcopy(L_trained); ctrl.frozen = False
    pert = copy.deepcopy(L_trained); pert.frozen = False
    if kind == "R":
        pert.R[:, idx] = 0.0
    else:
        i, j = idx
        pert.L[i, j] = 0.0; pert.L[j, i] = 0.0

    def diverge():
        cf, pf = ctrl.frozen, pert.frozen
        ctrl.frozen = pert.frozen = True
        a = ctrl.classify_profile(probes); b = pert.classify_profile(probes)
        ctrl.frozen, pert.frozen = cf, pf
        return float(np.mean([js_divergence(u, v) for u, v in zip(a, b)]))

    traj, done = [], 0
    for m in marks:
        while done < m and done < len(stream_tail):
            s, correct, obs, cat = stream_tail[done]
            ctrl.step(s, correct, cons_obs=obs)
            pert.step(s, correct, cons_obs=obs)
            done += 1
        traj.append(diverge())
    return traj


# ---------------------------------------------------------------------------

def part_a():
    pop, tcs, summary = [], [], []
    for seed in range(N_SEEDS):
        L, snaps, stream = train(seed)
        L.frozen = True
        probes = make_probes(seed)
        tail = make_stream(20_000 + seed, S11_WINDOW)
        alive = np.where(L.alive)[0]

        rels = [("R", i, CONS[i][0], CONS[i][1]) for i in range(C)]
        for a_i in range(len(alive)):
            for b_i in range(a_i + 1, len(alive)):
                i, j = int(alive[a_i]), int(alive[b_i])
                rels.append(("L", (i, j), f"lateral {i}-{j}", "lateral"))

        for kind, idx, name, cls in rels:
            pop.append(dict(
                seed=seed, name=name, cls=cls, kind=kind,
                lb=L.load_bearing(probes, kind, idx),
                pl=plasticity_from_snaps(snaps, kind, idx),
                s11=s11_trajectory(L, tail, probes, kind, idx, [S11_WINDOW])[0],
                gate=float(L.g[idx]) if kind == "R" else None))

        summary.append(dict(seed=seed, g=L.g.tolist(),
                            final_err=float(np.mean(L._err_hist[-500:])),
                            n_units=int(L.alive.sum())))
        tcs.append(lb_timecourse(seed))
        print(f"  A seed {seed}: g_danger={L.g[IDX_DANGER]:.3f} "
              f"g_par={L.g[IDX_PARALLEL]:.3f} err={summary[-1]['final_err']:.3f}", flush=True)
    return pop, tcs, summary


def part_b():
    """Same relation, same job, three plasticity levels."""
    etas = [0.000, 0.008, 0.100]
    rows = []
    for seed in range(N_SEEDS):
        for eta in etas:
            cons = [("danger", "coherent", eta)] + list(CONS[1:])
            L, snaps, stream = train(seed, cons=cons)
            L.frozen = True
            probes = make_probes(seed)
            tail = make_stream(20_000 + seed, S11_WINDOW)
            marks = [m for m in S11_MARKS if m <= S11_WINDOW]
            rows.append(dict(
                seed=seed, eta=eta,
                gate=float(L.g[IDX_DANGER]),
                lb=L.load_bearing(probes, "R", IDX_DANGER),
                pl=plasticity_from_snaps(snaps, "R", IDX_DANGER),
                traj=s11_trajectory(L, tail, probes, "R", IDX_DANGER, marks),
                marks=marks))
        print(f"  B seed {seed} done", flush=True)
    return rows


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return corr(ra, rb)


def main():
    pop, tcs, summary = part_a()
    brows = part_b()

    out = dict(cons=[c[0] for c in CONS], pop=pop, timecourse=tcs,
               summary=summary, partb=brows, n_seeds=N_SEEDS,
               n_trials=N_TRIALS, warmup=WARMUP, s11_window=S11_WINDOW)
    with open("results_exp2.json", "w") as f:
        json.dump(out, f)

    def col(cls_, key):
        return np.array([r[key] for r in pop if r["cls"] == cls_ and r[key] is not None], float)

    print("\nPART A -- population of relations")
    print("  class      | n  |  gate   | load-bearing |  plasticity | S11 test")
    for cls_ in ["coherent", "parallel", "random", "constant", "lateral"]:
        lb = col(cls_, "lb")
        if not len(lb):
            continue
        g = col(cls_, "gate"); pl = col(cls_, "pl"); s = col(cls_, "s11")
        gs = f"{g.mean():.3f}" if len(g) else "  -- "
        print(f"  {cls_:9s}  | {len(lb):2d} |  {gs}  |   {lb.mean():.5f}    "
              f"|  {pl.mean():.2e} | {s.mean():.5f}")

    lb = [r["lb"] for r in pop]; pl = [r["pl"] for r in pop]
    print(f"\n  corr(load-bearing, plasticity)  r = {corr(lb, pl):+.3f}  "
          f"rho = {spearman(lb, pl):+.3f}   <- near zero means two axes, not one")

    gd = np.array([s["g"][IDX_DANGER] for s in summary])
    gp = np.array([s["g"][IDX_PARALLEL] for s in summary])
    print(f"  gate, crosscutting: {gd.mean():.3f} +/- {gd.std()/np.sqrt(len(gd)):.3f} | "
          f"parallel: {gp.mean():.3f} +/- {gp.std()/np.sqrt(len(gp)):.3f}")

    print("\nPART B -- same relation, three plasticity levels")
    print("   eta   | gate  | load-bearing | plasticity | S11 @0    S11 @end   repair")
    for eta in [0.000, 0.008, 0.100]:
        rs = [r for r in brows if r["eta"] == eta]
        g = np.mean([r["gate"] for r in rs])
        l_ = np.mean([r["lb"] for r in rs])
        p_ = np.mean([r["pl"] for r in rs])
        t0 = np.mean([r["traj"][0] for r in rs])
        tE = np.mean([r["traj"][-1] for r in rs])
        print(f"  {eta:.3f}  | {g:.3f} |   {l_:.5f}    |  {p_:.2e}  | {t0:.5f}  {tE:.5f}   "
              f"{100*(1 - tE/max(t0,1e-12)):5.1f}%")


if __name__ == "__main__":
    main()
