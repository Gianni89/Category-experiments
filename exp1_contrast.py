"""
Experiment 1 -- Contrast-shaped boundaries.
===========================================

Post 1's signature prediction: identical positive instances plus systematically
different contrast neighbourhoods yield different category boundaries.  The
brief (Section 25) demands more than permission -- the model should explain WHY.

MECHANISM (derived, not assumed).  Precision follows gradient ascent on
U = V - lam*K with K = sum_k beta_k:

    delta b_k  =  eta_b * (x_k * err  -  lam_b * beta_k)

Errors while a unit is active push its precision up; holding precision costs in
proportion to the precision held.  The fixed point is therefore

    beta_k*  ~  E[x_k * err] / lam_b

i.e. equilibrium precision is proportional to the residual error rate the unit
is still generating.  A nearer contrast class produces more errors at any given
width, so equilibrium precision is higher and the category is narrower.  Width
and boundary position are consequences of the capacity constraint, not
stipulations -- which is what Section 25 asks for and the prose formalism does
not supply.

DESIGN.  A positive class P sits at 0 with a contrast class FIXED at -2.5 and a
second contrast class SWEPT on the right.  Within a replicate the positive
exemplar stream is byte-identical across every condition -- same draws, same
order, same learner initialisation -- so every difference measured is caused by
the right-hand contrast alone.  Each learner's own left flank is its internal
control, which turns the prediction into a within-subject one.

PREDICTIONS, stated before running.
  (a) The right boundary moves monotonically inward as the right contrast nears.
  (b) The category narrows on the contrast side specifically, so the asymmetry
      between the two sides is monotone in delta and crosses ZERO at
      delta = 2.5, where the two flanks are symmetric.
  (c) The prototype moves far less than the boundary.  If it moves as much,
      this is a prototype effect and the "edge" reading gains nothing.

MEASURE CHANGED AFTER SEEING THE DATA -- declared, because it matters.
Prediction (b) was first operationalised as the half-width of the positive
unit's normalised activation bump.  That measure is confounded: x is a softmax
over units, so bringing ANY competitor closer suppresses the bump on both
sides at once, and the half-widths move together.  It was replaced by the
"reach" -- the distance from the prototype to the point where the classifier
actually switches category -- which is both unconfounded and the quantity the
philosophical claim is about (how far the classificatory capacity extends).
The zero-crossing prediction was made in advance and does not depend on which
of the two measures is used.  Both are reported below.

WHAT WOULD FALSIFY.  Flat or non-monotonic (a) or (b); a zero-crossing away
from the symmetric point; comparable travel of prototype and boundary.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from jars_model import Params, Learner


SIGMA = 0.40
LEFT = -2.50                       # fixed contrast flank (internal control)
DELTAS = [1.00, 1.50, 2.00, 2.50, 3.00, 3.50]
GRID = np.arange(-5.0, 6.0 + 1e-9, 0.01)

QUICK = "--quick" in sys.argv
N_TRIALS = 3000 if QUICK else 9000
N_SEEDS = 3 if QUICK else 12


def make_streams(seed: int, n: int):
    """Fixed per replicate: positives and the *shape* of each contrast cloud."""
    rng = np.random.default_rng(10_000 + seed)
    u = rng.random(n)
    kind = np.where(u < 0.50, 0, np.where(u < 0.75, 1, 2))   # 0=P, 1=left, 2=right
    p_items = rng.normal(0.0, SIGMA, size=(n, 2))
    l_items = rng.normal(0.0, SIGMA, size=(n, 2)) + np.array([LEFT, 0.0])
    r_shape = rng.normal(0.0, SIGMA, size=(n, 2))
    return kind, p_items, l_items, r_shape


def run_condition(seed: int, delta: float, lam_b: float | None = None):
    p = Params(n_app=2, n_ctx=0, N=6, C=0, n_act=2)
    if lam_b is not None:
        p.lam_b = lam_b
    rng = np.random.default_rng(seed)                     # identical init across conditions
    L = Learner(p, rng, n_init_units=3)

    kind, p_items, l_items, r_shape = make_streams(seed, N_TRIALS)

    p_seen = []
    for t in range(N_TRIALS):
        if kind[t] == 0:
            s, correct = p_items[t], 0                    # approach
            p_seen.append(s)
        elif kind[t] == 1:
            s, correct = l_items[t], 1                    # avoid
        else:
            s, correct = r_shape[t] + np.array([delta, 0.0]), 1
        L.step(s, correct)

    L.frozen = True
    return L, np.array(p_seen)


def measure(L: Learner, p_seen: np.ndarray, delta: float):
    """All measurements are taken on the frozen structure: fast timescale only."""
    acts = np.array([L.settle(s)[0] for s in p_seen[-400:]]).mean(axis=0)
    kP = int(np.argmax(acts))

    probes = np.stack([GRID, np.zeros_like(GRID)], axis=1)
    X = np.array([L.settle(s)[0] for s in probes])
    Z = L.A @ X.T
    dz = Z[1] - Z[0]                                      # > 0 means "avoid"

    def crossing(lo_idx, hi_idx, rising=True):
        seg = dz[lo_idx:hi_idx]
        sgn = np.sign(seg)
        idxs = np.where(np.diff(sgn) > 0)[0] if rising else np.where(np.diff(sgn) < 0)[0]
        if not len(idxs):
            return np.nan
        i = idxs[-1] if not rising else idxs[0]
        i += lo_idx
        denom = dz[i + 1] - dz[i]
        frac = -dz[i] / denom if abs(denom) > 1e-12 else 0.0
        return float(GRID[i] + frac * (GRID[i + 1] - GRID[i]))

    zero = int(np.argmin(np.abs(GRID)))
    b_right = crossing(zero, len(GRID) - 1, rising=True)
    b_left = crossing(0, zero, rising=False)

    xp = X[:, kP]
    peak = int(np.argmax(xp))
    half = 0.5 * xp[peak]

    def half_width(direction: int):
        idx = peak
        while 0 <= idx < len(GRID) and xp[idx] > half:
            idx += direction
        if idx < 0 or idx >= len(GRID):
            return np.nan
        return abs(GRID[idx] - GRID[peak])

    w_right = half_width(+1)
    w_left = half_width(-1)
    sens = np.abs(np.gradient(xp, GRID))
    right_half = sens[zero:]

    proto = float(L.w[kP, 0])
    return dict(
        delta=delta,
        b_right=b_right, b_left=b_left,
        b_right_rel_mid=b_right - delta / 2.0,
        # "reach": how far the classificatory capacity extends from the
        # prototype before the classifier switches category.  This, not the
        # half-width of the normalised activation bump, is the extent of the
        # concept; the bump's width is confounded by the softmax normalisation.
        reach_right=float(b_right - proto),
        reach_left=float(proto - b_left),
        asym_reach=float((proto - b_left) - (b_right - proto)),
        proto=proto,
        precision=float(np.exp(L.b[kP])),
        w_right=float(w_right), w_left=float(w_left),
        asymmetry=float(w_left - w_right),
        peak_pos=float(GRID[peak]),
        sens_peak_right=float(GRID[zero + int(np.argmax(right_half))]),
        xp=xp.tolist(), sens=sens.tolist(),
    )


KEYS = ["b_right", "b_left", "b_right_rel_mid", "reach_right", "reach_left",
        "asym_reach", "proto", "precision", "w_right", "w_left", "asymmetry",
        "peak_pos", "sens_peak_right"]


def main():
    results = {d: [] for d in DELTAS}
    for seed in range(N_SEEDS):
        for d in DELTAS:
            L, p_seen = run_condition(seed, d)
            results[d].append(measure(L, p_seen, d))
        print(f"  seed {seed} done", flush=True)

    # Robustness: does the qualitative pattern depend on the price of precision?
    robust = {}
    for lam in [0.010, 0.020, 0.040]:
        rows = []
        for seed in range(min(N_SEEDS, 6)):
            for d in DELTAS:
                Lr, ps = run_condition(seed, d, lam_b=lam)
                rows.append(measure(Lr, ps, d))
        by_d = {d: [r for r in rows if r["delta"] == d] for d in DELTAS}
        robust[str(lam)] = {k: [float(np.nanmean([r[k] for r in by_d[d]])) for d in DELTAS]
                            for k in ["b_right", "asym_reach", "reach_left", "proto", "precision"]}
        print(f"  robustness lam_b={lam} done", flush=True)

    out = {"grid": GRID.tolist(), "deltas": DELTAS, "sigma": SIGMA, "left": LEFT,
           "n_trials": N_TRIALS, "n_seeds": N_SEEDS, "robust": robust, "runs": {}}
    for d in DELTAS:
        rs = results[d]
        agg = {}
        for key in KEYS:
            vals = np.array([r[key] for r in rs], dtype=float)
            n_ok = max(int(np.sum(~np.isnan(vals))), 1)
            agg[key] = dict(mean=float(np.nanmean(vals)),
                            sem=float(np.nanstd(vals) / np.sqrt(n_ok)),
                            vals=vals.tolist())
        agg["xp_mean"] = np.nanmean(np.array([r["xp"] for r in rs]), axis=0).tolist()
        agg["sens_mean"] = np.nanmean(np.array([r["sens"] for r in rs]), axis=0).tolist()
        out["runs"][str(d)] = agg

    with open("results_exp1.json", "w") as f:
        json.dump(out, f)

    print("\ndelta | b_right  b_left | proto | prec | reach_L reach_R | asym_reach")
    for d in DELTAS:
        a = out["runs"][str(d)]
        print(f"{d:4.2f}  | {a['b_right']['mean']:6.3f}  {a['b_left']['mean']:6.3f} "
              f"| {a['proto']['mean']:+5.3f} | {a['precision']['mean']:5.2f} "
              f"| {a['reach_left']['mean']:6.3f}  {a['reach_right']['mean']:6.3f} "
              f"| {a['asym_reach']['mean']:+6.3f} +/- {a['asym_reach']['sem']:.3f}")

    m = lambda k: np.array([out["runs"][str(d)][k]["mean"] for d in DELTAS])
    pr, bd, asym = m("proto"), m("b_right"), m("asym_reach")
    print(f"\nprototype travel: {pr.max()-pr.min():.3f} | boundary travel: {bd.max()-bd.min():.3f} "
          f"| ratio {(bd.max()-bd.min())/max(pr.max()-pr.min(),1e-9):.1f}x")
    print(f"boundary monotone in delta?  {bool(np.all(np.diff(bd) > 0))}")
    print(f"asymmetry monotone in delta? {bool(np.all(np.diff(asym) < 0))}")
    print(f"asym_reach at symmetric point (delta=2.5): "
          f"{out['runs']['2.5']['asym_reach']['mean']:+.3f}")
    rl = m("reach_left")
    print(f"reach_left (fixed flank!) across conditions: {rl.min():.3f} -> {rl.max():.3f} "
          f"(range {rl.max()-rl.min():.3f})")
    print("\nrobustness over the price of precision (lam_b):")
    for lam, r in robust.items():
        a = np.array(r["asym_reach"]); bb = np.array(r["b_right"]); rl2 = np.array(r["reach_left"])
        print(f"  lam_b={lam}: boundary monotone {bool(np.all(np.diff(bb)>0))}; "
              f"asym {a[0]:+.2f} -> {a[-1]:+.2f}, monotone {bool(np.all(np.diff(a)<0))}; "
              f"opposite-flank reach range {rl2.max()-rl2.min():.3f}")


if __name__ == "__main__":
    main()
