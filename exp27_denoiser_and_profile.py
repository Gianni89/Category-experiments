"""
Experiment 27 -- Two adversaries for the discriminability criterion
===================================================================

Experiment 26 proposed that membership requires contributing to a capacity's DISCRIMINATIONS
rather than merely to its successes, operationalised as a Shapley value on the capacity's AUC.
The revision plan accepts the direction and names two cases that would decide whether the
measure is the abstraction we want or the next architecture-specific proxy.  This builds both.

PART 1 -- GENERIC SUPPORT THAT IMPROVES ORDERING
  Gains and constant offsets cannot reorder cases, which is why they failed the discriminability
  test.  But generic machinery need not be order-preserving.  A denoiser, normaliser or
  attentional filter improves the fidelity of the signals reaching every classifier, and that
  does change the ordering -- for the better.  If such a mechanism earns a positive
  discriminability score, then contributing to discriminations is not sufficient to separate
  individuating organisation from generic support, and the false-positive problem survives.

  The denoiser here has no receptive field of its own, no readout weight, and no category-
  specific tuning.  When it is absent, every unit's activation picks up item-specific noise;
  when present, the activations are clean.  It helps all three capacities by the same mechanism.

  THE CANDIDATE REPAIR, measured alongside: a member's contribution should be ROUTED.  Build
  alternative capacities over the same resources whose readouts do NOT include the component
  under test, and ask whether it still improves their discriminability.  A feature cannot help a
  classification that does not use it.  A denoiser can help any classification whatever, because
  what it improves is the fidelity of the signals rather than the content of any one of them.
  Reported as psi_unrouted: the mean discriminability score over capacities that do not route it.

PART 2 -- STRUCTURE THAT MOVES THE PROFILE BUT NOT THE RANKING
  The framework's object is a graded application profile, and the contrast results turn on centre
  shifts, typicality asymmetry and within-category discrimination.  A resource could reshape that
  profile while leaving the positive-versus-negative ordering alone, and binary AUC would not see
  it.  The shaper built here raises the profile only for cases already deep inside the category,
  so it sharpens within-category structure without changing any cross-class comparison.

  Measured against a graded target as well as a binary one: the Shapley value on the Spearman
  correlation between the profile and the graded target.

PREDICTIONS, STATED BEFORE RUNNING
  1. The denoiser has a clearly positive discriminability score for all three capacities,
     comparable in size to a real member's.  The plan's first adversary succeeds and the
     criterion as stated in exp26 is insufficient.
  2. The denoiser's psi_unrouted is just as large as its psi on the capacities it actually
     serves, because routing is irrelevant to what it does.
  3. Every genuine member has psi_unrouted near zero: a1 cannot improve a classification whose
     readout does not include a1.  So routed discriminability separates them, and the further
     distinction the plan asks for is "contributes to the discriminatory structure OF THIS
     CATEGORY" rather than "improves the conditions under which discriminations are expressed".
  4. The shaper has a binary-AUC score near zero while contributing substantially on the graded
     measure.  So AUC is too coarse for the framework's own object, and the criterion should be
     stated over the application profile with AUC as the binary-task instrument.
  5. If 4 holds, the two parts point the same way: the philosophical clause is contribution to
     the capacity's discriminatory structure, and AUC-Shapley is one instrument for it, exactly
     as the revision plan proposes rather than as exp26 proposed.

WHAT COUNTS AS FAILURE
  * If the denoiser's discriminability score is negligible, prediction 1 is wrong, the exp26
    criterion survives unqualified, and the routing repair is unnecessary -- that would be the
    stronger result for the criterion and should be reported as plainly as the alternative.
  * If psi_unrouted is large for genuine members, the repair does not separate anything and
    should not be proposed.
  * If the shaper's binary score is substantial after all, AUC is less coarse than feared and
    part 2's worry is theoretical.
  * If the shaper contributes nothing on either measure, it is not a member and the case is void.
"""

from __future__ import annotations

import itertools
import json
import sys
from math import factorial

import numpy as np

QUICK = "--quick" in sys.argv
N_ITEMS = 600 if QUICK else 1600
N_SEEDS = 2 if QUICK else 5
N_ALT = 3 if QUICK else 6          # alternative capacities per component, for the routing test
EPS = 0.02
NOISE_SD = 0.55                    # what the denoiser removes (declared: raised after the pilot,
                                   # where it improved ordering but barely improved accuracy)
SLOPE = 2.2
CAPS = ["A", "B", "C"]


def unit(centre, prec):
    return dict(c=np.array(centre, float), p=np.array(prec, float))


def auc(scores, labels):
    pos = int(labels.sum()); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def shapley(n, value, n_out):
    phi = np.zeros((n_out, n))
    for i in range(n):
        others = [q for q in range(n) if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[:, i] += w * (value(set(S) | {i}) - value(S))
    return phi


# =============================================================== part 1
class Part1:
    """Three capacities, a shared feature, a pure gain, and a denoiser with no readout role."""

    def __init__(self, seed):
        rng = np.random.default_rng(27_000 + seed)
        a1 = unit((1.0, -1.1, 0.0), (1.0, 0.18, 0.02))
        a2 = unit((1.0, 1.1, 0.0), (1.0, 0.18, 0.02))
        b1 = unit((-1.1, 1.0, 0.0), (0.18, 1.0, 0.02))
        b2 = unit((1.1, 1.0, 0.0), (0.18, 1.0, 0.02))
        c1 = unit((0.0, -1.1, 1.0), (0.02, 0.18, 1.0))
        c2 = unit((0.0, 1.1, 1.0), (0.02, 0.18, 1.0))
        shared = unit((1.0, 1.0, 1.0), (0.30, 0.30, 0.30))
        inert = unit((-1.7, -1.7, -1.7), (1.0, 1.0, 1.0))
        self.units = [a1, a2, b1, b2, c1, c2, shared, inert]
        self.names = ["a1", "a2", "b1", "b2", "c1", "c2", "shared", "inert",
                      "pure gain", "denoiser"]
        self.GAIN, self.DEN = 8, 9
        self.n = len(self.names)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 3))
        self.clean = np.zeros((N_ITEMS, len(self.units)))
        for k, u in enumerate(self.units):
            self.clean[:, k] = np.exp(-np.sum(u["p"] * (self.X - u["c"]) ** 2, axis=1))
        self.noisy = np.clip(self.clean + rng.normal(0, NOISE_SD, self.clean.shape), 0, None)
        self.y = np.stack([self.X[:, 0] > 0, self.X[:, 1] > 0, self.X[:, 2] > 0], axis=1)
        self.W = np.zeros((3, len(self.units)))
        self.W[0, [0, 1, 6]] = 1.0
        self.W[1, [2, 3, 6]] = 1.0
        self.W[2, [4, 5, 6]] = 1.0
        self.theta = np.array([1.30, 1.30, 1.30])
        # alternative readouts for the routing test: random weightings over the same units
        self.alts = []
        for k in range(N_ALT * 3):
            w = (rng.random(len(self.units)) < 0.5).astype(float)
            tgt = self.X[:, k % 3] > 0
            self.alts.append((w, tgt))
        self.cache = {}

    def drive(self, S, W):
        S = set(S)
        A = self.clean if self.DEN in S else self.noisy
        gain = 1.45 if self.GAIN in S else 1.0
        live = [k for k in sorted(S) if k < len(self.units)]
        if not live:
            return np.zeros((N_ITEMS, W.shape[0]))
        return gain * (A[:, live] @ W[:, live].T)

    def acc(self, S):
        key = ("acc", frozenset(S))
        if key not in self.cache:
            self.cache[key] = ((self.drive(S, self.W) > self.theta) == self.y).mean(axis=0)
        return self.cache[key]

    def disc(self, S):
        key = ("auc", frozenset(S))
        if key not in self.cache:
            d = self.drive(S, self.W)
            self.cache[key] = np.array([auc(d[:, m], self.y[:, m]) for m in range(3)])
        return self.cache[key]

    def disc_alt(self, S, idx):
        w, tgt = self.alts[idx]
        key = ("alt", idx, frozenset(S))
        if key not in self.cache:
            d = self.drive(S, w[None, :])
            self.cache[key] = np.array([auc(d[:, 0], tgt)])
        return self.cache[key]


def run_part1(seed):
    P = Part1(seed)
    phi = shapley(P.n, P.acc, 3)
    psi = shapley(P.n, P.disc, 3)
    # routed test: for each component, the alternative capacities whose readout excludes it
    unrouted = np.full((P.n,), np.nan)
    routed = np.full((P.n,), np.nan)
    alt_psi = np.zeros((len(P.alts), P.n))
    for k in range(len(P.alts)):
        alt_psi[k] = shapley(P.n, lambda S, k=k: P.disc_alt(S, k), 1)[0]
    for i in range(P.n):
        off, on = [], []
        for k, (w, _) in enumerate(P.alts):
            if i >= len(P.units) or w[i] == 0:      # the gain and denoiser are never "routed"
                off.append(alt_psi[k, i])
            else:
                on.append(alt_psi[k, i])
        unrouted[i] = float(np.mean(off)) if off else np.nan
        routed[i] = float(np.mean(on)) if on else np.nan
    return dict(seed=seed, names=P.names,
                phi=phi.tolist(), psi=psi.tolist(),
                psi_unrouted=unrouted.tolist(), psi_routed=routed.tolist())


# =============================================================== part 2
class Part2:
    """One capacity with a graded target, and a member that only shapes the inside of it."""

    def __init__(self, seed):
        rng = np.random.default_rng(27_500 + seed)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 2))
        # the category is carried by x0; typicality INSIDE the category is carried by x1,
        # so that a resource shaping typicality is uncorrelated with the binary distinction
        self.y = self.X[:, 0] > 0
        self.t = 1.0 / (1.0 + np.exp(-1.6 * self.X[:, 1]))
        d1 = unit((0.8, 0.0), (0.9, 0.002))           # carry the category, and nothing about x1
        d2 = unit((1.6, 0.0), (0.9, 0.002))
        shaper = unit((0.0, 2.5), (0.002, 0.5))       # carries typicality, not category membership
        inert = unit((-1.8, -1.8), (1.0, 1.0))
        self.units = [d1, d2, shaper, inert]
        self.names = ["d1", "d2", "shaper", "inert"]
        self.n = len(self.units)
        self.A = np.zeros((N_ITEMS, self.n))
        for k, u in enumerate(self.units):
            self.A[:, k] = np.exp(-np.sum(u["p"] * (self.X - u["c"]) ** 2, axis=1))
        self.W = np.array([1.0, 1.0, 1.0, 0.0])
        self.theta = 0.55
        self.cache = {}

    def mu(self, S):
        S = sorted(S)
        d = self.A[:, S] @ self.W[S] if S else np.zeros(N_ITEMS)
        return 1.0 / (1.0 + np.exp(-SLOPE * (d - self.theta)))

    def binary(self, S):
        key = ("b", frozenset(S))
        if key not in self.cache:
            self.cache[key] = np.array([auc(self.mu(S), self.y)])
        return self.cache[key]

    def graded(self, S):
        key = ("g", frozenset(S))
        if key not in self.cache:
            self.cache[key] = np.array([spearman(self.mu(S), self.t)])
        return self.cache[key]

    def within(self, S):
        """Ordering structure inside the category only: the cases the capacity says yes to."""
        key = ("w", frozenset(S))
        if key not in self.cache:
            sel = self.y
            self.cache[key] = np.array([spearman(self.mu(S)[sel], self.t[sel])])
        return self.cache[key]


def run_part2(seed):
    P = Part2(seed)
    out = dict(seed=seed, names=P.names)
    out["psi_binary"] = shapley(P.n, P.binary, 1)[0].tolist()
    out["psi_graded"] = shapley(P.n, P.graded, 1)[0].tolist()
    out["psi_within"] = shapley(P.n, P.within, 1)[0].tolist()
    return out


def main():
    p1 = [run_part1(s) for s in range(N_SEEDS)]
    p2 = [run_part2(s) for s in range(N_SEEDS)]
    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps=EPS, noise_sd=NOISE_SD,
                   part1=p1, part2=p2), open("results_exp27.json", "w"))

    names = p1[0]["names"]
    g = lambda key, i, m=None: float(np.mean([
        (r[key][m][i] if m is not None else r[key][i]) for r in p1]))

    print("PART 1 -- does generic support that improves ordering earn discriminability credit?")
    print(f"  {'component':11s} {'phi A':>7s} {'phi B':>7s} {'phi C':>7s}   "
          f"{'psi A':>7s} {'psi B':>7s} {'psi C':>7s}   {'psi where NOT routed':>21s}")
    for i, nm in enumerate(names):
        ph = " ".join(f"{g('phi', i, m):7.3f}" for m in range(3))
        ps = " ".join(f"{g('psi', i, m):7.3f}" for m in range(3))
        un = np.mean([r["psi_unrouted"][i] for r in p1])
        print(f"  {nm:11s} {ph}   {ps}   {un:21.3f}")

    print("\n  psi where NOT routed: the component's discriminability score, averaged over")
    print("  alternative capacities built over the same resources whose readouts exclude it.")

    print("\nPART 1 VERDICTS")
    print(f"  {'component':11s} {'contribution':>13s} {'+ discriminability':>20s} "
          f"{'+ routed':>10s}   {'built in as':>14s}")
    truth = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C", "c2": "C",
             "shared": "A, B, C", "inert": "none", "pure gain": "none", "denoiser": "none"}
    for i, nm in enumerate(names):
        c0 = [c for m, c in enumerate(CAPS) if g("phi", i, m) >= EPS]
        c1 = [c for m, c in enumerate(CAPS) if g("phi", i, m) >= EPS and g("psi", i, m) >= EPS]
        un = np.mean([r["psi_unrouted"][i] for r in p1])
        c2 = c1 if un < EPS else []
        f = lambda L: ", ".join(L) if L else "none"
        print(f"  {nm:11s} {f(c0):>13s} {f(c1):>20s} {f(c2):>10s}   {truth[nm]:>14s}")

    print("\nPART 2 -- a member that reshapes the profile without moving the ranking")
    print(f"  {'component':11s} {'psi on binary AUC':>18s} {'psi on the graded target':>25s} "
          f"{'psi within the category':>24s}")
    for i, nm in enumerate(p2[0]["names"]):
        b = np.mean([r["psi_binary"][i] for r in p2])
        gr = np.mean([r["psi_graded"][i] for r in p2])
        w = np.mean([r["psi_within"][i] for r in p2])
        print(f"  {nm:11s} {b:18.3f} {gr:25.3f} {w:24.3f}")


if __name__ == "__main__":
    main()
