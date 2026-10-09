"""
Experiment 28 -- A counterexample to the routed-discriminatory-structure criterion
==================================================================================

THE COUNTEREXAMPLE
  The revision plan asks for a concrete counterexample to the routed criterion before any
  further experiment is run.  Here is one, and it is one the plan's own gloss predicts:

      "'generic' is a role, not an intrinsic type of component.  If an otherwise general-purpose
       resource is specifically recruited into the organisation that computes C, then in that
       role it can count as part of R_C."

  A SHARPENER is exactly that.  It is a modulator -- it carries no signal of its own and feeds no
  readout -- but it acts on one capacity's resources alone, cleaning the units that carry A and
  nothing else.  Generic in mechanism, specific in scope.  Intuitively, and on the plan's own
  wording, it has been recruited into A's organisation and belongs to it; a system's attention to
  hue is part of how it classifies colours, not part of the furniture of the room.

  But the routing test as exp27 states it must exclude it.  Its effect does not run through any
  readout, so it survives rewiring, so it is scored as background support.  If that is what
  happens, the criterion as written is too strong, and the question is whether a principled
  refinement separates the sharpener from the denoiser without reintroducing the comparative
  membership rules that were rejected for banning genuine sharing.

THE PROPOSED REFINEMENT, TESTED HERE
  Split the off-route score by what the alternative capacity is for.  Build alternative
  realisations of the SAME category, and alternative realisations of OTHER categories, and ask
  where a component's unrouted influence shows up:

      off-route, same category     does it help other ways of computing this very category?
      off-route, other categories  does it help classifications of something else?

  Generic support helps both: fidelity is fidelity whatever is being classified.  A resource
  recruited into one capacity's organisation helps only the first.  The comparative element then
  appears exactly where it belongs -- in sorting kinds of unrouted influence -- and never touches
  routed resources, so a genuinely shared feature is never asked a comparative question at all.

PREDICTIONS, STATED BEFORE RUNNING
  1. The sharpener contributes to A's discriminations and to nothing else: psi(A) clearly
     positive, psi(B) and psi(C) at zero.
  2. The routing test as written excludes it: its off-route score is clearly positive, because
     its effect never ran through a readout in the first place.  The counterexample stands.
  3. Splitting the off-route score separates it from the denoiser.  The sharpener's off-route
     influence is confined to alternative realisations of A; the denoiser's is spread evenly over
     alternative realisations of all three capacities.
  4. Every genuine member scores zero on both halves of the split, so the refinement costs
     nothing where the original test already worked, and the shared feature is unaffected
     because it is routed and never reaches this test.

WHAT COUNTS AS FAILURE
  * If the sharpener's off-route score is near zero, the routing test handles it already and
    there is no counterexample -- the criterion survives unamended and that is the better result
    for the plan.
  * If the denoiser's off-route influence also turns out to be concentrated on one capacity, the
    split separates nothing and should not be proposed.
  * If the sharpener contributes to B or C, it is not capacity-specific and the case is void.
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
N_ALT = 2 if QUICK else 5           # alternative realisations per capacity
EPS = 0.02
NOISE_SD = 0.55
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


def shapley(n, value, n_out=1):
    phi = np.zeros((n_out, n))
    for i in range(n):
        others = [q for q in range(n) if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[:, i] += w * (value(set(S) | {i}) - value(S))
    return phi


class System:
    def __init__(self, seed):
        rng = np.random.default_rng(28_000 + seed)
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
                      "denoiser", "sharpener"]
        self.DEN, self.SHARP = 8, 9
        self.A_UNITS = [0, 1]                 # the units the sharpener acts on, and only those
        self.n = len(self.names)

        self.X = rng.uniform(-2, 2, (N_ITEMS, 3))
        self.clean = np.zeros((N_ITEMS, len(self.units)))
        for k, u in enumerate(self.units):
            self.clean[:, k] = np.exp(-np.sum(u["p"] * (self.X - u["c"]) ** 2, axis=1))
        self.noise = rng.normal(0, NOISE_SD, self.clean.shape)
        self.y = np.stack([self.X[:, 0] > 0, self.X[:, 1] > 0, self.X[:, 2] > 0], axis=1)

        self.W = np.zeros((3, len(self.units)))
        self.W[0, [0, 1, 6]] = 1.0
        self.W[1, [2, 3, 6]] = 1.0
        self.W[2, [4, 5, 6]] = 1.0
        self.theta = np.array([1.30, 1.30, 1.30])

        # alternative realisations: random weightings over the same units, one set per target
        self.alts = []
        for m in range(3):
            for _ in range(N_ALT):
                self.alts.append((m, (rng.random(len(self.units)) < 0.5).astype(float)))
        self.cache = {}

    def activations(self, S):
        """Noise is removed globally by the denoiser, and on A's units alone by the sharpener."""
        A = self.clean + self.noise
        if self.DEN in S:
            A = self.clean.copy()
        elif self.SHARP in S:
            A = A.copy()
            A[:, self.A_UNITS] = self.clean[:, self.A_UNITS]
        return np.clip(A, 0, None)

    def drive(self, S, W):
        A = self.activations(set(S))
        live = [k for k in sorted(set(S)) if k < len(self.units)]
        if not live:
            return np.zeros((N_ITEMS, W.shape[0]))
        return A[:, live] @ W[:, live].T

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

    def disc_alt(self, S, k):
        m, w = self.alts[k]
        key = ("alt", k, frozenset(S))
        if key not in self.cache:
            d = self.drive(S, w[None, :])
            self.cache[key] = np.array([auc(d[:, 0], self.y[:, m])])
        return self.cache[key]


def run(seed):
    P = System(seed)
    phi = shapley(P.n, P.acc, 3)
    psi = shapley(P.n, P.disc, 3)

    alt_psi = np.zeros((len(P.alts), P.n))
    for k in range(len(P.alts)):
        alt_psi[k] = shapley(P.n, lambda S, k=k: P.disc_alt(S, k))[0]

    # the component's own capacity: the one it contributes most to, by psi
    own = [int(np.argmax(psi[:, i])) if psi[:, i].max() > EPS else None for i in range(P.n)]
    same, other = np.full(P.n, np.nan), np.full(P.n, np.nan)
    for i in range(P.n):
        s_vals, o_vals = [], []
        for k, (m, w) in enumerate(P.alts):
            if i < len(P.units) and w[i] != 0:
                continue                       # routed into this alternative: not an off-route case
            (s_vals if own[i] is not None and m == own[i] else o_vals).append(alt_psi[k, i])
        same[i] = float(np.mean(s_vals)) if s_vals else np.nan
        other[i] = float(np.mean(o_vals)) if o_vals else np.nan

    return dict(seed=seed, names=P.names, phi=phi.tolist(), psi=psi.tolist(),
                own=own, off_same=same.tolist(), off_other=other.tolist(),
                off_all=[float(np.nanmean([same[i], other[i]])) for i in range(P.n)])


def main():
    out = [run(s) for s in range(N_SEEDS)]
    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps=EPS, n_alt=N_ALT, runs=out),
              open("results_exp28.json", "w"))
    names = out[0]["names"]
    g = lambda key, i, m=None: float(np.nanmean([
        (r[key][m][i] if m is not None else r[key][i]) for r in out]))

    print("CONTRIBUTION, DISCRIMINABILITY, AND OFF-ROUTE INFLUENCE")
    print(f"  {'component':11s} {'phi A':>7s} {'phi B':>7s} {'phi C':>7s}   "
          f"{'psi A':>7s} {'psi B':>7s} {'psi C':>7s}   {'off-route':>10s}   "
          f"{'same cat.':>10s} {'other cat.':>11s}")
    for i, nm in enumerate(names):
        ph = " ".join(f"{g('phi', i, m):7.3f}" for m in range(3))
        ps = " ".join(f"{g('psi', i, m):7.3f}" for m in range(3))
        print(f"  {nm:11s} {ph}   {ps}   {g('off_all', i):10.3f}   "
              f"{g('off_same', i):10.3f} {g('off_other', i):11.3f}")

    print("\n  off-route: the component's discriminability score for alternative realisations")
    print("  whose readouts do not include it, split by whether the alternative computes the")
    print("  same category it serves or a different one.\n")

    print("VERDICTS")
    print(f"  {'component':11s} {'with a success clause':>22s} {'routed discriminatory':>22s} "
          f"{'+ split off-route':>18s}   {'built in as':>14s}")
    truth = {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "c1": "C", "c2": "C",
             "shared": "A, B, C", "inert": "none", "denoiser": "none", "sharpener": "A"}
    for i, nm in enumerate(names):
        off_all, off_other = g("off_all", i), g("off_other", i)
        disc = [c for m, c in enumerate(CAPS) if g("psi", i, m) >= EPS]
        # the criterion as the revision plan now states it: routed contribution to structure
        plan = disc if off_all < EPS else []
        # the same, but keeping the success clause the plan proposes dropping
        with_success = [c for m, c in enumerate(CAPS)
                        if g("psi", i, m) >= EPS and g("phi", i, m) >= EPS] if off_all < EPS else []
        split = disc if off_other < EPS else []
        f = lambda L: ", ".join(L) if L else "none"
        print(f"  {nm:11s} {f(with_success):>22s} {f(plan):>22s} {f(split):>18s}   {truth[nm]:>14s}")


if __name__ == "__main__":
    main()
