"""
Experiment 21 -- Does "positive contribution" absorb generic enabling machinery?
===============================================================================

THE OBJECTION
  Experiment 19 concluded that membership of a capacity's organisation should be settled by
  positive coalition-averaged contribution.  A global attentional gain, a working-memory
  resource or a general arousal mechanism improves performance on every classification while
  carrying nothing distinctive of RED, DOG or DANGEROUS.  On the criterion as it stands, such
  a component belongs to every concept.  Is that the intended result, and if not, what
  distinguishes it from a component genuinely shared by RED and RIPE?

THE SYSTEM
  Three capacities over a three-dimensional input, each a thresholded readout:
      A (x0 > 0) from {a1, a2, s}      B (x1 > 0) from {b1, b2, s}      C (x2 > 0) from {c1, c2}
  No single unit can cross a readout's threshold alone, so the units of a capacity cooperate.
  Four further components:
      s       SHARED AND DISCRIMINATORY.  One unit feeding A's and B's readouts, tuned to the
              region where both are positive, so it carries information that bears on both.
      gflat   GENERIC, ACTIVE.  A unit with a receptive field so broad it responds to
              everything, feeding all three readouts.  It raises every drive toward threshold
              and discriminates nothing.  This is the realistic analogue of general arousal.
      gmult   GENERIC, MULTIPLICATIVE.  A global gain: when ablated every activation is
              multiplied by 0.55.  The analogue of an attentional precision parameter.
      gdenoise
              GENERIC, AND NOT A RECALIBRATION.  When ablated, every unit's activation picks
              up item-specific noise.  It improves all three capacities, carries no
              information about any category, and -- unlike the other two -- its effect cannot
              be undone by any global rescaling, because what it removes is not a bias but a
              variance.  This is the hard case for the distinction being proposed.
      two inert units, wired to nothing.
  The readout thresholds are set slightly high, so that gflat's content-free boost genuinely
  buys accuracy.  If it bought nothing there would be no objection to answer.

FOUR MEASUREMENTS
  contribution   exact Shapley value of each component for each capacity, as in exp19.
  breadth        the fraction of capacities to which a component contributes above eps.
  discriminative content
                 the AUC of the component's own activation against each capacity's target.
                 A component that merely amplifies has AUC 0.5; one that carries information
                 about the category does not.  A pure multiplier has no activation of its own,
                 and is reported as n/a -- which is itself the answer in that case.
  content-neutral repair
                 the test that matters.  Remove the component from a coalition, then repair
                 the system with a CAPACITY-NEUTRAL adjustment: one global multiplier and one
                 additive constant, the same two numbers for all three capacities, chosen to
                 maximise mean accuracy across them.  Nothing in this repair knows which
                 capacity is which or where any boundary lies.  Recovery is the share of the
                 loss that the repair gives back, averaged over random coalitions in the same
                 way the Shapley value is -- single ablation from the grand coalition is
                 useless here for the reason exp19 established, since the other components
                 mask the loss.

PREDICTIONS, STATED BEFORE RUNNING
  1. All three of s, gflat and gmult have positive Shapley values for every capacity they
     touch, so the criterion as it stands does include generic machinery.  The objection is
     sound and will be confirmed rather than dissolved.
  2. Breadth separates them only by degree: s contributes to 2 of 3 capacities, gflat and
     gmult to 3 of 3.  Breadth alone therefore cannot carry the distinction, because "shared
     by all" is the limit of "shared by two" and nothing qualitative happens on the way.
  3. Discriminative content separates them qualitatively: AUC is about 0.5 for gflat and
     undefined for gmult, and clearly above 0.5 for s on both A and B and for each ai on its
     own capacity.
  4. Content-neutral repair separates gflat and gmult from the discriminating components:
     their removal is repaired almost completely (recovery > 0.9) by two capacity-neutral
     numbers, while removing s or any ai is barely repaired at all (recovery < 0.3), because
     no global adjustment can restore information that is not there.
  5. gdenoise is the case where the two tests come apart.  It should have breadth 1.0 and no
     discriminative content, like the other two, but should NOT be content-neutrally
     repairable, because variance cannot be rescaled away.  If that is what happens, then the
     repair test catches only the calibration-like forms of generic machinery, and the
     distinction cannot be drawn by replaceability alone.
  6. The inert units may well carry information about the categories while contributing
     nothing, since a receptive field anywhere in the space correlates with some boundary.
     That would show discriminative content is necessary but not sufficient, and has to be
     conjoined with contribution rather than replacing it.

WHAT COUNTS AS FAILURE
  * If the repair test gives recovery above 0.5 for s or for any ai, it is not tracking
    discriminative content and should not be proposed as a criterion.
  * If it gives recovery below 0.7 for gflat or gmult, then generic machinery is not in fact
    content-neutrally replaceable in this system, and the proposed distinction has no purchase
    even on a case built to display it.
  * If both tests fail, the honest conclusion is that positive contribution cannot be refined
    in this direction, and the theory should accept generic enabling machinery as part of
    every realiser and distinguish realisation from individuation some other way.
"""

from __future__ import annotations

import itertools
import json
import sys
from math import factorial

import numpy as np

QUICK = "--quick" in sys.argv
N_ITEMS = 500 if QUICK else 1500
N_SEEDS = 2 if QUICK else 5
EPS = 0.02
GAIN_OFF = 0.55
NOISE_SD = 0.22          # what ablating gdenoise adds to every activation
N_COALITIONS = 20 if QUICK else 60
CAPS = ["A", "B", "C"]
GAMMA_GRID = np.round(np.arange(0.5, 3.01, 0.05), 3)
BETA_GRID = np.round(np.arange(-1.0, 1.01, 0.05), 3)


def unit(centre, prec):
    return dict(c=np.array(centre, float), p=np.array(prec, float))


def build():
    """Components 0..10; gmult is component 9 and has no receptive field of its own."""
    a1 = unit((1.0, -1.1, 0.0), (1.0, 0.18, 0.02))
    a2 = unit((1.0, 1.1, 0.0), (1.0, 0.18, 0.02))
    b1 = unit((-1.1, 1.0, 0.0), (0.18, 1.0, 0.02))
    b2 = unit((1.1, 1.0, 0.0), (0.18, 1.0, 0.02))
    c1 = unit((0.0, -1.1, 1.0), (0.02, 0.18, 1.0))
    c2 = unit((0.0, 1.1, 1.0), (0.02, 0.18, 1.0))
    s = unit((1.0, 1.0, 0.0), (0.45, 0.45, 0.02))          # shared, and discriminating
    gflat = unit((0.0, 0.0, 0.0), (0.004, 0.004, 0.004))   # responds to everything
    inert1 = unit((-1.6, -1.6, -1.6), (1.0, 1.0, 1.0))
    inert2 = unit((1.6, -1.6, -1.6), (1.0, 1.0, 1.0))
    units = [a1, a2, b1, b2, c1, c2, s, gflat, inert1, inert2]
    #        0   1   2   3   4   5   6   7      8       9
    W = np.zeros((3, len(units)))
    W[0, [0, 1, 6, 7]] = 1.0      # capacity A
    W[1, [2, 3, 6, 7]] = 1.0      # capacity B
    W[2, [4, 5, 7]] = 1.0         # capacity C
    theta = np.array([1.35, 1.35, 1.30])
    truth = {"A": {0, 1, 6}, "B": {2, 3, 6}, "C": {4, 5}}   # what the construction counts as
    names = ["a1", "a2", "b1", "b2", "c1", "c2", "s", "gflat", "in1", "in2",
             "gmult", "gdenoise"]
    return units, W, theta, truth, names


GMULT, GDENOISE = 10, 11


class System:
    def __init__(self, seed):
        self.units, self.W, self.theta, self.truth, self.names = build()
        rng = np.random.default_rng(21_000 + seed)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 3))
        self.A = np.zeros((N_ITEMS, len(self.units)))
        for k, u in enumerate(self.units):
            self.A[:, k] = np.exp(-np.sum(u["p"] * (self.X - u["c"]) ** 2, axis=1))
        self.n = len(self.units) + 2                      # + gmult + gdenoise
        self.A_noisy = np.clip(self.A + rng.normal(0, NOISE_SD, self.A.shape), 0, None)
        self.y = np.stack([self.X[:, 0] > 0, self.X[:, 1] > 0, self.X[:, 2] > 0], axis=1)
        self.cache = {}

    def drive(self, S):
        """Unscaled drive for a coalition: one matrix product, reusable over a (gamma, beta) grid."""
        S = set(S)
        scale = 1.0 if GMULT in S else GAIN_OFF
        A = self.A if GDENOISE in S else self.A_noisy
        S = sorted(S - {GMULT, GDENOISE})
        if not S:
            return np.zeros((N_ITEMS, 3))
        return scale * (A[:, S] @ self.W[:, S].T)

    def v(self, S):
        key = frozenset(S)
        if key not in self.cache:
            self.cache[key] = ((self.drive(S) > self.theta) == self.y).mean(axis=0)
        return self.cache[key]

    def best_neutral(self, S):
        """One multiplier and one constant, shared by all capacities, maximising mean accuracy."""
        D = self.drive(S)
        best, best_mean = None, -1.0
        for gamma in GAMMA_GRID:
            G = gamma * D
            for beta in BETA_GRID:
                acc = (((G + beta) > self.theta) == self.y).mean(axis=0)
                if acc.mean() > best_mean:
                    best_mean, best = acc.mean(), (float(gamma), float(beta), acc)
        return best


def shapley(sysm):
    n = sysm.n
    players = list(range(n))
    phi = np.zeros((3, n))
    for i in players:
        others = [q for q in players if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[:, i] += w * (sysm.v(set(S) | {i}) - sysm.v(S))
    return phi


def informativeness(scores, labels):
    """|AUC - 0.5|, folded so that the direction of the tuning does not matter."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0 or np.allclose(scores, scores[0]):
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    a = float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return abs(a - 0.5)


def replaceability(sysm, i, seed):
    """Coalition-averaged: how much of i's contribution survives content-neutral recalibration?

    For random coalitions S not containing i, the raw contribution is v(S+i) - v(S).  Both
    sides are then separately given their best capacity-neutral calibration -- one multiplier
    and one constant, shared by all capacities and chosen without reference to any boundary --
    and the calibrated contribution is v*(S+i) - v*(S).

    A component that is nothing but a recalibration has a calibrated contribution of zero: once
    the system is allowed to set its own gain and threshold, it adds nothing.  A component that
    carries information keeps its contribution however the system is calibrated.  Replaceability
    is 1 - (calibrated / raw), on the totals, so that coalitions where i matters dominate.
    """
    rng = np.random.default_rng(700 + seed + 13 * i)
    others = [q for q in range(sysm.n) if q != i]
    raw = np.zeros(3); cal = np.zeros(3); gammas = []; betas = []
    for _ in range(N_COALITIONS):
        p = rng.random()
        S = {q for q in others if rng.random() < p}
        raw += sysm.v(S | {i}) - sysm.v(S)
        g1, b1, with_i = sysm.best_neutral(S | {i})
        g0, b0, without = sysm.best_neutral(S)
        cal += with_i - without
        gammas += [g1, g0]; betas += [b1, b0]
    surv = [float(c / r) if r > 1e-3 else None for c, r in zip(cal, raw)]
    return dict(raw=(raw / N_COALITIONS).tolist(), calibrated=(cal / N_COALITIONS).tolist(),
                survival=surv, gamma=float(np.mean(gammas)), beta=float(np.mean(betas)))


def run(seed):
    S = System(seed)
    phi = shapley(S)
    res = dict(seed=seed, names=S.names, truth={k: sorted(v) for k, v in S.truth.items()},
               shapley=phi.tolist(), components=[])
    for i in range(S.n):
        act = None if i in (GMULT, GDENOISE) else S.A[:, i]
        row = dict(
            index=i, name=S.names[i],
            phi={c: float(phi[m, i]) for m, c in enumerate(CAPS)},
            member={c: bool(phi[m, i] >= EPS) for m, c in enumerate(CAPS)},
            breadth=float(np.mean([phi[m, i] >= EPS for m in range(3)])),
            info=None if act is None else {c: informativeness(act, S.y[:, m]) for m, c in enumerate(CAPS)},
            repair=replaceability(S, i, seed))
        res["components"].append(row)
    return res


def main():
    out = [run(s) for s in range(N_SEEDS)]
    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps=EPS, runs=out),
              open("results_exp21.json", "w"))

    names = out[0]["names"]

    def mean_over(f):
        return [float(np.mean([f(r["components"][i]) for r in out])) for i in range(len(names))]

    def info_of(i):
        vals = [r["components"][i]["info"] for r in out]
        if vals[0] is None:
            return None
        return max(float(np.mean([v[c] for v in vals])) for c in CAPS)

    def rep_of(i):
        rec = []
        for m in range(3):
            vals = [r["components"][i]["repair"]["survival"][m] for r in out]
            vals = [v for v in vals if v is not None]
            if vals:
                rec.append(float(np.mean(vals)))
        return rec

    print("CONTRIBUTION, BREADTH, DISCRIMINATIVE CONTENT, CONTENT-NEUTRAL REPAIR")
    print(f"  {'component':10s} {'phi A':>7s} {'phi B':>7s} {'phi C':>7s} {'breadth':>8s} "
          f"{'|AUC-.5|':>9s} {'survival A, B, C':>24s}")
    for i, nm in enumerate(names):
        p = [np.mean([r["components"][i]["phi"][c] for r in out]) for c in CAPS]
        br = np.mean([r["components"][i]["breadth"] for r in out])
        inf = info_of(i)
        rec = []
        for m in range(3):
            vals = [r["components"][i]["repair"]["survival"][m] for r in out]
            vals = [v for v in vals if v is not None]
            rec.append("  -  " if not vals else f"{np.mean(vals):5.2f}")
        print(f"  {nm:10s} " + " ".join(f"{x:7.3f}" for x in p) + f" {br:8.2f} " +
              ("      n/a" if inf is None else f"{inf:9.2f}") + "   " + ", ".join(rec))

    print("\n  survival is the share of a component's contribution that is still there after both")
    print("  sides are given their best content-neutral calibration (one global multiplier and one")
    print("  additive constant, shared by all capacities, chosen without reference to any")
    print("  boundary).  Near 0 means the component was nothing but a recalibration.  1 or more")
    print("  means none of what it does can be had that way.  '-' means no contribution there.")

    print("\nTHE THREE TESTS, SIDE BY SIDE")
    print(f"  {'component':10s} {'member by contribution':>23s} {'breadth':>8s} "
          f"{'carries content':>16s} {'content-neutrally replaceable':>30s}  {'two-clause':>10s}")
    for i, nm in enumerate(names):
        caps_in = [c for m, c in enumerate(CAPS)
                   if np.mean([r["components"][i]["phi"][c] for r in out]) >= EPS]
        inf = info_of(i)
        content = "no activation" if inf is None else ("yes" if inf > 0.05 else "no") + f" ({inf:.2f})"
        rec = rep_of(i)
        verdict = "n/a" if not rec else ("yes" if max(rec) < 0.3 else "no") + f" ({max(rec):+.2f})"
        br = np.mean([r["components"][i]["breadth"] for r in out])
        two_clause = "yes" if (caps_in and inf is not None and inf > 0.08) else "no"
        print(f"  {nm:10s} {', '.join(caps_in) if caps_in else 'none':>23s} {br:8.2f} "
              f"{content:>16s} {verdict:>30s}  {two_clause:>10s}")


if __name__ == "__main__":
    main()
