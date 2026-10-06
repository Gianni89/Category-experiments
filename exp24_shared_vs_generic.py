"""
Experiment 24 -- A shared discriminative resource against a rich generic modulator
==================================================================================

THE QUESTION
  Experiment 21 showed that positive contribution alone admits generic machinery, and that a
  discriminatory-content clause excludes it.  But the generic components there were weak
  adversaries: two had no activity of their own, and the third was flat and uninformative.
  The real adversary is a mechanism with rich, input-varying activity that CORRELATES with
  several classifications while playing a nonspecific supporting role -- gain, salience or
  difficulty modulation rather than a feature the system carves with.

  The further constraint, which rules out the obvious fix: the framework is meant to allow
  genuinely shared resources.  A component may contribute to RED and RIPE, carry discriminatory
  structure relevant to both, and belong to both.  So the criterion must separate rich generic
  support from genuine sharing WITHOUT excluding anything merely for being used widely.  Any
  rule of the form "membership requires more information for this capacity than for any other"
  would ban exactly the overlap the framework was designed to permit.

THE DESIGN, AND WHY IT IS THE SHARPEST VERSION OF THE TEST
  Three capacities over a three-dimensional input, each a thresholded readout over two
  cooperating units, as in exp21.  Two further components are added, and they have THE SAME
  ACTIVATION FUNCTION as each other -- the same broad receptive field over the positive octant,
  so that each is equally informative about all three targets:

      shared     feeds all three readouts additively, with a positive weight.  A feature that
                 three classifications carve with.  It ought to belong to all three.
      modulator  feeds no readout.  While present it multiplies every other unit's activation
                 by (1 + kappa * its own activation).  It helps all three capacities, its
                 activity is exactly as informative as `shared`'s, and it carves nothing.

  Because their activations are identical by construction, any measure computed from a
  component's own activity alone MUST give them the same score.  That is the point: it makes
  the limits of an information-based clause a matter of arithmetic rather than of luck, and it
  forces the question onto how the activity is used.

  gflat (a flat, uninformative always-on unit) and two inert units are kept from exp21 as
  controls at the other two corners: contribution without content, and content without
  contribution.

THE CANDIDATE DISCRIMINATORS
  contribution        exact Shapley value for each capacity.
  content             |AUC - 0.5| of the component's own activation against each target.
  differential content
                      content for this capacity minus the largest content for any other.  This
                      is the rule the reviewer and I converged on and then rejected on
                      philosophical grounds; it is computed here so the rejection can be shown
                      to be right on the numbers as well.
  engagement          the measure this experiment is really testing: is the component's
                      informative activity ROUTED INTO this classification, or does it only
                      modulate whatever others are doing?  Operationally, the solo profile --
                      the capacity's graded application profile when that component is the only
                      one available.  A feature produces a profile of its own that varies across
                      cases and aligns with the category; a modulator produces nothing at all by
                      itself, because there is nothing for it to multiply.  Reported as the
                      variation of the solo profile and its correlation with the target.
  perturbation        a second candidate, kept because it failed: perturb the component's own
                      state and ask whether the induced change in drive is explained by its own
                      activation or by the rest of the organisation's drive.  See the pilot
                      declaration below.
  intervention        the share of items whose classification flips when the component is
                      disabled, which is what "changes the application profile" comes to.

PREDICTIONS, STATED BEFORE RUNNING
  1. `shared` and `modulator` have identical content and identical differential content, to
     numerical precision.  This is analytic given the construction and verifies the build.
  2. Both have positive Shapley contributions to all three capacities.  So contribution plus
     content admits both, and the exp21 criterion cannot separate them.  The objection that
     prompted this experiment is therefore sound.
  3. Differential content does not separate them either, and would additionally exclude
     `shared` -- which is the correct component to keep -- because its content is equal across
     the three capacities and its differential is therefore about zero.  If that is what
     happens, differential information is confirmed unusable as a membership criterion, for the
     reason given in the revision plan rather than merely by assertion.
  4. The solo profile separates them completely: `shared` and the capacity-specific units
     produce a solo profile that varies across cases and correlates with the target, while
     `modulator` produces a solo profile with exactly zero variation, because a multiplier with
     nothing to multiply changes nothing.
  5. So the criterion that works is contribution + content + engagement, and none of the three
     is comparative across capacities.  `shared` is admitted to all three organisations and
     overlap is preserved; `modulator` is excluded everywhere; `gflat` is excluded on content;
     the inert units are excluded on contribution.

DECLARED AFTER THE PILOT (the predictions above were not changed)
  * The perturbation measure was the engagement measure in the pilot, and it does not work.  Two
    reasons, both worth recording.  First, in a system that contains a global modulator, every
    component's induced change is scaled by that modulator, so nothing is measured in isolation.
    Second, and more fundamentally, the modulator's activation is strongly correlated with the
    rest of the organisation's drive -- both peak on the same items -- so its induced change is
    well predicted by its own activation after all (R^2 0.94 against `shared`'s 0.98).  The
    measure is still computed and reported, because a failed candidate that looks plausible is
    worth showing.
  * The solo-profile measure replaces it.  It is also the better fit to the post's own wording:
    membership already requires that the classification "make use of" the resource and that its
    activity "help distinguish the cases"; the solo profile is what asks whether those two are
    the same use.
  * The system was retuned so that both candidates are genuinely adversarial.  In the pilot the
    shared unit contributed 0.016-0.022 and the modulator 0.009-0.015, around the membership
    threshold, so neither was a real test of anything.  The readout threshold and the shared
    unit's weight were raised until both contribute on the same scale as the capacity-specific
    units.  Nothing about the measures was tuned, and the ground truth is unchanged.

WHAT COUNTS AS FAILURE
  * If own-activity sufficiency does not separate `shared` from `modulator` -- both near 1, or
    both well below -- then the engagement measure fails and the distinction between shared
    structure and rich generic support cannot be drawn by intervention in this architecture.
    That is a real possibility and it would have to be reported: it would mean the distinction
    is structural rather than functional, and the post should say so instead of proposing a
    criterion it cannot operationalise.
  * If `shared` is excluded from any of the three capacities by any of the clauses, the proposal
    bans legitimate overlap and must be rejected whatever else it does.
  * If `modulator` turns out to contribute negligibly, the adversary is not adversarial and the
    case is void.
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
CONTENT_EPS = 0.08
SOLO_EPS = 0.10           # a solo profile counts as aligned with the category above this
DELTA = 0.10              # the multiplicative perturbation used for the failed measure
KAPPA = 0.95              # how strongly the modulator scales the rest of the organisation
W_SHARED = 1.60           # the shared feature's readout weight   (declared: raised after pilot)
THETA = 1.95              # the readout threshold                 (declared: raised after pilot)
SLOPE = 3.0               # sigmoid slope for the graded application profile
CAPS = ["A", "B", "C"]


def unit(centre, prec):
    return dict(c=np.array(centre, float), p=np.array(prec, float))


def build():
    a1 = unit((1.0, -1.1, 0.0), (1.0, 0.18, 0.02))
    a2 = unit((1.0, 1.1, 0.0), (1.0, 0.18, 0.02))
    b1 = unit((-1.1, 1.0, 0.0), (0.18, 1.0, 0.02))
    b2 = unit((1.1, 1.0, 0.0), (0.18, 1.0, 0.02))
    c1 = unit((0.0, -1.1, 1.0), (0.02, 0.18, 1.0))
    c2 = unit((0.0, 1.1, 1.0), (0.02, 0.18, 1.0))
    broad = unit((1.0, 1.0, 1.0), (0.30, 0.30, 0.30))      # used twice, identically
    gflat = unit((0.0, 0.0, 0.0), (0.004, 0.004, 0.004))
    in1 = unit((-1.6, -1.6, -1.6), (1.0, 1.0, 1.0))
    in2 = unit((1.6, -1.6, -1.6), (1.0, 1.0, 1.0))
    #        0   1   2   3   4   5   6=shared  7=gflat  8,9=inert   10=modulator
    units = [a1, a2, b1, b2, c1, c2, broad, gflat, in1, in2]
    W = np.zeros((3, len(units)))
    W[0, [0, 1, 7]] = 1.0
    W[1, [2, 3, 7]] = 1.0
    W[2, [4, 5, 7]] = 1.0
    W[:, 6] = W_SHARED          # the shared feature's weight into all three readouts
    theta = np.array([THETA, THETA, THETA])
    names = ["a1", "a2", "b1", "b2", "c1", "c2", "shared", "gflat", "in1", "in2", "modulator"]
    truth = {"A": {0, 1, 6}, "B": {2, 3, 6}, "C": {4, 5, 6}}
    return units, W, theta, names, truth


MOD = 10
SHARED = 6


class System:
    def __init__(self, seed):
        self.units, self.W, self.theta, self.names, self.truth = build()
        rng = np.random.default_rng(24_000 + seed)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 3))
        self.A = np.zeros((N_ITEMS, len(self.units)))
        for k, u in enumerate(self.units):
            self.A[:, k] = np.exp(-np.sum(u["p"] * (self.X - u["c"]) ** 2, axis=1))
        self.g = self.A[:, SHARED].copy()          # the modulator's activation: the same function
        self.n = len(self.units) + 1
        self.y = np.stack([self.X[:, 0] > 0, self.X[:, 1] > 0, self.X[:, 2] > 0], axis=1)
        self.cache = {}

    def drive(self, S, scale_mod=1.0, boost=None):
        """Drive on each capacity for coalition S.

        `boost` lets a component's own state be perturbed without touching anything else:
        a dict {unit index: factor}, or the string 'mod' to perturb the modulator's strength.
        """
        S = set(S)
        A = self.A
        if boost:
            A = A.copy()
            for k, f in boost.items():
                A[:, k] = A[:, k] * f
        gain = 1.0 + KAPPA * scale_mod * self.g if MOD in S else np.ones(N_ITEMS)
        S = sorted(S - {MOD})
        if not S:
            return np.zeros((N_ITEMS, 3))
        return gain[:, None] * (A[:, S] @ self.W[:, S].T)

    def v(self, S):
        key = frozenset(S)
        if key not in self.cache:
            self.cache[key] = ((self.drive(S) > self.theta) == self.y).mean(axis=0)
        return self.cache[key]

    def profile(self, S):
        return (self.drive(S) > self.theta)

    def mu(self, S):
        """The graded application profile: what the post calls mu_C(x)."""
        z = SLOPE * (self.drive(S) - self.theta)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


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
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0 or np.allclose(scores, scores[0]):
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    a = float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return abs(a - 0.5)


def r2(y, x):
    """Share of the variance of y explained by a linear function of x."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def solo_profile(sysm, i):
    """What the capacity's application profile looks like when i is the only resource present."""
    alone = sysm.mu({i})
    empty = sysm.mu(set())
    out = {}
    for m, cap in enumerate(CAPS):
        v = alone[:, m]
        sd = float(np.std(v))
        out[cap] = dict(variation=sd,
                        alignment=0.0 if sd < 1e-9 else float(np.corrcoef(v, sysm.y[:, m])[0, 1]),
                        shift=float(np.mean(v) - np.mean(empty[:, m])))
    return out


def engagement(sysm, i):
    """Perturb i's own state by DELTA and ask what the induced change in drive depends on."""
    everyone = set(range(sysm.n))
    base = sysm.drive(everyone)
    if i == MOD:
        pert = sysm.drive(everyone, scale_mod=1.0 + DELTA)
        own = sysm.g
    else:
        pert = sysm.drive(everyone, boost={i: 1.0 + DELTA})
        own = sysm.A[:, i]
    d = pert - base
    # the rest of the organisation's drive, with i taken out
    rest = sysm.drive(everyone - {i})
    out = {}
    for m, cap in enumerate(CAPS):
        out[cap] = dict(own_r2=r2(d[:, m], own), rest_r2=r2(d[:, m], rest[:, m]),
                        size=float(np.mean(np.abs(d[:, m]))))
    return out


def intervention(sysm, i):
    everyone = set(range(sysm.n))
    full = sysm.profile(everyone)
    cut = sysm.profile(everyone - {i})
    return {cap: float(np.mean(full[:, m] != cut[:, m])) for m, cap in enumerate(CAPS)}


def run(seed):
    S = System(seed)
    phi = shapley(S)
    rows = []
    for i in range(S.n):
        act = S.g if i == MOD else S.A[:, i]
        info = {cap: informativeness(act, S.y[:, m]) for m, cap in enumerate(CAPS)}
        diff = {cap: info[cap] - max(info[c] for c in CAPS if c != cap) for cap in CAPS}
        eng = engagement(S, i)
        rows.append(dict(index=i, name=S.names[i],
                         phi={cap: float(phi[m, i]) for m, cap in enumerate(CAPS)},
                         info=info, diff_info=diff, engagement=eng,
                         solo=solo_profile(S, i), flips=intervention(S, i)))
    return dict(seed=seed, names=S.names, truth={k: sorted(v) for k, v in S.truth.items()},
                components=rows)


def main():
    out = [run(s) for s in range(N_SEEDS)]
    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps=EPS, content_eps=CONTENT_EPS,
                   delta=DELTA, kappa=KAPPA, runs=out), open("results_exp24.json", "w"))
    names = out[0]["names"]
    avg = lambda f: [float(np.mean([f(r["components"][i]) for r in out])) for i in range(len(names))]

    print("CONTRIBUTION AND CONTENT")
    print(f"  {'component':11s} {'phi A':>7s} {'phi B':>7s} {'phi C':>7s}   "
          f"{'content A':>9s} {'content B':>9s} {'content C':>9s}   {'diff. content (max)':>19s}")
    for i, nm in enumerate(names):
        p = [np.mean([r["components"][i]["phi"][c] for r in out]) for c in CAPS]
        inf = [np.mean([r["components"][i]["info"][c] for r in out]) for c in CAPS]
        dif = max(np.mean([r["components"][i]["diff_info"][c] for r in out]) for c in CAPS)
        print(f"  {nm:11s} " + " ".join(f"{x:7.3f}" for x in p) + "   " +
              " ".join(f"{x:9.3f}" for x in inf) + f"   {dif:19.3f}")

    print("\nENGAGEMENT: the solo profile -- what the capacity does when this is the only resource")
    print(f"  {'component':11s} {'variation of the solo profile':>31s}   {'alignment with the target':>27s}")
    for i, nm in enumerate(names):
        var, al = [], []
        for c in CAPS:
            var.append(f"{np.mean([r['components'][i]['solo'][c]['variation'] for r in out]):5.3f}")
            al.append(f"{np.mean([r['components'][i]['solo'][c]['alignment'] for r in out]):+5.2f}")
        print(f"  {nm:11s} {', '.join(var):>31s}   {', '.join(al):>27s}")

    print("\nTHE MEASURE THAT FAILED: what the induced change in drive depends on")
    print(f"  {'component':11s} {'own-activity R2 (A, B, C)':>30s}   {'rest-of-organisation R2':>26s}"
          f"   {'items flipped by ablation':>26s}")
    for i, nm in enumerate(names):
        own, rest, fl = [], [], []
        for c in CAPS:
            o = [r["components"][i]["engagement"][c]["own_r2"] for r in out]
            s = [r["components"][i]["engagement"][c]["rest_r2"] for r in out]
            o = [x for x in o if not np.isnan(x)]; s = [x for x in s if not np.isnan(x)]
            own.append("  -  " if not o else f"{np.mean(o):5.2f}")
            rest.append("  -  " if not s else f"{np.mean(s):5.2f}")
            fl.append(f"{np.mean([r['components'][i]['flips'][c] for r in out]):5.2f}")
        print(f"  {nm:11s} {', '.join(own):>30s}   {', '.join(rest):>26s}   {', '.join(fl):>26s}")

    print("\nVERDICTS  (member = contribution >= %.2f, content >= %.2f, solo alignment >= %.2f)"
          % (EPS, CONTENT_EPS, SOLO_EPS))
    print(f"  {'component':11s} {'by contribution alone':>22s} {'+ content':>22s} {'+ engagement':>22s}")
    for i, nm in enumerate(names):
        c1 = [c for c in CAPS if np.mean([r["components"][i]["phi"][c] for r in out]) >= EPS]
        inf = {c: np.mean([r["components"][i]["info"][c] for r in out]) for c in CAPS}
        c2 = [c for c in c1 if inf[c] >= CONTENT_EPS]
        c3 = [c for c in c2
              if abs(np.mean([r["components"][i]["solo"][c]["alignment"] for r in out])) >= SOLO_EPS]
        f = lambda L: ", ".join(L) if L else "none"
        print(f"  {nm:11s} {f(c1):>22s} {f(c2):>22s} {f(c3):>22s}")

    print("\n  ground truth: shared belongs to A, B and C; modulator and gflat to none;")
    print("  a1/a2 to A, b1/b2 to B, c1/c2 to C; in1/in2 to none.")


if __name__ == "__main__":
    main()
