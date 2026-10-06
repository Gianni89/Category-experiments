"""
Experiment 26 -- Does the engagement clause exclude genuinely conjunctive members?
==================================================================================

THE CHALLENGE
  The revision plan accepts a third membership clause but refuses my wording for it.  Defining
  engagement as "the resource, taken on its own, makes a difference to how the category
  applies" risks excluding a cooperative or conjunctive member that has a classificatory role
  only in combination.  The plan asks whether engagement can be stated cleanly as something
  more abstract -- classificatory routing or role -- without smuggling in the very
  concept/non-concept boundary the criterion is meant to explain.  This experiment builds the
  case that decides it.

THE ADVERSARY
  A capacity X whose target is (x0 > 0) XOR (x1 > 0), realised by two components u1 and u2 that
  enter the readout only through their PRODUCT.  Each one is built so that:
      * its own activation carries no information at all about X's target -- being high on x0
        tells you nothing about an exclusive disjunction unless you also know x1;
      * on its own it contributes nothing to the readout, because a product with one term
        missing is zero;
  and yet it is plainly constitutive of X: remove it and the capacity is gone.
  So u1 and u2 should fail the marginal content clause AND the solo-profile clause, while being
  members by construction.  If that is what happens, the problem is worse than the plan says,
  because it defeats the SECOND clause as well as the third.

  Alongside it: two ordinary capacities A and B carried by cooperating additive units, a feature
  shared between them, the rich generic modulator from exp24 (same activation as the shared
  feature, but multiplying everything rather than feeding any readout), and an inert unit.

THE FOUR MEASUREMENTS
  contribution        exact Shapley value, as throughout.
  marginal content    |AUC - 0.5| of the component's own activation against the target.
  solo alignment      the exp24 clause: the alignment of the capacity's application profile with
                      the target when this component is the only resource present.
  discriminability    the candidate for an architecture-independent statement of engagement:
                      the Shapley value computed on the capacity's DISCRIMINABILITY -- the AUC
                      of its application profile against its target -- rather than on its
                      accuracy.  The thought: generic support helps a classification get more
                      cases right, while a resource the classification carves with changes which
                      cases it ranks above which.  A pure gain or a constant offset cannot
                      reorder cases at all, so it should contribute to accuracy and nothing to
                      discriminability.

PREDICTIONS, STATED BEFORE RUNNING
  1. u1 and u2 have marginal content near 0 and solo alignment near 0 for capacity X, while
     contributing substantially to it.  The two-clause criterion and the three-clause version I
     proposed both exclude them.  The plan's caution is therefore correct, and it bites the
     content clause as well as the engagement clause.
  2. The discriminability measure admits the conjunctive members, because the product term
     genuinely reorders cases, and it excludes a constant offset and a pure gain, because
     neither can reorder anything.
  3. It does NOT exclude this modulator, whose gain varies with the input and is correlated with
     the targets, so that multiplying by it does reorder cases and does improve discriminability.
     That is the prediction I am least confident of and the one the experiment is really for.
  4. If 3 holds, then no measure in this family -- marginal content, differential content,
     conditional content, discriminability contribution -- separates a rich target-correlated
     modulator from a genuinely shared feature, and only solo operation does.  The question then
     is not which further clause to add but what the framework should say, and the honest answer
     is that a modulator whose variation carries a category's distinction is doing classificatory
     work for that category whatever the implementation calls it.
  5. Solo alignment is the special case in which the conditioning set is empty.  That is why it
     worked in exp24, where no member was conjunctive, and why it fails here.

WHAT COUNTS AS FAILURE
  * If the conjunctive members turn out to have marginal content or a solo profile after all,
    the adversary is not adversarial and the case says nothing.
  * If discriminability contribution excludes a genuine member, it is not usable as a clause.
  * The experiment cannot fail in the ordinary sense on the modulator: either it is separated,
    in which case there is an architecture-independent clause, or it is not, in which case the
    framework has to decide what to say.  Both outcomes are reportable and the second is more
    interesting.

DECLARED AFTER THE PILOT (the predictions above were rewritten; here is exactly what happened)
  The pilot built two conditional measures -- content within strata of the other members' state,
  and a perturbation whose alignment with the target was reported separately for items the rest
  of the organisation gets right and gets wrong.  Both failed, for reasons worth recording:
    * Conditional content punishes genuine sharing.  Conditioning on the other members' state
      removes exactly the information a redundant or correlated member carries, so the shared
      feature scored 0.03 and 0.06 -- lower than the inert unit -- and was excluded.
    * The right/wrong split is often undefined.  Where the rest of the organisation produces a
      constant verdict, the items it gets wrong are exactly the items of one class, so the
      correlation within that stratum has no variance to work with.
  Prediction 1 was confirmed in the pilot and is unchanged: the conjunctive members have
  marginal content 0.03 and 0.02 for their capacity and solo alignment 0.00, while contributing
  0.142 each, so the criterion as it stands excludes two members that the construction makes
  constitutive.  That result is what the rest of the experiment now has to answer to.
"""

from __future__ import annotations

import itertools
import json
import sys
from math import factorial

import numpy as np

QUICK = "--quick" in sys.argv
N_ITEMS = 600 if QUICK else 2000
N_SEEDS = 2 if QUICK else 5
EPS = 0.02
CONTENT_EPS = 0.08
ALIGN_EPS = 0.10
DELTA = 0.12
KAPPA = 1.45            # declared: raised after the pilot so the modulator clears the contribution threshold
N_STRATA = 4
SLOPE = 3.0
CAPS = ["A", "B", "X"]


def unit(centre, prec):
    return dict(c=np.array(centre, float), p=np.array(prec, float))


def build():
    a1 = unit((1.0, -1.2), (1.0, 0.15))
    a2 = unit((1.0, 1.2), (1.0, 0.15))
    b1 = unit((-1.2, 1.0), (0.15, 1.0))
    b2 = unit((1.2, 1.0), (0.15, 1.0))
    u1 = unit((2.2, 0.0), (0.45, 0.001))     # rises with x0, says nothing about x1
    u2 = unit((0.0, 2.2), (0.001, 0.45))     # rises with x1, says nothing about x0
    shared = unit((1.0, 1.0), (0.42, 0.42))
    gflat = unit((0.0, 0.0), (0.004, 0.004))
    inert = unit((-1.7, -1.7), (1.0, 1.0))
    units = [a1, a2, b1, b2, u1, u2, shared, gflat, inert]
    names = ["a1", "a2", "b1", "b2", "u1", "u2", "shared", "gflat", "inert",
             "modulator", "pure gain"]
    truth = {"A": {0, 1, 6}, "B": {2, 3, 6}, "X": {4, 5}}
    return units, names, truth


MOD, PURE = 9, 10
U1, U2 = 4, 5
W_A = [0, 1, 6, 7]
W_B = [2, 3, 6, 7]
THETA = np.array([1.75, 1.75, 0.0])   # declared: raised so the offset and the gain genuinely help
W_X = 9.0           # scale on the product term, so that X is a real capacity


class System:
    def __init__(self, seed):
        self.units, self.names, self.truth = build()
        rng = np.random.default_rng(26_000 + seed)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 2))
        self.A = np.zeros((N_ITEMS, len(self.units)))
        for k, u in enumerate(self.units):
            self.A[:, k] = np.exp(-np.sum(u["p"] * (self.X - u["c"]) ** 2, axis=1))
        self.c1 = self.A[:, U1] - self.A[:, U1].mean()
        self.c2 = self.A[:, U2] - self.A[:, U2].mean()
        self.n = len(self.units) + 2
        self.y = np.stack([self.X[:, 0] > 0, self.X[:, 1] > 0,
                           (self.X[:, 0] > 0) ^ (self.X[:, 1] > 0)], axis=1)
        self.cache = {}

    def drive(self, S, scale_mod=1.0, boost=None):
        S = set(S)
        A = self.A
        c1, c2 = self.c1, self.c2
        if boost:
            A = A.copy()
            for k, f in boost.items():
                A[:, k] = A[:, k] * f
            c1 = A[:, U1] - self.A[:, U1].mean()
            c2 = A[:, U2] - self.A[:, U2].mean()
        gain = 1.0 + KAPPA * scale_mod * self.A[:, 6] if MOD in S else np.ones(N_ITEMS)
        if PURE in S:                      # a pure gain: one number, the same for every case
            gain = gain * 1.45
        d = np.zeros((N_ITEMS, 3))
        for m, cols in enumerate([W_A, W_B]):
            live = [k for k in cols if k in S]
            if live:
                d[:, m] = A[:, live].sum(axis=1)
        # capacity X is carried by the product of its two components and nothing else
        if U1 in S and U2 in S:
            d[:, 2] = -W_X * c1 * c2
        return gain[:, None] * d

    def v(self, S):
        key = frozenset(S)
        if key not in self.cache:
            self.cache[key] = ((self.drive(S) > THETA) == self.y).mean(axis=0)
        return self.cache[key]

    def discriminability(self, S):
        """AUC of the capacity's application profile against its target, for coalition S."""
        key = ("auc", frozenset(S))
        if key not in self.cache:
            d = self.drive(S)
            self.cache[key] = np.array([auc(d[:, m], self.y[:, m]) for m in range(3)])
        return self.cache[key]

    def mu(self, S, **kw):
        z = SLOPE * (self.drive(S, **kw) - THETA)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

    def state(self, i):
        if i == MOD:
            return self.A[:, 6]            # the modulator's activation: the shared unit's
        if i == PURE:
            return np.zeros(N_ITEMS)       # a pure gain has no case-by-case state at all
        return self.A[:, i]


def auc(scores, labels):
    """Probability that a positive case is ranked above a negative one."""
    pos = labels.sum(); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def shapley(sysm, value):
    """Exact Shapley values for every capacity, for whichever value function is passed."""
    n = sysm.n
    phi = np.zeros((3, n))
    for i in range(n):
        others = [q for q in range(n) if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[:, i] += w * (value(set(S) | {i}) - value(S))
    return phi


def auc_gap(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) < 5 or len(neg) < 5 or np.allclose(scores, scores[0]):
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    a = float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return abs(a - 0.5)


def conditional_content(sysm, i, m, members):
    """|AUC - 0.5| of i's state against the target, within strata of the other members' state."""
    own = sysm.state(i)
    rest = [k for k in members if k != i and k != MOD]
    if not rest:
        return auc_gap(own, sysm.y[:, m])
    key = sysm.A[:, rest].sum(axis=1)
    edges = np.quantile(key, np.linspace(0, 1, N_STRATA + 1))
    vals, wts = [], []
    for a in range(N_STRATA):
        sel = (key >= edges[a]) & (key <= edges[a + 1] if a == N_STRATA - 1 else key < edges[a + 1])
        g = auc_gap(own[sel], sysm.y[sel, m])
        if g is not None:
            vals.append(g); wts.append(sel.sum())
    return float(np.average(vals, weights=wts)) if vals else None


def conditional_engagement(sysm, i, m):
    """Does perturbing i push the profile toward the truth, whether or not the rest is right?"""
    everyone = set(range(sysm.n))
    base = sysm.mu(everyone)
    if i in (MOD, PURE):
        pert = sysm.mu(everyone, scale_mod=1.0 + DELTA)
    else:
        pert = sysm.mu(everyone, boost={i: 1.0 + DELTA})
    d = pert[:, m] - base[:, m]
    signed = np.where(sysm.y[:, m], 1.0, -1.0)
    rest_pred = sysm.drive(everyone - {i})[:, m] > THETA[m]
    right = rest_pred == sysm.y[:, m]
    out = {}
    for tag, sel in [("rest_right", right), ("rest_wrong", ~right)]:
        if sel.sum() < 10 or np.std(d[sel]) < 1e-12:
            out[tag] = 0.0
        else:
            out[tag] = float(np.corrcoef(d[sel], signed[sel])[0, 1])
    out["size"] = float(np.mean(np.abs(d)))
    out["n_wrong"] = int((~right).sum())
    return out


def solo_alignment(sysm, i, m):
    v = sysm.mu({i})[:, m]
    if np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(v, sysm.y[:, m])[0, 1])


def run(seed):
    S = System(seed)
    phi = shapley(S, S.v)
    psi = shapley(S, S.discriminability)
    members = {cap: [i for i in range(S.n) if phi[m, i] >= EPS] for m, cap in enumerate(CAPS)}
    rows = []
    for i in range(S.n):
        row = dict(index=i, name=S.names[i], phi={}, psi={}, marginal={}, solo={})
        for m, cap in enumerate(CAPS):
            row["phi"][cap] = float(phi[m, i])
            row["psi"][cap] = float(psi[m, i])
            row["marginal"][cap] = auc_gap(S.state(i), S.y[:, m])
            row["solo"][cap] = solo_alignment(S, i, m)
        rows.append(row)
    return dict(seed=seed, names=S.names, truth={k: sorted(v) for k, v in S.truth.items()},
                members={k: sorted(v) for k, v in members.items()}, components=rows)


def main():
    out = [run(s) for s in range(N_SEEDS)]
    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps=EPS, content_eps=CONTENT_EPS,
                   align_eps=ALIGN_EPS, delta=DELTA, kappa=KAPPA, runs=out),
              open("results_exp26.json", "w"))
    names = out[0]["names"]

    def m_(i, field, cap, sub=None):
        vals = [r["components"][i][field][cap] for r in out]
        if sub is not None:
            vals = [v[sub] for v in vals]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("MARGINAL MEASURES  (the clauses as they currently stand)")
    print(f"  {'component':11s} {'phi A':>7s} {'phi B':>7s} {'phi X':>7s}   "
          f"{'marg. content A/B/X':>22s}   {'solo alignment A/B/X':>23s}")
    for i, nm in enumerate(names):
        p = " ".join(f"{m_(i, 'phi', c):7.3f}" for c in CAPS)
        mc = ", ".join(f"{m_(i, 'marginal', c):5.2f}" for c in CAPS)
        so = ", ".join(f"{m_(i, 'solo', c):+5.2f}" for c in CAPS)
        print(f"  {nm:11s} {p}   {mc:>22s}   {so:>23s}")

    print("\nDISCRIMINABILITY  (Shapley value on the capacity's AUC rather than its accuracy)")
    print(f"  {'component':11s} {'psi A':>7s} {'psi B':>7s} {'psi X':>7s}")
    for i, nm in enumerate(names):
        print(f"  {nm:11s} " + " ".join(f"{m_(i, 'psi', c):7.3f}" for c in CAPS))

    print("\nVERDICTS")
    print(f"  {'component':11s} {'contribution':>14s} {'+ content':>12s} {'+ solo profile':>16s} "
          f"{'+ discriminability':>20s}   {'built in as':>14s}")
    for i, nm in enumerate(names):
        c0 = [c for c in CAPS if m_(i, "phi", c) >= EPS]
        c1 = [c for c in c0 if m_(i, "marginal", c) >= CONTENT_EPS]
        c2 = [c for c in c1 if abs(m_(i, "solo", c)) >= ALIGN_EPS]
        c3 = [c for c in c0 if m_(i, "psi", c) >= EPS]
        built = [c for c in CAPS if i in out[0]["truth"][c]]
        f = lambda L: ", ".join(L) if L else "none"
        print(f"  {nm:11s} {f(c0):>14s} {f(c1):>12s} {f(c2):>16s} {f(c3):>20s}   {f(built):>14s}")


if __name__ == "__main__":
    main()
