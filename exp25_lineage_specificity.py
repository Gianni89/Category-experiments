"""
Experiment 25 -- Does the lineage repair behave gradually, and can relations carry it?
======================================================================================

TWO QUESTIONS LEFT OPEN BY EXPERIMENT 23
  exp23 showed that a surviving shared component can satisfy the persistence criterion as
  written, across a step at which every contributor specific to the capacity was destroyed.
  The proposed repair -- that what carries over must include a component SPECIFIC to the
  capacity -- broke the chain there and left ordinary turnover intact.  But it was tested only
  in its binary limiting case, and it has an unattractive consequence: a capacity whose
  organisation is entirely shared with another could never have a lineage at all.

  Part 1 asks whether a graded version behaves gradually.  The survivor is swept from equally
  shared between the two capacities to strongly specific to one of them, and the carried
  differential membership is reported continuously:

      d_A(r) = m_A(r) - m_B(r)

  with the carryover quantity being the share of the later organisation's positive d_A mass
  held by components that survived the transition.

  Part 2 asks the harder question.  The revision plan proposes that what must carry over is
  "the organisation that differentiates this capacity", where the differentiation may lie in
  membership, in resource state, or in RELATIONAL organisation.  The membership version cannot
  express the last of these.  So Part 2 builds the case that isolates it: two survivors whose
  membership is EXACTLY equally shared between the two capacities -- so that every
  membership-based differential is zero by construction -- but whose mutual relation differs
  completely between the capacities.  They cooperate for one (neither suffices alone) and
  substitute for the other (either suffices).  Their Shapley values are then equal in both
  capacities, their interaction magnitudes are equal, and the sign of the interaction is
  opposite: +V against -V.

  That is the sharpest possible form of the question, because every quantity computed from
  membership alone is identical across the two capacities and the only thing distinguishing
  them is the kind of organisation the two survivors are in -- which is what Gamma is for.

PREDICTIONS, STATED BEFORE RUNNING
  Part 1
   1. The criterion as written reports an unbroken chain at every point of the sweep, because
      the survivor always carries some load.  It is insensitive to specificity by construction.
   2. The binary specificity repair breaks the chain everywhere except at the extreme where the
      survivor has left the rival capacity's organisation altogether.  That is the all-or-nothing
      behaviour the revision plan objects to.
   3. The graded differential rises monotonically with the survivor's skew, so a threshold on
      carried differential mass gives a chain that breaks for equally shared survivors, holds
      for strongly skewed ones, and changes over a range rather than at a point.
  Part 2
   4. The two survivors have equal Shapley values in both capacities, so every membership-based
      differential is zero and both the binary and the graded membership repairs break the
      chain -- wrongly, since the organisations are plainly different.
   5. The signed interaction between them is strongly positive for the cooperative capacity and
      strongly negative for the substitutive one, so a relational differential -- the distance
      between the signed interaction profiles, not between their magnitudes -- is large.
      [Outcome: half right, and declared here rather than quietly rewritten.  The relational
      differential is large, as predicted, but it comes from strong substitution in B against
      near-independence in A, not from cooperation in A against substitution in B.  Once the
      later stage contains members that can carry A on their own, most coalitions have A
      already satisfied, so the survivors' cooperative interaction for A averages away.  Raising
      A's threshold to restore it costs the survivors their ability to carry A at the earlier
      stage, which destroys the case; that was tried and reverted rather than tuned around.]
   6. Therefore a criterion that admits relational differentiation holds the chain in Part 2
      while the membership-only versions break it, which is the result that decides whether the
      broader prose criterion is worth stating.

WHAT COUNTS AS FAILURE
  * If the graded differential does not rise monotonically with the skew in Part 1, the measure
    is not tracking specificity and should not be proposed.
  * If the two survivors in Part 2 turn out to have unequal membership across the capacities,
    the case is not isolating relational differentiation and says nothing about it.
  * If the relational differential is small in Part 2, relations cannot carry lineage in this
    architecture and the prose criterion should be narrowed back to membership.
"""

from __future__ import annotations

import itertools
import json
import sys
from math import factorial

import numpy as np

QUICK = "--quick" in sys.argv
N_ITEMS = 600 if QUICK else 1500
N_SEEDS = 2 if QUICK else 5
EPS = 0.02
TAU = 0.10           # how much carried differential counts as non-negligible
CAPS = ["A", "B"]


def gauss(X, centre, prec):
    return np.exp(-np.sum(np.array(prec) * (X - np.array(centre)) ** 2, axis=1))


class Stage:
    """A coalition game over the components present at one stage."""

    def __init__(self, units, W, theta, seed):
        rng = np.random.default_rng(25_000 + seed)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 2))
        self.y = np.stack([self.X[:, 0] > 0, self.X[:, 1] > 0], axis=1)
        self.A = np.stack([gauss(self.X, c, p) for c, p in units], axis=1)
        self.W, self.theta = W, np.array(theta, float)
        self.n = len(units)
        self.cache = {}

    def v(self, S):
        key = frozenset(S)
        if key not in self.cache:
            S = sorted(S)
            d = self.A[:, S] @ self.W[:, S].T if S else np.zeros((N_ITEMS, 2))
            self.cache[key] = ((d > self.theta) == self.y).mean(axis=0)
        return self.cache[key]


def attribution(st):
    n = st.n
    phi = np.zeros((2, n))
    for i in range(n):
        others = [q for q in range(n) if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[:, i] += w * (st.v(set(S) | {i}) - st.v(S))
    inter = np.zeros((2, n, n))
    for i, j in itertools.combinations(range(n), 2):
        others = [q for q in range(n) if q not in (i, j)]
        tot = np.zeros(2)
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 2) / factorial(n - 1)
            for S in itertools.combinations(others, r):
                S = set(S)
                tot += w * (st.v(S | {i, j}) - st.v(S | {i}) - st.v(S | {j}) + st.v(S))
        inter[:, i, j] = inter[:, j, i] = tot
    return phi, inter


# ------------------------------------------------------------------ part 1
def part1_stage(present, w_B, seed):
    """A is carried by whichever of the two triples is present, plus the shared survivor s.

    Components: a1 a2 a3 | a1' a2' a3' | s | b1 b2.  s feeds A at weight 1 and B at weight w_B.
    """
    layout = [((1.0, -1.3), (1.0, 0.15)), ((1.0, 0.0), (1.0, 0.15)), ((1.0, 1.3), (1.0, 0.15)),
              ((1.1, -1.5), (1.0, 0.15)), ((0.9, 0.1), (1.0, 0.15)), ((1.1, 1.5), (1.0, 0.15)),
              ((1.0, 1.0), (0.42, 0.42)),
              ((-1.3, 1.0), (0.15, 1.0)), ((1.3, 1.0), (0.15, 1.0))]
    names = ["a1", "a2", "a3", "a1'", "a2'", "a3'", "s", "b1", "b2"]
    idx = {n: k for k, n in enumerate(names)}
    keep = [idx[n] for n in present] + [idx["b1"], idx["b2"]]
    units = [layout[k] for k in keep]
    kept_names = [names[k] for k in keep]
    W = np.zeros((2, len(keep)))
    for a, nm in enumerate(kept_names):
        if nm.startswith("a"):
            W[0, a] = 1.0
        elif nm.startswith("b"):
            W[1, a] = 1.0
        elif nm == "s":
            W[0, a] = 1.0
            W[1, a] = w_B
    st = Stage(units, W, [1.30, 1.10], seed)
    phi, inter = attribution(st)
    return kept_names, phi, inter


def membership(names, phi):
    return {cap: {names[a]: float(phi[m, a]) for a in range(len(names))}
            for m, cap in enumerate(CAPS)}


def part1(w_B, seed):
    n0, p0, _ = part1_stage(["a1", "a2", "a3", "s"], w_B, seed)
    n1, p1, _ = part1_stage(["a1'", "a2'", "a3'", "s"], w_B, seed)
    m0, m1 = membership(n0, p0), membership(n1, p1)
    mem0 = {k for k, v in m0["A"].items() if v >= EPS}
    mem1 = {k for k, v in m1["A"].items() if v >= EPS}
    kept = mem0 & mem1

    d = {k: m1["A"][k] - m1["B"].get(k, 0.0) for k in mem1}
    total = sum(max(v, 0.0) for v in d.values())
    carried = sum(max(d[k], 0.0) for k in kept) / total if total > 0 else 0.0
    plain = sum(max(m1["A"][k], 0.0) for k in kept) / sum(max(m1["A"][k], 0.0) for k in mem1)
    specific = any(k not in {q for q, v in m1["B"].items() if v >= EPS} for k in kept)
    return dict(w_B=w_B, seed=seed, kept=sorted(kept), members=sorted(mem1),
                m_A_s=m1["A"].get("s", 0.0), m_B_s=m1["B"].get("s", 0.0),
                d_s=m1["A"].get("s", 0.0) - m1["B"].get("s", 0.0),
                plain_overlap=float(plain), carried_differential=float(carried),
                holds_plain=bool(plain > 1e-6), holds_binary=bool(specific),
                holds_graded=bool(carried >= TAU))


# ------------------------------------------------------------------ part 2
def part2_stage(present, seed):
    """Two survivors, equally shared, cooperating for A and substituting for B.

    A's threshold needs both; B's is crossed by either.  Components: s1 s2 | a1' a2' | b1.
    """
    layout = [((0.9, 1.0), (0.30, 0.30)), ((1.0, 0.9), (0.30, 0.30)),
              ((1.0, -1.2), (1.0, 0.18)), ((1.0, 1.2), (1.0, 0.18)),
              ((-1.2, 1.0), (0.18, 1.0))]
    names = ["s1", "s2", "a1'", "a2'", "b1"]
    idx = {n: k for k, n in enumerate(names)}
    keep = [idx[n] for n in present] + [idx["b1"]]
    units = [layout[k] for k in keep]
    kept_names = [names[k] for k in keep]
    W = np.zeros((2, len(keep)))
    for a, nm in enumerate(kept_names):
        if nm in ("s1", "s2"):
            W[0, a] = 1.0; W[1, a] = 1.0
        elif nm.startswith("a"):
            W[0, a] = 1.0
        else:
            W[1, a] = 1.0
    # A needs two contributors at once; B is satisfied by one
    st = Stage(units, W, [1.30, 0.80], seed)
    phi, inter = attribution(st)
    return kept_names, phi, inter


def part2(seed):
    n0, p0, i0 = part2_stage(["s1", "s2"], seed)
    n1, p1, i1 = part2_stage(["s1", "s2", "a1'", "a2'"], seed)
    m0, m1 = membership(n0, p0), membership(n1, p1)
    mem0 = {k for k, v in m0["A"].items() if v >= EPS}
    mem1 = {k for k, v in m1["A"].items() if v >= EPS}
    kept = mem0 & mem1
    pos = {n: a for a, n in enumerate(n1)}

    d = {k: m1["A"][k] - m1["B"].get(k, 0.0) for k in mem1}
    total = sum(max(v, 0.0) for v in d.values())
    carried_mem = (sum(max(d[k], 0.0) for k in kept) / total) if total > 0 else 0.0

    # relational differential: distance between the SIGNED interaction profiles
    def rel(pairs, I):
        return {(x, y): (float(I[0, pos[x], pos[y]]), float(I[1, pos[x], pos[y]])) for x, y in pairs}
    kept_pairs = list(itertools.combinations(sorted(kept), 2))
    all_pairs = list(itertools.combinations(sorted(mem1), 2))
    r_kept = rel(kept_pairs, i1); r_all = rel(all_pairs, i1)
    diff_kept = sum(abs(a - b) for a, b in r_kept.values())
    diff_all = sum(abs(a - b) for a, b in r_all.values())
    carried_rel = diff_kept / diff_all if diff_all > 1e-9 else 0.0

    return dict(seed=seed, members=sorted(mem1), kept=sorted(kept),
                m_A={k: round(m1["A"][k], 4) for k in sorted(mem1)},
                m_B={k: round(m1["B"].get(k, 0.0), 4) for k in sorted(mem1)},
                I_A_s1s2=float(i1[0, pos["s1"], pos["s2"]]),
                I_B_s1s2=float(i1[1, pos["s1"], pos["s2"]]),
                carried_membership_differential=float(carried_mem),
                carried_relational_differential=float(carried_rel),
                holds_plain=True, holds_graded_membership=bool(carried_mem >= TAU),
                holds_relational=bool(carried_rel >= TAU))


def main():
    sweep = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    p1 = [part1(w, s) for w in sweep for s in range(N_SEEDS)]
    p2 = [part2(s) for s in range(N_SEEDS)]
    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps=EPS, tau=TAU, sweep=sweep,
                   part1=p1, part2=p2), open("results_exp25.json", "w"))

    print("PART 1 -- sweeping the survivor from equally shared to capacity-specific")
    print(f"  {'w_B':>5s} {'m_A(s)':>7s} {'m_B(s)':>7s} {'d(s)':>7s} {'plain overlap':>14s} "
          f"{'carried differential':>21s}   {'as written':>11s} {'binary':>7s} {'graded':>7s}")
    for w in sweep:
        rs = [r for r in p1 if r["w_B"] == w]
        f = lambda k: np.mean([r[k] for r in rs])
        yn = lambda k: "holds" if np.mean([r[k] for r in rs]) > 0.5 else "BREAKS"
        print(f"  {w:5.1f} {f('m_A_s'):7.3f} {f('m_B_s'):7.3f} {f('d_s'):+7.3f} "
              f"{f('plain_overlap'):14.2f} {f('carried_differential'):21.2f}   "
              f"{yn('holds_plain'):>11s} {yn('holds_binary'):>7s} {yn('holds_graded'):>7s}")

    print("\nPART 2 -- equal membership, opposite relations")
    r = p2[0]
    print(f"  members of A at the later stage: {r['members']}; carried over: {r['kept']}")
    print(f"  Shapley value of each survivor:  A {r['m_A']}")
    print(f"                                   B {r['m_B']}")
    print(f"  interaction between the survivors:  A {np.mean([x['I_A_s1s2'] for x in p2]):+.3f}  "
          f"B {np.mean([x['I_B_s1s2'] for x in p2]):+.3f}")
    print(f"  carried differential, membership:   {np.mean([x['carried_membership_differential'] for x in p2]):.3f}")
    print(f"  carried differential, relational:   {np.mean([x['carried_relational_differential'] for x in p2]):.3f}")
    print(f"  chain holds -- as written: yes   graded membership: "
          f"{'yes' if np.mean([x['holds_graded_membership'] for x in p2]) > 0.5 else 'NO'}   "
          f"relational: {'yes' if np.mean([x['holds_relational'] for x in p2]) > 0.5 else 'NO'}")


if __name__ == "__main__":
    main()
