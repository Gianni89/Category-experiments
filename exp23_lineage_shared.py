"""
Experiment 23 -- Can a surviving shared component fake a lineage?
================================================================

THE OBJECTION
  The persistence criterion from exp17 counts M_t as continuing into M_{t+1} when (i) some of
  the load carries over, (ii) the new members were recruited by this capacity's own errors, and
  (iii) they integrate with the module.  Once organisations are allowed to overlap, "some
  member remains" may be far too weak.  Build a history in which EVERY category-specific
  contributor is destroyed at once while a shared component survives throughout, and the
  capacity is then rebuilt.  Does the criterion wrongly call that one continuous jar?

DECLARED: THIS IS A CONSTRUCTED HISTORY, NOT A LEARNED ONE
  The first version of this experiment tried to grow a shared component in a learned pool, by
  gating both targets on a shared context channel (RIPE = hue AND ctx, ROLLS = shape AND ctx).
  It does not work, and the reason is worth recording: a Gaussian unit with a linear readout
  absorbs the gate into each specialist -- one unit for hue-and-context, another for
  shape-and-context -- so the pool solves the gated world with no shared unit at all.  Nothing
  in that learner factors a common condition out into a component of its own.
  So the history is constructed here instead, in the manner of exp19: the stages are built so
  that what persists and what is destroyed is a fact about the construction.  This is the right
  move for the question, which is about what the criterion counts, not about what learning
  does.  Clause (ii) -- that the new members were recruited by this capacity's own deficit --
  is satisfied by stipulation at every stage, which isolates clause (i), the clause at issue.

THE THREE HISTORIES
  gradual        the three members of A's organisation are replaced one at a time, so that two
                 of the three originals are present at every step.  Ordinary turnover, and the
                 case the criterion exists to count as continuous.
  wholesale      all three are destroyed at once, along with the shared component, and three
                 new ones are recruited.  The control that shows the chain can break at all.
  spare-shared   all three are destroyed at once and three new ones are recruited, but the
                 shared component -- a member of both A's and B's organisations -- survives
                 throughout.  Every contributor specific to A is gone.  This is the
                 adversarial history.

THE TWO CRITERIA, SCORED SIDE BY SIDE AT EVERY CHECKPOINT
  as it stands   overlap of load between consecutive member sets > 0, recruits triggered by
                 RIPE's own errors, and new members coupled to the surviving ones.
  with the repair
                 the same, except that the overlap must include at least one member that is
                 SPECIFIC to RIPE -- one that is not also a member of ROLLS's organisation.
                 Continuity of the capacity-bearing organisation, rather than continuity of
                 some member.

  Membership is the capacity-relative diagnostic of exp19 and exp20, not the community
  partition exp17 used, since exp19 showed the partition cannot represent shared components --
  which is precisely what this experiment turns on.

PREDICTIONS, STATED BEFORE RUNNING
  1. gradual: both criteria report an unbroken chain.  If the repaired criterion breaks here it
     is too strong and should be withdrawn.
  2. wholesale: both criteria break the chain at the replacement.  Capacity persists, jar does
     not -- exp17's result, in a form where the construction fixes what persists.
  3. spare-shared: the criterion as it stands reports NO break, because the spared shared
     component keeps the load overlap above zero, while the repaired criterion breaks.  That
     difference is the finding: the criterion as written can be satisfied by a component that
     carries nothing specific to the capacity whose lineage is in question.
  4. The share of A's load carried by the spared component after the replacement is not a
     technicality: it should be a substantial fraction, which is worse for the criterion than a
     bare non-zero overlap would be, since no reasonable "more than a negligible amount"
     threshold would exclude it either.

WHAT COUNTS AS FAILURE
  * If spare-shared breaks the chain under the criterion as it stands, the objection does not
    bite in this model and the criterion needs no repair -- that result should be reported as
    plainly as the other.
  * If the repaired criterion also breaks the chain in the gradual condition, the repair is too
    strong: it would deny continuity to ordinary turnover, which is the thing the criterion
    exists to allow.
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
FLOOR = 0.01
CAPS = ["A", "B"]


def gauss(X, centre, prec):
    return np.exp(-np.sum(np.array(prec) * (X - np.array(centre)) ** 2, axis=1))


# Components available across the whole history.  A's organisation is {a1,a2,a3,s} at the
# start; B's is {b1,b2,s}.  The primed units are the replacements recruited later.
LAYOUT = [
    ("a1",  (1.0, -1.3), (1.0, 0.15)),
    ("a2",  (1.0,  0.0), (1.0, 0.15)),
    ("a3",  (1.0,  1.3), (1.0, 0.15)),
    ("a1'", (1.1, -1.5), (1.0, 0.15)),
    ("a2'", (0.9,  0.1), (1.0, 0.15)),
    ("a3'", (1.1,  1.5), (1.0, 0.15)),
    ("b1",  (-1.3, 1.0), (0.15, 1.0)),
    ("b2",  (1.3,  1.0), (0.15, 1.0)),
    ("s",   (1.0,  1.0), (0.42, 0.42)),
]
NAMES = [n for n, _, _ in LAYOUT]
IDX = {n: k for k, n in enumerate(NAMES)}
W_A = np.array([1, 1, 1, 1, 1, 1, 0, 0, 1], float)      # A's readout
W_B = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], float)      # B's readout
THETA = np.array([1.30, 1.10])

ORIGINAL = ["a1", "a2", "a3", "s"]
REPLACED = ["a1'", "a2'", "a3'", "s"]

HISTORIES = {
    # each stage lists which components are present
    "gradual": [ORIGINAL,
                ["a1'", "a2", "a3", "s"],
                ["a1'", "a2'", "a3", "s"],
                REPLACED],
    "wholesale": [ORIGINAL,
                  ["a1'", "a2'", "a3'"],          # the shared component goes too
                  ["a1'", "a2'", "a3'"]],
    "spare-shared": [ORIGINAL,
                     REPLACED,
                     REPLACED],
}


class Stage:
    """The coalition game over the components present at one stage of a history."""

    def __init__(self, present, seed):
        rng = np.random.default_rng(23_000 + seed)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 2))
        self.y = np.stack([self.X[:, 0] > 0, self.X[:, 1] > 0], axis=1)
        self.present = [n for n in present] + ["b1", "b2"]      # B's own units are always there
        self.players = [IDX[n] for n in self.present]
        A = np.stack([gauss(self.X, c, p) for _, c, p in LAYOUT], axis=1)
        self.A = A
        self.W = np.stack([W_A, W_B])
        self.cache = {}

    def v(self, S):
        key = frozenset(S)
        hit = self.cache.get(key)
        if hit is None:
            S = sorted(S)
            drive = self.A[:, S] @ self.W[:, S].T if S else np.zeros((N_ITEMS, 2))
            hit = ((drive > THETA) == self.y).mean(axis=0)
            self.cache[key] = hit
        return hit


def attribution(st):
    idx = st.players
    n = len(idx)
    phi = np.zeros((2, n))
    for a, i in enumerate(idx):
        others = [q for q in idx if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[:, a] += w * (st.v(set(S) | {i}) - st.v(S))
    inter = np.zeros((2, n, n))
    for a, b in itertools.combinations(range(n), 2):
        i, j = idx[a], idx[b]
        others = [q for q in idx if q not in (i, j)]
        tot = np.zeros(2)
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 2) / factorial(n - 1)
            for S in itertools.combinations(others, r):
                S = set(S)
                tot += w * (st.v(S | {i, j}) - st.v(S | {i}) - st.v(S | {j}) + st.v(S))
        inter[:, a, b] = inter[:, b, a] = tot
    return phi, inter


def snapshot(present, seed):
    st = Stage(present, seed)
    phi, inter = attribution(st)
    names = st.present
    mem = {cap: {names[a] for a in range(len(names)) if phi[m, a] >= EPS}
           for m, cap in enumerate(CAPS)}
    load = {cap: {names[a]: float(phi[m, a]) for a in range(len(names))}
            for m, cap in enumerate(CAPS)}
    coup = {frozenset((names[a], names[b])): float(abs(inter[0, a, b]))
            for a, b in itertools.combinations(range(len(names)), 2)}
    return dict(members=mem, load=load, coupling=coup, acc=st.v(st.players).tolist())


def continues(prev, cur, specific_only):
    """Does cur's A-organisation continue prev's?  Clause (ii) holds by stipulation."""
    kept = prev["members"]["A"] & cur["members"]["A"]
    if specific_only:
        kept = {u for u in kept if u not in cur["members"]["B"]}
    total = sum(max(cur["load"]["A"].get(u, 0.0), 0.0) for u in cur["members"]["A"])
    overlap = sum(max(cur["load"]["A"].get(u, 0.0), 0.0) for u in kept) / total if total > 0 else 0.0
    # clause (iii), read as exp17 read it: a new member must join the same organisation, which
    # it may do through other members rather than by a direct link to a surviving one.
    fresh = cur["members"]["A"] - prev["members"]["A"]
    reach, frontier = set(kept), list(kept)
    while frontier:
        u = frontier.pop()
        for q in cur["members"]["A"]:
            if q not in reach and cur["coupling"].get(frozenset((u, q)), 0.0) > FLOOR:
                reach.add(q); frontier.append(q)
    integrated = bool(kept) and fresh <= reach
    return dict(ok=bool(overlap > 1e-6 and integrated), overlap=float(overlap),
                kept=sorted(kept), integrated=bool(integrated))


def run(history, seed):
    stages = [snapshot(p, seed) for p in HISTORIES[history]]
    steps = []
    for k in range(1, len(stages)):
        steps.append(dict(step=k,
                          acc=stages[k]["acc"],
                          members=sorted(stages[k]["members"]["A"]),
                          plain=continues(stages[k - 1], stages[k], False),
                          repaired=continues(stages[k - 1], stages[k], True)))
    return dict(history=history, seed=seed,
                members0=sorted(stages[0]["members"]["A"]),
                acc0=stages[0]["acc"], steps=steps)


def main():
    out = [run(h, s) for h in HISTORIES for s in range(N_SEEDS)]
    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps=EPS, runs=out),
              open("results_exp23.json", "w"))

    print("STAGE BY STAGE (seed 0)")
    for h in HISTORIES:
        r = [x for x in out if x["history"] == h][0]
        print(f"\n  {h}:  stage 0 members {r['members0']}  A-accuracy {r['acc0'][0]:.3f}")
        for st in r["steps"]:
            print(f"    step {st['step']}: members {str(st['members']):34s} A-acc {st['acc'][0]:.3f}  "
                  f"as-is {'continues' if st['plain']['ok'] else 'BREAKS   '} "
                  f"(overlap {st['plain']['overlap']:.2f}, kept {st['plain']['kept']})  "
                  f"repaired {'continues' if st['repaired']['ok'] else 'BREAKS'} "
                  f"(overlap {st['repaired']['overlap']:.2f})")

    print("\nVERDICT")
    print(f"  {'history':14s} {'chain holds, as-is':>20s} {'chain holds, repaired':>23s} "
          f"{'load kept by the shared component':>34s}")
    for h in HISTORIES:
        rs = [x for x in out if x["history"] == h]
        plain_ok = all(st["plain"]["ok"] for r in rs for st in r["steps"])
        rep_ok = all(st["repaired"]["ok"] for r in rs for st in r["steps"])
        shared_load = np.mean([st["plain"]["overlap"] - st["repaired"]["overlap"]
                               for r in rs for st in r["steps"][:1]])
        print(f"  {h:14s} {'yes' if plain_ok else 'no':>20s} {'yes' if rep_ok else 'no':>23s} "
              f"{shared_load:34.2f}")


if __name__ == "__main__":
    main()
