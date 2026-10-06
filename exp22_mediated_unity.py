"""
Experiment 22 -- Is pairwise interaction the only kind of unity?
===============================================================

THE QUESTION
  In exp19 the distributed capacity was carried by six components whose pairwise interaction
  was 0.007, an order of magnitude below the cooperative cases.  The report called that one
  organisation with almost no unity.  The objection: those six components still converge on
  one classificatory output, and their organisation may lie in that common downstream
  mechanism rather than in pairwise synergy.  If so, the measure is capturing one kind of
  unity and calling its absence the absence of unity simpliciter.

  The test: build two systems with the same capacity, the same membership and the same
  near-zero pairwise interaction, differing only in whether the components are organised into
  a common classificatory mechanism or merely combined from outside.  If the diagnostic cannot
  tell them apart, it is missing a kind of unity.  If it can, the measure that tells them apart
  is the thing to add.

THE THREE SYSTEMS
  cooperative  three units, no one of which can cross the readout threshold alone, feeding one
               readout.  The ordinary case, included as the reference point for what strong
               direct integration looks like.
  mediated     six tiles with additive coverage -- each tile alone suffices for its own band --
               all feeding ONE readout, which is itself an ablatable component.  Without the
               readout there is no capacity at all, whatever the tiles do.
  external     the same six tiles, each with its OWN readout, each producing its own decision
               for its own band.  The capacity is what you get by taking whichever decision
               fires, and that combination happens OUTSIDE the system: there is no component
               in which the six come together.  Accuracy, membership and tile-to-tile
               interaction are built to match the mediated system as closely as possible.

  Making a readout an ablatable player is a modelling decision and is declared as such.  It is
  the only way for a common classificatory mechanism to be a thing the coalition game can see;
  in the learner proper the readout weights belong to the units.

THE MEASURES
  direct integration   mean |I_ij| over member pairs, as in exp19.
  hub score            the largest fraction of the other members with which any single member
                       interacts above the noise floor.  One component that every other member
                       needs is a common mechanism; this is what the star topology of a
                       mediated organisation looks like from inside the coalition game.
  member connectivity  whether the member set is connected as a graph through members only,
                       and into how many pieces it falls.
  mediated integration mean |I_ik| between members and the best hub, as against mean |I_ij|
                       between members themselves.

PREDICTIONS, STATED BEFORE RUNNING
  1. All three systems reach comparable accuracy, and in all three the membership recovered by
     positive contribution is exactly the set of components the construction uses.
  2. Direct integration is high in the cooperative system (about 0.09, as in exp19) and near
     zero among the tiles of both the mediated and the external system.  On the direct measure
     alone, mediated and external are indistinguishable.  That is the objection made good.
  3. The hub score separates them cleanly: about 1.0 for the mediated system, where every tile
     needs the one readout, and about 1/6 for the external system, where each tile needs only
     its own.
  4. The member set of the mediated system is connected; the external system's falls into six
     disconnected pairs, even though it computes the same function to the same accuracy.
  5. So the distributed case of exp19 should be re-described.  Its six tiles had no ablatable
     common mechanism to find because the readout was not a player; with the readout in the
     game, the answer should be that they are unified, but mediately rather than directly.

WHAT COUNTS AS FAILURE
  * If the hub score does not separate mediated from external by at least a factor of three,
    the coalition game cannot see common downstream organisation and the conceptual answer has
    to be given without a measure to back it.
  * If membership itself differs between the two systems, they are not matched and the
    comparison says nothing.
  * If the external system turns out to have a high hub score anyway, that would mean the
    measure detects something other than a common mechanism, and it should not be proposed.
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
FLOOR = 0.01          # an interaction counts as a link at the same scale as exp19's delta
SYSTEMS = ["cooperative", "mediated", "external"]


def gauss(X, centre, prec):
    return np.exp(-np.sum(np.array(prec) * (X - np.array(centre)) ** 2, axis=1))


class System:
    """v(S): accuracy on the one capacity, with only the components in S available."""

    def __init__(self, kind, seed):
        self.kind = kind
        rng = np.random.default_rng(22_000 + seed)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 2))
        self.y = self.X[:, 0] > 0
        if kind == "cooperative":
            self.A = np.stack([gauss(self.X, (1.0, y), (1.0, 0.15)) for y in (-1.3, 0.0, 1.3)], 1)
            self.names = ["u1", "u2", "u3"]
            self.readouts = []                       # the readout is not a player here
            self.theta = 1.30
        else:
            self.A = np.stack([gauss(self.X, (1.0, y), (1.0, 4.0))
                               for y in (-1.67, -1.0, -0.33, 0.33, 1.0, 1.67)], 1)
            self.theta = 0.90
            if kind == "mediated":
                self.names = [f"t{k+1}" for k in range(6)] + ["R"]
                self.readouts = [6]                  # one shared readout, ablatable
            else:
                self.names = [f"t{k+1}" for k in range(6)] + [f"R{k+1}" for k in range(6)]
                self.readouts = list(range(6, 12))   # one readout each, combined outside
        self.n = len(self.names)
        self.cache = {}

    def v(self, S):
        key = frozenset(S)
        if key in self.cache:
            return self.cache[key]
        S = set(S)
        tiles = sorted(i for i in S if i not in self.readouts)
        if self.kind == "external":
            pred = np.zeros(N_ITEMS, bool)
            for k in tiles:
                if (6 + k) in S:                     # this tile's own readout is present
                    pred |= (2.0 * self.A[:, k]) > self.theta
        else:
            has_readout = (not self.readouts) or bool(set(self.readouts) & S)
            if not has_readout or not tiles:
                pred = np.zeros(N_ITEMS, bool)
            else:
                w = 1.0 if self.kind == "cooperative" else 2.0
                pred = (w * self.A[:, tiles].sum(axis=1)) > self.theta
        out = float((pred == self.y).mean())
        self.cache[key] = out
        return out


def attribution(sysm):
    n = sysm.n
    players = list(range(n))
    phi = np.zeros(n)
    for i in players:
        others = [q for q in players if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[i] += w * (sysm.v(set(S) | {i}) - sysm.v(S))
    inter = np.zeros((n, n))
    for i, j in itertools.combinations(players, 2):
        others = [q for q in players if q not in (i, j)]
        tot = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 2) / factorial(n - 1)
            for S in itertools.combinations(others, r):
                S = set(S)
                tot += w * (sysm.v(S | {i, j}) - sysm.v(S | {i}) - sysm.v(S | {j}) + sysm.v(S))
        inter[i, j] = inter[j, i] = tot
    return phi, inter


def pieces(members, inter):
    """How many connected components the member set falls into, linking through members only."""
    members = list(members)
    seen, groups = set(), []
    for m in members:
        if m in seen:
            continue
        stack, grp = [m], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); grp.append(u)
            stack += [q for q in members if q not in seen and abs(inter[u, q]) > FLOOR]
        groups.append(sorted(grp))
    return groups


def analyse(kind, seed):
    S = System(kind, seed)
    phi, inter = attribution(S)
    members = [i for i in range(S.n) if phi[i] >= EPS]
    direct = [abs(inter[i, j]) for i, j in itertools.combinations(members, 2)]

    # the hub: the member that the most other members interact with
    hub, hub_score = None, 0.0
    for k in members:
        rest = [q for q in members if q != k]
        if not rest:
            continue
        share = np.mean([abs(inter[k, q]) > FLOOR for q in rest])
        if share > hub_score:
            hub, hub_score = k, float(share)
    non_hub = [i for i in members if i != hub]
    through_hub = [abs(inter[hub, q]) for q in non_hub] if hub is not None else []
    among_rest = [abs(inter[i, j]) for i, j in itertools.combinations(non_hub, 2)]

    grp = pieces(members, inter)
    return dict(kind=kind, seed=seed, n=S.n, names=S.names,
                members=[S.names[i] for i in members], n_members=len(members),
                shapley=phi.tolist(), interaction=inter.tolist(),
                accuracy=S.v(range(S.n)), chance=S.v([]),
                direct_integration=float(np.mean(direct)) if direct else 0.0,
                among_non_hub=float(np.mean(among_rest)) if among_rest else 0.0,
                hub=None if hub is None else S.names[hub], hub_score=hub_score,
                mediated_integration=float(np.mean(through_hub)) if through_hub else 0.0,
                n_pieces=len(grp), pieces=[[S.names[i] for i in g] for g in grp])


def main():
    out = [analyse(k, s) for k in SYSTEMS for s in range(N_SEEDS)]
    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps=EPS, floor=FLOOR, runs=out),
              open("results_exp22.json", "w"))

    print(f"  {'system':12s} {'acc':>6s} {'members':>8s} {'direct':>8s} {'non-hub':>8s} "
          f"{'hub':>6s} {'hub score':>10s} {'via hub':>8s} {'pieces':>7s}")
    for kind in SYSTEMS:
        rs = [r for r in out if r["kind"] == kind]
        print(f"  {kind:12s} {np.mean([r['accuracy'] for r in rs]):6.3f} "
              f"{np.mean([r['n_members'] for r in rs]):8.1f} "
              f"{np.mean([r['direct_integration'] for r in rs]):8.3f} "
              f"{np.mean([r['among_non_hub'] for r in rs]):8.3f} "
              f"{rs[0]['hub'] or '-':>6s} {np.mean([r['hub_score'] for r in rs]):10.2f} "
              f"{np.mean([r['mediated_integration'] for r in rs]):8.3f} "
              f"{np.mean([r['n_pieces'] for r in rs]):7.1f}")

    print("\n  direct    mean |I| over all member pairs")
    print("  non-hub   mean |I| over member pairs that exclude the hub -- the tiles' integration")
    print("            with one another, which is what exp19's measure was reporting")
    print("  hub score the largest share of other members any one member interacts with")
    print("  via hub   mean |I| between the hub and the other members")
    print("  pieces    how many disconnected parts the member set falls into\n")

    for kind in SYSTEMS:
        r = [x for x in out if x["kind"] == kind][0]
        print(f"  {kind:12s} members {r['members']}")
        print(f"  {'':12s} pieces  {r['pieces']}")


if __name__ == "__main__":
    main()
