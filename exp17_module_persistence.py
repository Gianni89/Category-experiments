"""
Experiment 17 -- In exp13, what persists: one jar, or one capacity realised by successive jars?
=============================================================================================

Round 4, drift question.  Under graded modularity a jar is a community of
components whose mutual coupling exceeds their coupling to everything else.
This experiment applies that criterion to the redundant learner of exp13 and
tracks the community through complete component turnover.

COUPLING between two units i, j (for target m) is their Shapley INTERACTION
index (Grabisch & Roubens 1999): the average, over all coalitions of the other
units, of how much i's contribution to accuracy on m depends on j's presence.
Redundant units interact strongly (each matters only when the other is gone);
units serving unrelated jobs do not.  C_ij = sum over targets of |I_ij^m|.
Communities of the coupling graph are found by modularity maximisation
(greedy, Clauset-Newman-Moore).  The RIPE module at time t is the community
carrying the most RIPE Shapley value.

PROPOSED PERSISTENCE CRITERION (jar-level), tested here:
  M_{t+1} continues M_t iff
    (i)  overlap: some of M_{t+1}'s load is carried by members of M_t
         (no structural step replaces the whole module at once), and
    (ii) recruitment by the module's own deficit: members added between t and
         t+1 were recruited by errors on the module's job, and
    (iii) integration: they end up in the same coupling community.
  A jar persists through a chain of continuations.  A capacity can persist
  WITHOUT a jar persisting if the chain breaks while accuracy recovers.

CONDITIONS
  gradual     exp13's redundant learner (units fail on 25% of learning trials),
              30,000 drift trials after a 16,000-trial warm-up
  wholesale   the same, but at drift trial 10,000 every unit in the RIPE module
              is removed at once; learning continues

PREDICTIONS, stated before running.
  1. gradual: two communities (hue units, shape units) at every checkpoint;
     the RIPE module's membership turns over completely, but the chain of
     continuations is unbroken (overlap > 0 at every step, recruits triggered
     by RIPE errors and joining the RIPE community).
  2. wholesale: accuracy on RIPE collapses and then recovers through
     recruitment; the chain breaks at the removal (overlap 0).  Capacity
     persists; by the criterion, the jar does not.
"""

from __future__ import annotations

import itertools
import json
import sys
from math import factorial

import networkx as nx
import numpy as np

from pool import Pool2

QUICK = "--quick" in sys.argv
T0 = 8000 if QUICK else 16000
T_DRIFT = 16000 if QUICK else 30000
CHECK = 1000
WHOLESALE_AT = 6000 if QUICK else 10000
N_SEEDS = 2 if QUICK else 6
D, M, AMP, NOISE = 4, 2, 1.2, 0.45
N_TEST = 150


def item(rng):
    f = rng.integers(0, 2, 3)
    s = np.zeros(D)
    s[:3] = (2 * f - 1) * AMP + rng.normal(0, NOISE, 3)
    s[3] = rng.normal(0, 1)
    return s, np.array([float(f[0]), float(f[1])])


class Logged(Pool2):
    """Pool2 that records which target's errors triggered each recruitment."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trigger_log = []          # (t, slot, target)
        self._trigger = None

    def recruit_at(self, s):
        k = super().recruit_at(s)
        if k is not None:
            self.trigger_log.append((self.t, int(k), self._trigger))
        return k

    def step(self, s, tgt, on):
        # find which target (if any) would trigger recruitment this step, using
        # the same test Pool2.step applies, so the log names it
        self._trigger = None
        if self.do_recruit and self.t % 250 == 0:
            for m in range(self.M):
                eb = self.errbuf[m][-150:]
                if on[m] and len(eb) >= 100 and np.mean([w_ for w_, _ in eb]) > self.err_thresh:
                    self._trigger = m
                    break
        super().step(s, tgt, on)


def value_table(P, cases, units):
    n = len(units)
    rng = np.random.default_rng(len(units))
    subsets = None
    if n <= 9:
        subsets = [frozenset(c) for r in range(n + 1) for c in itertools.combinations(units, r)]
    table = {}

    def v(S):
        S = frozenset(S)
        if S not in table:
            a = np.zeros_like(P.alive); a[list(S)] = True
            table[S] = P.acc_by_target(cases, a)
        return table[S]

    if subsets is not None:
        for S in subsets:
            v(S)
    return v, n <= 9, rng


def shapley_and_interaction(P, cases, units):
    v, exact, rng = value_table(P, cases, units)
    n = len(units)
    phi = {i: np.zeros(M) for i in units}
    inter = {}
    if exact:
        for i in units:
            others = [u for u in units if u != i]
            for r in range(len(others) + 1):
                w = factorial(r) * factorial(n - r - 1) / factorial(n)
                for S in itertools.combinations(others, r):
                    phi[i] += w * (v(set(S) | {i}) - v(S))
        for i, j in itertools.combinations(units, 2):
            others = [u for u in units if u not in (i, j)]
            tot = np.zeros(M)
            for r in range(len(others) + 1):
                w = factorial(r) * factorial(n - r - 2) / factorial(n - 1)
                for S in itertools.combinations(others, r):
                    S = set(S)
                    tot += w * (v(S | {i, j}) - v(S | {i}) - v(S | {j}) + v(S))
            inter[(i, j)] = tot
    else:
        for _ in range(150):
            order = list(rng.permutation(units)); S = set(); prev = v(S)
            for u in order:
                S.add(u); cur = v(S); phi[u] += (cur - prev) / 150; prev = cur
        for i, j in itertools.combinations(units, 2):
            others = [u for u in units if u not in (i, j)]
            tot = np.zeros(M)
            for _ in range(40):
                S = {u for u in others if rng.random() < rng.random()}
                tot += (v(S | {i, j}) - v(S | {i}) - v(S | {j}) + v(S)) / 40
            inter[(i, j)] = tot
    return phi, inter


def modules(units, inter):
    G = nx.Graph(); G.add_nodes_from(units)
    for (i, j), I in inter.items():
        w = float(np.abs(I).sum())
        if w > 1e-4:
            G.add_edge(i, j, weight=w)
    if G.number_of_edges() == 0:
        return [{u} for u in units]
    return [set(c) for c in nx.algorithms.community.greedy_modularity_communities(G, weight="weight")]


def run(seed, wholesale):
    P = Logged(D, M, seed, mode="local", unit_dropout=0.25, prune_under_dropout=True)
    env = np.random.default_rng(2_000 + seed)
    on = np.array([True, True])
    for _ in range(T0):
        s, tg = item(env); P.step(s, tg, on)
    test_rng = np.random.default_rng(8_000 + seed)
    cases = [(s, tg, on) for s, tg in (item(test_rng) for _ in range(N_TEST))]

    uid = lambda k: (int(k), int(P.born[k]))
    trace = []
    prev_module = None
    first_module = None
    log_start = len(P.trigger_log)

    def checkpoint(t):
        nonlocal prev_module, first_module, log_start
        drop = P.unit_dropout; P.unit_dropout = 0.0; P.frozen = True
        units = [int(k) for k in np.where(P.alive)[0]]
        acc = P.acc_by_target(cases).tolist()
        phi, inter = shapley_and_interaction(P, cases, units) if len(units) > 1 else ({units[0]: np.array(acc)}, {})
        mods = modules(units, inter)
        ripe_mod = max(mods, key=lambda c: sum(phi[u][0] for u in c))
        mod_uids = {uid(u) for u in ripe_mod}
        ripe_load = {uid(u): float(phi[u][0]) for u in ripe_mod}
        total = sum(max(x, 0) for x in ripe_load.values()) + 1e-12
        if prev_module is None:
            overlap_load, jacc = 1.0, 1.0
            first_module = set(mod_uids)
        else:
            overlap_load = sum(max(ripe_load[u], 0) for u in mod_uids if u in prev_module) / total
            jacc = len(mod_uids & prev_module) / len(mod_uids | prev_module)
        from_start = sum(max(ripe_load[u], 0) for u in mod_uids if u in first_module) / total
        # recruits since last checkpoint: what triggered them, and did they join the RIPE module?
        new_logs = P.trigger_log[log_start:]; log_start = len(P.trigger_log)
        joined = [(trg, (slot, tt) in mod_uids) for (tt, slot, trg) in new_logs if P.alive[slot] and P.born[slot] == tt]
        wpairs = [float(np.abs(I).sum()) for (i, j), I in inter.items() if i in ripe_mod and j in ripe_mod]
        bpairs = [float(np.abs(I).sum()) for (i, j), I in inter.items() if (i in ripe_mod) != (j in ripe_mod)]
        within = float(np.mean(wpairs)) if wpairs else float("nan")
        between = float(np.mean(bpairs)) if bpairs else float("nan")
        trace.append(dict(t=t, acc=acc, n_units=len(units), n_modules=len(mods), module_size=len(ripe_mod),
                          overlap_load=float(overlap_load), jaccard=float(jacc), load_from_start=float(from_start),
                          recruits=[dict(trigger=trg, joined=j) for trg, j in joined],
                          within=within, between=between))
        prev_module = mod_uids
        P.unit_dropout = drop; P.frozen = False

    checkpoint(0)
    for t in range(1, T_DRIFT + 1):
        if wholesale and t == WHOLESALE_AT:
            # remove every unit of the current RIPE module at once
            units = [int(k) for k in np.where(P.alive)[0]]
            drop = P.unit_dropout; P.unit_dropout = 0.0; P.frozen = True
            phi, inter = shapley_and_interaction(P, cases, units)
            mods = modules(units, inter)
            ripe_mod = max(mods, key=lambda c: sum(phi[u][0] for u in c))
            P.unit_dropout = drop; P.frozen = False
            for k in ripe_mod:
                P.alive[k] = False; P.V[:, k] = 0.0
        s, tg = item(env); P.step(s, tg, on)
        if t % CHECK == 0:
            checkpoint(t)
    return dict(seed=seed, wholesale=wholesale, trace=trace)


def main():
    out = []
    for wholesale in [False, True]:
        for seed in range(N_SEEDS):
            r = run(seed, wholesale); out.append(r)
            tr = r["trace"]
            breaks = [x["t"] for x in tr[1:] if x["overlap_load"] < 1e-3]
            recs = [rc for x in tr for rc in x["recruits"]]
            print(f"  {'wholesale' if wholesale else 'gradual  '} seed {seed}: min RIPE acc {min(x['acc'][0] for x in tr):.2f} "
                  f"end {tr[-1]['acc'][0]:.2f}  modules {np.mean([x['n_modules'] for x in tr]):.1f}  "
                  f"RIPE-module size {np.mean([x['module_size'] for x in tr]):.1f}  load from start-members at end "
                  f"{tr[-1]['load_from_start']:.2f}  chain breaks at {breaks}  recruits {len(recs)} "
                  f"(RIPE-triggered {sum(rc['trigger'] == 0 for rc in recs)}, joined RIPE module "
                  f"{sum(rc['joined'] for rc in recs if rc['trigger'] == 0)})", flush=True)
    json.dump(dict(T0=T0, T_drift=T_DRIFT, check=CHECK, wholesale_at=WHOLESALE_AT, n_seeds=N_SEEDS, runs=out),
              open("results_exp17.json", "w"))
    print("\nSUMMARY")
    for wholesale in [False, True]:
        rs = [r for r in out if r["wholesale"] == wholesale]
        breaks = [sum(x["overlap_load"] < 1e-3 for x in r["trace"][1:]) for r in rs]
        recs = [rc for r in rs for x in r["trace"] for rc in x["recruits"] if rc["trigger"] == 0]
        wb = np.nanmean([x["within"] for r in rs for x in r["trace"]])
        bb = np.nanmean([x["between"] for r in rs for x in r["trace"]])
        print(f"  {'wholesale' if wholesale else 'gradual  '}: chain breaks per run {np.mean(breaks):.2f}; "
              f"load carried by start members at end {np.mean([r['trace'][-1]['load_from_start'] for r in rs]):.2f}; "
              f"RIPE-triggered recruits that joined the RIPE module {np.mean([rc['joined'] for rc in recs]) if recs else float('nan'):.0%} "
              f"(n={len(recs)}); coupling per pair within RIPE module {wb:.3f} vs between it and other units {bb:.3f}")


if __name__ == "__main__":
    main()
