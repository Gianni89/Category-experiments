"""
Experiment 20 -- Does the membership diagnostic survive contact with learned systems?
=====================================================================================

Experiment 19 validated the capacity-relative diagnostic against systems whose organisation
was known because it was built in.  Those systems are frozen, noiseless and hand-wired.  The
question here is the one exp19 could not ask: on a system that has actually LEARNED, where
there is no ground truth at all, is the inferred organisation stable?

This is NOT a recovery test.  Nothing here is scored against a known answer, because there
isn't one.  What is measured is whether the same answer keeps coming back when the things
that should not matter are varied.

THE SYSTEMS
  The four conditions of exp13, trained with the same world, schedule and seeds: reliable or
  redundant (unit dropout 0.25 with valuation under dropout), plastic or noisy (parameter
  noise after warm-up).  Two capacities share one pool: RIPE (hue) and ROLLS (shape).  After
  warm-up and drift, learning is frozen and the pool is never touched again.  The redundant
  conditions are the interesting ones: they carry 5-8 live units, several of which share each
  job, which is exactly the configuration where single-unit ablation was shown to mislead.

THE DIAGNOSTIC (unchanged from exp19, variant 2)
  v_m(S) is accuracy on capacity m with only the units in S alive, learning frozen and dropout
  off.  Exact Shapley values and exact Shapley interaction indices over all 2^n coalitions
  when n <= 12, sampled otherwise (declared in the output).  Membership of capacity m is
  {i : phi_i^m >= eps}, with eps = 0.02 as before.

THE FOUR STABILITY TESTS
  sample     the diagnostic is recomputed on 10 independent evaluation samples of 300 items
             drawn from the same world.  Reported: mean pairwise Jaccard of the membership
             sets, and how often a unit flips in or out.
  epsilon    eps is swept over 0.005 to 0.08; reported as Jaccard against the eps = 0.02 set,
             and the width of the plateau.
  attribution  permutation-sampled Shapley (400 permutations) and coalition-sampled
             interaction (200 per pair) against the exact values: correlation and the Jaccard
             of the membership each produces.
  seeds      membership cannot be compared unit-by-unit across seeds, because the units are
             different units.  What is compared is the shape of the answer: how many members
             per capacity, how much the two capacities' member sets overlap, and the unity
             profile (absolute and signed integration).

  A fifth check needs no ground truth and is worth having: v_m(members) against v_m(all).  If
  the members really do jointly sustain the capacity, dropping every non-member should cost
  almost nothing.

PREDICTIONS, STATED BEFORE RUNNING
  1. Cross-sample stability is high in the reliable conditions (mean Jaccard > 0.9) and lower
     but still usable in redundant + noisy (> 0.75), because that is where contributions are
     spread thinly over many substitutable units and small sample differences can push a
     marginal unit across eps.
  2. There is an eps plateau of at least a fourfold range over which membership does not
     change.  It will be narrower than the eightfold plateau of the constructed systems.
  3. Sampled attribution agrees with exact: correlation > 0.95 on phi, membership Jaccard
     > 0.9.  If this fails, the diagnostic is unusable at scale, since exact enumeration is
     only possible on toy pools.
  4. The redundant conditions show NEGATIVE signed integration for the hue capacity -- their
     unity is substitutive -- and the reliable conditions show positive or near-zero signed
     integration with far fewer members.  This matters for how exp13's drift result is
     described: if it holds, the module that drifts is held together by substitution.
  5. v_m(members) is within 0.01 of v_m(all) everywhere.

WHAT COUNTS AS FAILURE
  * mean cross-sample Jaccard below 0.7 in any condition;
  * no eps plateau of at least a fourfold range;
  * sampled-against-exact membership Jaccard below 0.9;
  * members failing to sustain the capacity (a gap above 0.02 against v_m(all)).
  Any of these would mean the diagnostic cannot carry the weight the post wants to put on it,
  and would need to be reported as such rather than patched.

DECLARED: the drift phase is 30,000 trials rather than exp13's 60,000, to keep 24 runs plus
the diagnostic inside a few minutes.  Warm-up, world, seeds and all learner parameters are
exp13's unchanged.  Nothing else was altered after the pilot.
"""

from __future__ import annotations

import itertools
import json
import sys
from math import factorial

import numpy as np

from exp13_drift import item, uids, D, M, SIG_PARAM
from pool import Pool2

QUICK = "--quick" in sys.argv
T0 = 6000 if QUICK else 16000
T_DRIFT = 6000 if QUICK else 30000
N_SEEDS = 2 if QUICK else 6
N_ITEMS = 300
N_SAMPLES = 4 if QUICK else 10
EPS = 0.02
EPS_SWEEP = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
EXACT_MAX = 12
CAPS = ["RIPE", "ROLLS"]


# ------------------------------------------------------------------ learned systems
def train(seed, redundant, noisy):
    P = Pool2(D, M, seed, mode="local", unit_dropout=0.25 if redundant else 0.0,
              prune_under_dropout=redundant)
    env = np.random.default_rng(2_000 + seed)
    noise_rng = np.random.default_rng(3_000 + seed)
    on = np.array([True, True])
    for _ in range(T0):
        s, tg = item(env); P.step(s, tg, on)
    for _ in range(T_DRIFT):
        s, tg = item(env); P.step(s, tg, on)
        if noisy:
            a = P.alive
            P.w[a] += noise_rng.normal(0, SIG_PARAM, P.w[a].shape)
            P.b[a] += noise_rng.normal(0, SIG_PARAM, P.b[a].shape)
            P.V[:, a] += noise_rng.normal(0, SIG_PARAM, P.V[:, a].shape)
    P.unit_dropout = 0.0
    P.frozen = True
    return P


def sample_cases(seed_offset, n=N_ITEMS):
    rng = np.random.default_rng(seed_offset)
    on = np.array([True, True])
    return [(s, tg, on) for s, tg in (item(rng) for _ in range(n))]


class Game:
    """v(S): accuracy on each capacity with only the units in S alive.

    In 'local' mode a unit's activation does not depend on which other units are alive, so the
    whole coalition game can be precomputed: activations once, then one matrix product per
    coalition.  Players are indexed 0..n-1 over the live units.
    """

    def __init__(self, P, cases):
        assert P.mode == "local"
        self.P = P
        self.units = [int(k) for k in np.where(P.alive)[0]]
        self.n = len(self.units)
        self.A = np.array([P.act(s) for s, _, _ in cases])[:, self.units]   # items x n
        self.V = P.V[:, self.units]                                        # M x n
        self.c = P.c
        self.y = np.array([tgt for _, tgt, _ in cases]) > 0.5              # items x M
        self.cache = {}

    def v(self, S):
        key = frozenset(S)
        hit = self.cache.get(key)
        if hit is None:
            S = sorted(S)
            drive = self.A[:, S] @ self.V[:, S].T if S else np.zeros((len(self.y), len(self.c)))
            pred = (drive + self.c) > 0.0          # sigmoid(z) > 0.5 iff z > 0
            hit = (pred == self.y).mean(axis=0)
            self.cache[key] = hit
        return hit


def exact_attribution(g):
    """Exact Shapley values and interaction indices for every capacity at once."""
    n, M = g.n, len(g.c)
    players = list(range(n))
    phi = np.zeros((M, n))
    for i in players:
        others = [q for q in players if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[:, i] += w * (g.v(set(S) | {i}) - g.v(S))
    inter = np.zeros((M, n, n))
    for i, j in itertools.combinations(players, 2):
        others = [q for q in players if q not in (i, j)]
        tot = np.zeros(M)
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 2) / factorial(n - 1)
            for S in itertools.combinations(others, r):
                S = set(S)
                tot += w * (g.v(S | {i, j}) - g.v(S | {i}) - g.v(S | {j}) + g.v(S))
        inter[:, i, j] = inter[:, j, i] = tot
    return phi, inter


def sampled_attribution(g, n_perm=400, n_coal=200, seed=0):
    rng = np.random.default_rng(seed)
    n, M = g.n, len(g.c)
    phi = np.zeros((M, n))
    for _ in range(n_perm):
        order = rng.permutation(n)
        S, prev = set(), g.v(set())
        for pos in order:
            S.add(int(pos))
            cur = g.v(S)
            phi[:, pos] += cur - prev
            prev = cur
    phi /= n_perm
    inter = np.zeros((M, n, n))
    for i, j in itertools.combinations(range(n), 2):
        others = [q for q in range(n) if q not in (i, j)]
        tot = np.zeros(M)
        for _ in range(n_coal):
            p = rng.random()
            S = {q for q in others if rng.random() < p}
            tot += g.v(S | {i, j}) - g.v(S | {i}) - g.v(S | {j}) + g.v(S)
        inter[:, i, j] = inter[:, j, i] = tot / n_coal
    return phi, inter


def membership(g, phi_row, eps=EPS):
    """Membership as a set of unit ids, so that it can be compared across samples."""
    return {g.units[a] for a in range(g.n) if phi_row[a] >= eps}


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def unity(g, phi_row, inter_m, m):
    mem_ids = sorted(membership(g, phi_row))
    pos = {u: a for a, u in enumerate(g.units)}
    mem = [pos[u] for u in mem_ids]
    others = [a for a in range(g.n) if a not in mem]
    within = [abs(inter_m[i, j]) for i, j in itertools.combinations(mem, 2)]
    signed = [inter_m[i, j] for i, j in itertools.combinations(mem, 2)]
    between = [abs(inter_m[i, j]) for i in mem for j in others]
    w = float(np.mean(within)) if within else 0.0
    b = float(np.mean(between)) if between else 0.0
    return dict(members=mem_ids, n_members=len(mem_ids),
                integration=w, integration_signed=float(np.mean(signed)) if signed else 0.0,
                leakage=b, contrast=None if (w + b) == 0 else w / (w + b),
                v_members=float(g.v(mem)[m]), v_all=float(g.v(range(g.n))[m]),
                v_empty=float(g.v([])[m]))


# ------------------------------------------------------------------ one learned system
def analyse(seed, redundant, noisy):
    P = train(seed, redundant, noisy)
    n_alive = int(P.alive.sum())
    exact_ok = n_alive <= EXACT_MAX

    ref = Game(P, sample_cases(80_000 + seed))
    attribute = exact_attribution if exact_ok else (lambda g: sampled_attribution(g, seed=11 + seed))
    phi, inter = attribute(ref)
    sphi, sinter = sampled_attribution(ref, seed=5 + seed)
    sample_games = [Game(P, sample_cases(90_000 + 1000 * k + seed)) for k in range(N_SAMPLES)]
    sample_phi = [attribute(g)[0] for g in sample_games]

    res = dict(seed=seed, redundant=redundant, noisy=noisy, n_alive=n_alive, exact=exact_ok,
               units=ref.units, capacities={})

    for mi, cap in enumerate(CAPS):
        ref_mem = membership(ref, phi[mi])

        # (a) independent evaluation samples
        sample_mems = [membership(g, p[mi]) for g, p in zip(sample_games, sample_phi)]
        pair_j = [jaccard(a, b) for a, b in itertools.combinations(sample_mems, 2)]
        in_count = {u: sum(u in s for s in sample_mems) for u in ref.units}

        # (b) the eps knob
        eps_j = {e: jaccard(membership(ref, phi[mi], e), ref_mem) for e in EPS_SWEEP}

        # (c) sampled against exact attribution
        iu = np.triu_indices(ref.n, 1)
        attr = dict(
            phi_corr=float(np.corrcoef(phi[mi], sphi[mi])[0, 1]) if ref.n > 1 else 1.0,
            inter_corr=float(np.corrcoef(inter[mi][iu], sinter[mi][iu])[0, 1]) if len(iu[0]) > 1 else 1.0,
            membership_jaccard=jaccard(membership(ref, sphi[mi]), ref_mem),
            phi_max_abs_diff=float(np.max(np.abs(phi[mi] - sphi[mi]))))

        res["capacities"][cap] = dict(
            phi=phi[mi].tolist(), interaction=inter[mi].tolist(),
            **unity(ref, phi[mi], inter[mi], mi),
            sample_jaccard_mean=float(np.mean(pair_j)) if pair_j else 1.0,
            sample_jaccard_min=float(np.min(pair_j)) if pair_j else 1.0,
            unstable_units=[u for u, c in in_count.items() if 0 < c < N_SAMPLES],
            eps_jaccard=eps_j, attribution=attr)
    return res


def main():
    out = []
    for redundant in [False, True]:
        for noisy in [False, True]:
            for seed in range(N_SEEDS):
                r = analyse(seed, redundant, noisy)
                out.append(r)
                tag = f"{'redundant' if redundant else 'reliable '} {'noisy  ' if noisy else 'plastic'}"
                bits = []
                for cap in CAPS:
                    c = r["capacities"][cap]
                    bits.append(f"{cap} {c['n_members']}/{r['n_alive']} J={c['sample_jaccard_mean']:.2f} "
                                f"I={c['integration']:+.3f}/{c['integration_signed']:+.3f}")
                print(f"  {tag} seed {seed}: " + "  ".join(bits), flush=True)

    json.dump(dict(t0=T0, t_drift=T_DRIFT, n_items=N_ITEMS, n_samples=N_SAMPLES, eps=EPS,
                   runs=out), open("results_exp20.json", "w"))

    def rows(red, noi):
        return [r for r in out if r["redundant"] == red and r["noisy"] == noi]

    print("\nSTABILITY BY CONDITION (mean over seeds, both capacities pooled)")
    print(f"  {'condition':22s} {'alive':>5s} {'members':>8s} {'sample J':>9s} {'worst J':>8s} "
          f"{'eps plateau':>12s} {'sampled J':>10s} {'phi r':>7s}")
    for red in [False, True]:
        for noi in [False, True]:
            rs = rows(red, noi)
            cs = [r["capacities"][c] for r in rs for c in CAPS]
            plateau = []
            for c in cs:
                ok = [float(e) for e, j in c["eps_jaccard"].items() if j == 1.0]
                plateau.append(max(ok) / min(ok) if ok else 1.0)
            name = f"{'redundant' if red else 'reliable'} + {'noisy' if noi else 'plastic'}"
            print(f"  {name:22s} {np.mean([r['n_alive'] for r in rs]):5.1f} "
                  f"{np.mean([c['n_members'] for c in cs]):8.1f} "
                  f"{np.mean([c['sample_jaccard_mean'] for c in cs]):9.2f} "
                  f"{np.min([c['sample_jaccard_min'] for c in cs]):8.2f} "
                  f"{np.mean(plateau):11.1f}x "
                  f"{np.mean([c['attribution']['membership_jaccard'] for c in cs]):10.2f} "
                  f"{np.mean([c['attribution']['phi_corr'] for c in cs]):7.3f}")

    print("\nUNITY PROFILE AND JOINT SUSTAINING")
    print(f"  {'condition':22s} {'cap':>6s} {'members':>8s} {'integ':>7s} {'signed':>7s} "
          f"{'leak':>6s} {'v(mem)':>7s} {'v(all)':>7s} {'gap':>6s}")
    for red in [False, True]:
        for noi in [False, True]:
            rs = rows(red, noi)
            for cap in CAPS:
                cs = [r["capacities"][cap] for r in rs]
                name = f"{'redundant' if red else 'reliable'} + {'noisy' if noi else 'plastic'}"
                print(f"  {name:22s} {cap:>6s} {np.mean([c['n_members'] for c in cs]):8.1f} "
                      f"{np.mean([c['integration'] for c in cs]):+7.3f} "
                      f"{np.mean([c['integration_signed'] for c in cs]):+7.3f} "
                      f"{np.mean([c['leakage'] for c in cs]):6.3f} "
                      f"{np.mean([c['v_members'] for c in cs]):7.3f} "
                      f"{np.mean([c['v_all'] for c in cs]):7.3f} "
                      f"{np.mean([c['v_all'] - c['v_members'] for c in cs]):+6.3f}")

    print("\nTHE EPS KNOB ON LEARNED SYSTEMS (Jaccard against the eps = 0.02 set)")
    print("    eps   " + "  ".join(f"{('redundant' if red else 'reliable')[:4]}+{('noisy' if noi else 'plas')}"
                                   for red in [False, True] for noi in [False, True]))
    for e in EPS_SWEEP:
        cells = []
        for red in [False, True]:
            for noi in [False, True]:
                cs = [r["capacities"][c] for r in rows(red, noi) for c in CAPS]
                cells.append(np.mean([c["eps_jaccard"][e] for c in cs]))
        print(f"  {e:6.3f}  " + "  ".join(f"{v:14.2f}" for v in cells))


if __name__ == "__main__":
    main()
