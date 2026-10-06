"""
Experiment 19 -- Does the functional-community diagnostic recover organisation we built in?
===========================================================================================

THE CRITERION UNDER TEST
  "A collection of components is one classificatory organisation to the degree that its
   members jointly sustain the capacity and are more strongly functionally integrated with
   one another than with the surrounding machinery."

THE PROPOSED OPERATIONALISATION (variant 1, 'global partition')
  Freeze learning.  For capacity m let v_m(S) be accuracy on m when only the components in
  S are available.  The coalition-sensitive interaction of i and j is

      I_ij^(m) = E_S [ v_m(S + i + j) - v_m(S + i) - v_m(S + j) + v_m(S) ]

  taken with Shapley-interaction weights over S subset of N \\ {i, j} (Grabisch & Roubens
  1999).  Couple the components by C_ij = sum_m |I_ij^(m)|, then partition the resulting
  graph into communities.  A jar is a community.

A SECOND VARIANT IS TESTED ALONGSIDE IT (variant 2, 'capacity-relative')
  Membership is settled by contribution rather than by partition: component i belongs to the
  organisation of capacity m iff its Shapley value for m exceeds a threshold eps (the same
  order as the model's pruning threshold, 0.02 accuracy).  Unity is then a graded property
  of that set,

      U_m = mean within-set coupling on m / mean coupling from set members to non-members,

  and the sets for different capacities may overlap.

WHY A CONSTRUCTED SYSTEM RATHER THAN A LEARNED ONE
  The question is whether the diagnostic recovers organisation that is KNOWN.  In a learned
  learner the organisation is whatever learning produced, so there is no ground truth to
  recover.  Here every system is wired by hand: receptive fields, readout weights and
  thresholds are set so that the intended organisation is a fact about the construction.
  Components are Gaussian units over a 2-D input; each capacity is a thresholded readout
  over its own units; classes are balanced, so v(empty set) = 0.5 for every capacity.

THE CASES (ground truth in brackets)
  separate     two capacities, each carried by three units whose readout needs at least two
               of them at once                                   [{a1,a2,a3}, {b1,b2,b3}]
  shared-gain  the same, plus one global gain component: ablating it multiplies every unit's
               activation by 0.55, which breaks both capacities  [two groups + g in both]
  distributed  one capacity carried by six tiles with additive coverage -- each tile alone
               suffices for its own band of the input -- alongside an intact second capacity
                                                                 [{t1..t6}, {b1,b2,b3}]
  overlap      two capacities sharing one component s that is needed by both
                                                                 [{a1,a2,s}, {b1,b2,s}]
  redundant    the same two capacities, but one of A's units is duplicated, so either copy
               can be ablated almost for free                 [{a1,a2,a3,a3'}, {b1,b2,b3}]
  interaction  the same two capacities, plus a component k that suppresses two of A's units
               but feeds no readout: it interacts strongly with A and is no part of it
                                                                 [{a1,a2,a3}, {b1,b2,b3}]
  Every case also contains two inert units wired to nothing, which belong to no organisation.

PREDICTIONS, STATED BEFORE RUNNING
  1. separate: both variants recover the two groups exactly (ARI = 1, Jaccard = 1), and the
     inert units are left out.
  2. shared-gain: variant 1 fails, in one of two ways -- either the two groups fuse into one
     community (ARI < 0.5) or the gain component is assigned arbitrarily to one of them
     (ARI about 0.7-0.9, unstable across algorithms).  Variant 2 puts the gain component in
     both organisations, which is the right answer, so Jaccard = 1 for both capacities.
  3. distributed: variant 1 fragments the six tiles, because additive coverage makes their
     pairwise interactions near zero (ARI < 0.6).  Variant 2 recovers all six as members of
     one organisation and reports a LOW unity index for it, which is the graded answer.
  4. overlap: variant 1 cannot represent overlap at all and must put s on one side, so one
     capacity's membership is wrong (Jaccard <= 0.8).  Variant 2 recovers both.
  5. redundant: single ablation assigns each copy of the duplicated unit almost no importance,
     so a single-ablation diagnostic would drop both from the organisation; the Shapley value
     restores them and the pair's interaction is strongly negative (each matters only when the
     other is gone), so both variants keep the group together.
  6. interaction: variant 1 absorbs the suppressor into A's community, because |I| is large
     for a component that merely interferes.  Variant 2 excludes it, because its Shapley value
     for both capacities is negative.  This is the case that separates "functionally
     integrated with" from "part of", and it is the reason membership has to be defined by
     positive contribution rather than by coupling magnitude.

WHAT COUNTS AS FAILURE
  * variant 1 is counted as failing a case when ARI < 0.8 against the ground-truth partition,
    or when the three community algorithms disagree with each other (the structure is then
    not in the data, it is in the algorithm);
  * variant 2 is counted as failing when per-capacity Jaccard < 0.9;
  * either variant is counted as failing if the inert units are placed inside an organisation.
  A failure of variant 1 in cases 2-4 is a reason to change the criterion (membership by
  contribution, unity as a graded property, overlap allowed), not a reason to tune the
  simulation until the communities come out right.

COMMUNITY DETECTION IS AN IMPLEMENTATION CHOICE
  Three algorithms are run on the same coupling matrix: greedy modularity, Louvain, and
  connected components after thresholding.  The result reported for variant 1 is their
  agreement as well as their accuracy; the criterion is only as good as its worst reasonable
  implementation.  The edge threshold is swept rather than fixed, for the same reason.

DECLARED AFTER THE PILOT RUN (nothing above was changed)
  * The unity index was a ratio of within- to between-coupling.  In the distributed case both
    terms are near zero, so the ratio ran to 1e7 and reported near-perfect unity for a set of
    components with no mutual integration at all.  It is reported as two numbers instead: an
    absolute integration (mean within-set |I|, in accuracy units) and a bounded contrast
    within/(within + between) in [0, 1], left undefined when both fall below a noise floor of
    0.002.  This is a repair forced by the case, and a finding about the criterion rather than
    a convenience: the comparative clause alone can be satisfied vacuously.
  * The first clause of the criterion -- that the members JOINTLY SUSTAIN the capacity -- was
    not operationalised at all in the pilot.  It is now reported as v_m(members) against
    v_m(everything).
  * The redundant case was rebuilt twice.  Duplicating one unit of a threshold-readout trio
    does not produce redundancy: the copy adds drive, so ablating it costs real accuracy
    (0.10 in the pilot).  Under a threshold readout, a component is spare only if something
    else already crosses the threshold without it, so the case is now three bands with two
    interchangeable copies each, one copy being enough for its band.  Single ablation of any
    component should now cost nothing.  Predicted before running the rebuilt case: single
    ablation reports every component as idle, the Shapley value reports all six as
    contributors, interaction within a twin pair is strongly NEGATIVE and interaction across
    pairs is near zero, so variant 1 should split the one organisation into three pairs --
    the "genuine organisation incorrectly split" counterexample -- while variant 2 recovers
    all six.  The signed within-set interaction is reported alongside the absolute one,
    because a set held together by substitution is not held together in the same way as one
    held together by cooperation.
  * The 'interaction' case was added after the pilot, which tested only whether the diagnostic
    recovers organisation and not the other half of the question, whether mere interaction
    fools it.
"""

from __future__ import annotations

import itertools
import json
import sys
from math import factorial

import networkx as nx
import numpy as np

QUICK = "--quick" in sys.argv
N_ITEMS = 400 if QUICK else 1200
N_SEEDS = 2 if QUICK else 5
KAPPA = 6.0           # readout sharpness
EPS_MEMBER = 0.02     # Shapley threshold for membership, matching the pruning threshold
EDGE_EPS = 0.01       # couplings below this are treated as absent when building the graph
EDGE_SWEEP = [0.002, 0.005, 0.01, 0.02, 0.05]
NOISE_FLOOR = 0.002   # below this, a coupling is not distinguishable from nothing
DELTA_INTEG = 0.01    # an interaction counts as real at the same scale as the pruning threshold
GAIN_OFF = 0.55       # what ablating the shared gain component does to every activation
SUPPRESS = 0.50       # what the suppressor does to the units it acts on, when present


# ----------------------------------------------------------------- systems
def unit(centre, prec):
    return dict(c=np.array(centre, float), p=np.array(prec, float))


def build(case):
    """Return (units, readouts, truth, extra).

    readouts[m] = (weights over units, threshold, target function)
    truth[m]    = the set of component indices the construction puts in m's organisation
    extra       = None, or ('gain', index), or ('suppress', index, [units acted on])
    """
    A_units = [unit((1.0, -1.3), (1.0, 0.15)), unit((1.0, 0.0), (1.0, 0.15)), unit((1.0, 1.3), (1.0, 0.15))]
    B_units = [unit((-1.3, 1.0), (0.15, 1.0)), unit((0.0, 1.0), (0.15, 1.0)), unit((1.3, 1.0), (0.15, 1.0))]
    inert = [unit((-1.5, -1.5), (1.0, 1.0)), unit((1.5, -1.5), (1.0, 1.0))]
    hue = lambda X: (X[:, 0] > 0).astype(float)
    shape = lambda X: (X[:, 1] > 0).astype(float)

    if case == "separate":
        units = A_units + B_units + inert
        wA = np.array([1, 1, 1, 0, 0, 0, 0, 0], float)
        wB = np.array([0, 0, 0, 1, 1, 1, 0, 0], float)
        return units, {"A": (wA, 1.30, hue), "B": (wB, 1.30, shape)}, {"A": {0, 1, 2}, "B": {3, 4, 5}}, None

    if case == "shared-gain":
        units = A_units + B_units + inert
        wA = np.array([1, 1, 1, 0, 0, 0, 0, 0], float)
        wB = np.array([0, 0, 0, 1, 1, 1, 0, 0], float)
        # the gain component is index 8: a global multiplier on every activation
        return (units, {"A": (wA, 1.30, hue), "B": (wB, 1.30, shape)},
                {"A": {0, 1, 2, 8}, "B": {3, 4, 5, 8}}, ("gain", 8))

    if case == "distributed":
        tiles = [unit((1.0, y), (1.0, 4.0)) for y in (-1.5, -0.9, -0.3, 0.3, 0.9, 1.5)]
        units = tiles + B_units + inert
        wA = np.array([2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0], float)
        wB = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], float)
        return units, {"A": (wA, 0.90, hue), "B": (wB, 1.30, shape)}, {"A": set(range(6)), "B": {6, 7, 8}}, None

    if case == "overlap":
        shared = unit((1.0, 1.0), (0.30, 0.30))
        units = A_units[:2] + B_units[:2] + [shared] + inert
        wA = np.array([1, 1, 0, 0, 1, 0, 0], float)
        wB = np.array([0, 0, 1, 1, 1, 0, 0], float)
        return units, {"A": (wA, 1.10, hue), "B": (wB, 1.10, shape)}, {"A": {0, 1, 4}, "B": {2, 3, 4}}, None

    if case == "redundant":
        # A is carried by three bands, each with two identical copies, and one copy is enough
        # for its band.  Ablating any single component therefore costs nothing at all:
        # synchronic masking by construction.
        bands = [unit((1.0, y), (1.0, 0.6)) for y in (-1.4, -1.4, 0.0, 0.0, 1.4, 1.4)]
        units = bands + B_units + inert
        wA = np.array([2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0], float)
        wB = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], float)
        return (units, {"A": (wA, 0.90, hue), "B": (wB, 1.30, shape)},
                {"A": set(range(6)), "B": {6, 7, 8}}, None)

    if case == "interaction":
        # index 8 suppresses two of A's units when present, and feeds no readout at all
        units = A_units + B_units + inert
        wA = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0], float)
        wB = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], float)
        return (units, {"A": (wA, 1.30, hue), "B": (wB, 1.30, shape)},
                {"A": {0, 1, 2}, "B": {3, 4, 5}}, ("suppress", 8, [0, 1]))

    raise ValueError(case)


CASES = ["separate", "shared-gain", "distributed", "overlap", "redundant", "interaction"]


def activations(units, X):
    out = np.zeros((len(X), len(units)))
    for k, u in enumerate(units):
        d = X - u["c"]
        out[:, k] = np.exp(-np.sum(u["p"] * d ** 2, axis=1))
    return out


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


class System:
    """A hand-wired system, with v_m(S) as its only interface."""

    def __init__(self, case, seed):
        self.case = case
        self.units, self.readouts, self.truth, self.extra = build(case)
        rng = np.random.default_rng(19_000 + seed)
        self.X = rng.uniform(-2, 2, (N_ITEMS, 2))
        self.A = activations(self.units, self.X)
        self.n = len(self.units) + (0 if self.extra is None else 1)
        self.special = None if self.extra is None else self.extra[1]
        self.inert = [len(self.units) - 2, len(self.units) - 1]
        self.y = {m: f(self.X) for m, (_, _, f) in self.readouts.items()}

    def v(self, S, m):
        w, theta, _ = self.readouts[m]
        S = set(S)
        scale = np.ones(len(self.units))
        if self.extra is not None:
            present = self.special in S
            S = S - {self.special}
            if self.extra[0] == "gain" and not present:
                scale[:] = GAIN_OFF                 # ablating the gain weakens everything
            elif self.extra[0] == "suppress" and present:
                scale[self.extra[2]] = SUPPRESS     # the suppressor acts only while present
        S = sorted(S)
        drive = (self.A[:, S] * scale[S]) @ w[S] if S else np.zeros(len(self.X))
        pred = sigmoid(KAPPA * (drive - theta)) > 0.5
        return float(np.mean(pred == (self.y[m] > 0.5)))


# ------------------------------------------------- Shapley value and interaction (exact)
def shapley_and_interaction(sysm, m):
    n = sysm.n
    players = list(range(n))
    cache = {}

    def v(S):
        S = frozenset(S)
        if S not in cache:
            cache[S] = sysm.v(S, m)
        return cache[S]

    for r in range(n + 1):
        for S in itertools.combinations(players, r):
            v(S)

    phi = np.zeros(n)
    for i in players:
        others = [q for q in players if q != i]
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for S in itertools.combinations(others, r):
                phi[i] += w * (v(set(S) | {i}) - v(S))

    inter = np.zeros((n, n))
    for i, j in itertools.combinations(players, 2):
        others = [q for q in players if q not in (i, j)]
        tot = 0.0
        for r in range(len(others) + 1):
            w = factorial(r) * factorial(n - r - 2) / factorial(n - 1)
            for S in itertools.combinations(others, r):
                S = set(S)
                tot += w * (v(S | {i, j}) - v(S | {i}) - v(S | {j}) + v(S))
        inter[i, j] = inter[j, i] = tot

    single = np.array([v(tuple(players)) - v(tuple(p for p in players if p != i)) for i in players])
    return phi, inter, single


def sampled_interaction(sysm, m, n_samples, seed):
    """The same quantity estimated from random coalitions, to show it does not need
    exhaustive ablation."""
    rng = np.random.default_rng(seed)
    n = sysm.n
    inter = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        others = [q for q in range(n) if q not in (i, j)]
        tot = 0.0
        for _ in range(n_samples):
            p = rng.random()
            S = {q for q in others if rng.random() < p}
            tot += (sysm.v(S | {i, j}, m) - sysm.v(S | {i}, m) - sysm.v(S | {j}, m) + sysm.v(S, m))
        inter[i, j] = inter[j, i] = tot / n_samples
    return inter


# ----------------------------------------------------------------- partition metrics
def adjusted_rand(a, b):
    labels = sorted(set(a) | set(b))
    import collections
    pairs = collections.Counter(zip(a, b))
    n = len(a)
    comb2 = lambda x: x * (x - 1) / 2
    sum_ij = sum(comb2(c) for c in pairs.values())
    ai = collections.Counter(a); bj = collections.Counter(b)
    sum_a = sum(comb2(c) for c in ai.values())
    sum_b = sum(comb2(c) for c in bj.values())
    exp = sum_a * sum_b / comb2(n)
    mx = 0.5 * (sum_a + sum_b)
    return float((sum_ij - exp) / (mx - exp)) if mx != exp else 1.0


def labels_from_sets(sets, n):
    lab = [-1] * n
    for k, S in enumerate(sets):
        for i in S:
            lab[i] = k
    nxt = len(sets)
    for i in range(n):
        if lab[i] == -1:
            lab[i] = nxt; nxt += 1
    return lab


# ----------------------------------------------------------------- community algorithms
def communities(C, algorithm, edge_eps=EDGE_EPS):
    n = C.shape[0]
    G = nx.Graph(); G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] > edge_eps:
                G.add_edge(i, j, weight=float(C[i, j]))
    if G.number_of_edges() == 0:
        return [{i} for i in range(n)]
    if algorithm == "greedy":
        return [set(c) for c in nx.algorithms.community.greedy_modularity_communities(G, weight="weight")]
    if algorithm == "louvain":
        return [set(c) for c in nx.algorithms.community.louvain_communities(G, weight="weight", seed=0)]
    if algorithm == "threshold":
        H = nx.Graph(); H.add_nodes_from(range(n))
        thresh = 0.25 * max(d["weight"] for *_, d in G.edges(data=True))
        H.add_edges_from((i, j) for i, j, d in G.edges(data=True) if d["weight"] >= thresh)
        return [set(c) for c in nx.connected_components(H)]
    raise ValueError(algorithm)


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


# ----------------------------------------------------------------- one case
def run_case(case, seed):
    S = System(case, seed)
    caps = list(S.readouts)
    phi, inter, single = {}, {}, {}
    for m in caps:
        phi[m], inter[m], single[m] = shapley_and_interaction(S, m)

    C = sum(np.abs(inter[m]) for m in caps)                      # variant 1 coupling

    truth_sets = [set(S.truth[m]) for m in caps]
    disjoint_truth = all(not (a & b) for a, b in itertools.combinations(truth_sets, 2))
    truth_labels = labels_from_sets(truth_sets, S.n) if disjoint_truth else None

    def score(comm, eps):
        comm_sorted = sorted(comm, key=lambda c: -len(c))
        entry = dict(sizes=[len(c) for c in comm_sorted], communities=[sorted(c) for c in comm_sorted],
                     edge_eps=eps)
        if truth_labels is not None:
            entry["ari"] = adjusted_rand(truth_labels, labels_from_sets(comm_sorted, S.n))
        # how well each capacity's true set is matched by the single best community
        entry["best_jaccard"] = {m: max(jaccard(set(S.truth[m]), c) for c in comm) for m in caps}
        entry["inert_absorbed"] = any(len(c) > 1 and (set(S.inert) & c) for c in comm)
        if S.special is not None:
            entry["special_with"] = [sorted(c - {S.special}) for c in comm if S.special in c][0]
        return entry

    v1, sweep = {}, {}
    for alg in ["greedy", "louvain", "threshold"]:
        v1[alg] = score(communities(C, alg, EDGE_EPS), EDGE_EPS)
        sweep[alg] = [score(communities(C, alg, e), e) for e in EDGE_SWEEP]
    shapes = [sorted(map(sorted, v1[a]["communities"])) for a in v1]
    agree = all(s == shapes[0] for s in shapes)
    sweep_agree = all(sorted(map(sorted, e["communities"])) == shapes[0]
                      for alg in sweep for e in sweep[alg])

    v2 = {}
    for m in caps:
        member = {i for i in range(S.n) if phi[m][i] >= EPS_MEMBER}
        others = [i for i in range(S.n) if i not in member]
        Cm = np.abs(inter[m])
        within = [Cm[i, j] for i, j in itertools.combinations(sorted(member), 2)]
        signed = [inter[m][i, j] for i, j in itertools.combinations(sorted(member), 2)]
        between = [Cm[i, j] for i in member for j in others]
        w_bar = float(np.mean(within)) if within else 0.0
        b_bar = float(np.mean(between)) if between else 0.0
        both_quiet = max(w_bar, b_bar) < NOISE_FLOOR
        v2[m] = dict(members=sorted(member), truth=sorted(S.truth[m]),
                     jaccard=jaccard(member, set(S.truth[m])),
                     integration=w_bar, leakage=b_bar,
                     integration_signed=float(np.mean(signed)) if signed else 0.0,
                     contrast=None if both_quiet or (w_bar + b_bar) == 0 else w_bar / (w_bar + b_bar),
                     integrated=bool(w_bar >= DELTA_INTEG),
                     # first clause of the criterion: do the members jointly sustain m?
                     v_members=S.v(member, m), v_all=S.v(range(S.n), m), v_empty=S.v([], m))

    return dict(case=case, seed=seed, n=S.n, capacities=caps, inert=S.inert, special=S.special,
                coupling=C.tolist(), interaction={m: inter[m].tolist() for m in caps},
                shapley={m: phi[m].tolist() for m in caps}, single={m: single[m].tolist() for m in caps},
                truth={m: sorted(S.truth[m]) for m in caps}, disjoint_truth=disjoint_truth,
                variant1=v1, sweep=sweep, algorithms_agree=bool(agree), stable_under_sweep=bool(sweep_agree),
                variant2=v2)


def main():
    out = []
    for case in CASES:
        for seed in range(N_SEEDS):
            r = run_case(case, seed)
            out.append(r)
            if seed == 0:
                g = r["variant1"]["greedy"]
                print(f"  {case:12s}: variant 1 (greedy) {g['sizes']} "
                      f"ARI {g.get('ari', float('nan')):.2f} agree={r['algorithms_agree']} "
                      f"stable={r['stable_under_sweep']}  |  variant 2 " +
                      "; ".join(f"{m}: {r['variant2'][m]['members']} J={r['variant2'][m]['jaccard']:.2f}"
                                for m in r["capacities"]), flush=True)

    # does a sampled estimate reproduce the exact interactions?
    corr = {}
    for case in ["shared-gain", "distributed"]:
        S = System(case, 0)
        exact = shapley_and_interaction(S, "A")[1]
        samp = sampled_interaction(S, "A", 200, seed=1)
        iu = np.triu_indices(S.n, 1)
        corr[case] = float(np.corrcoef(exact[iu], samp[iu])[0, 1])

    json.dump(dict(n_items=N_ITEMS, n_seeds=N_SEEDS, eps_member=EPS_MEMBER, edge_eps=EDGE_EPS,
                   edge_sweep=EDGE_SWEEP, noise_floor=NOISE_FLOOR, sampling_correlation=corr,
                   runs=out), open("results_exp19.json", "w"))

    def mean(rs, f):
        vals = [f(r) for r in rs]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print("\nVARIANT 1 -- global partition of the coupling graph")
    print(f"  {'case':12s} {'greedy':>8s} {'louvain':>8s} {'thresh':>8s}   agree  stable  inert-in")
    for case in CASES:
        rs = [r for r in out if r["case"] == case]
        cells = []
        for alg in ["greedy", "louvain", "threshold"]:
            if rs[0]["disjoint_truth"]:
                cells.append(f"{mean(rs, lambda r: r['variant1'][alg]['ari']):8.2f}")
            else:
                cells.append(f"{mean(rs, lambda r: min(r['variant1'][alg]['best_jaccard'].values())):8.2f}")
        label = "ARI" if rs[0]["disjoint_truth"] else "J*"
        print(f"  {case:12s} " + " ".join(cells) +
              f"  {mean(rs, lambda r: r['algorithms_agree']):6.0%} "
              f"{mean(rs, lambda r: r['stable_under_sweep']):6.0%} "
              f"{mean(rs, lambda r: any(r['variant1'][a]['inert_absorbed'] for a in r['variant1'])):8.0%}"
              f"   ({label})")

    print("\nVARIANT 2 -- capacity-relative membership from Shapley values")
    print(f"  {'case':12s} {'cap':>3s}  {'Jaccard':>7s} {'integr.':>8s} {'signed':>7s} {'leak':>6s} "
          f"{'contrast':>8s} {'v(mem)':>7s} {'v(all)':>7s}")
    for case in CASES:
        rs = [r for r in out if r["case"] == case]
        for m in rs[0]["capacities"]:
            c = mean(rs, lambda r: r["variant2"][m]["contrast"])
            print(f"  {case:12s} {m:>3s}  {mean(rs, lambda r: r['variant2'][m]['jaccard']):7.2f} "
                  f"{mean(rs, lambda r: r['variant2'][m]['integration']):8.3f} "
                  f"{mean(rs, lambda r: r['variant2'][m]['integration_signed']):7.3f} "
                  f"{mean(rs, lambda r: r['variant2'][m]['leakage']):6.3f} "
                  f"{'   n/a' if np.isnan(c) else f'{c:8.2f}'} "
                  f"{mean(rs, lambda r: r['variant2'][m]['v_members']):7.3f} "
                  f"{mean(rs, lambda r: r['variant2'][m]['v_all']):7.3f}")

    print("\nWHERE THE EXTRA COMPONENT LANDS (variant 1)")
    for case in ["shared-gain", "interaction"]:
        for r in [x for x in out if x["case"] == case]:
            print(f"  {case:12s} seed {r['seed']}: " +
                  "  ".join(f"{a}->{r['variant1'][a]['special_with']}" for a in r["variant1"]))

    print("\nSINGLE ABLATION AGAINST SHAPLEY (capacity A, seed 0)")
    for case in CASES:
        r = [x for x in out if x["case"] == case and x["seed"] == 0][0]
        print(f"  {case:12s} single {np.round(r['single']['A'], 3)}\n"
              f"  {'':12s} shapley{np.round(r['shapley']['A'], 3)}")

    print("\nVARIANT 2 UNDER ITS OWN KNOB (mean Jaccard at each membership threshold)")
    print("    eps   " + "  ".join(f"{c[:11]:>11s}" for c in CASES))
    eps_sweep = {}
    for eps in [0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08]:
        row = []
        for case in CASES:
            js = []
            for r in [x for x in out if x["case"] == case]:
                for m in r["capacities"]:
                    mem = {i for i, p in enumerate(r["shapley"][m]) if p >= eps}
                    tr = set(r["truth"][m])
                    js.append(len(mem & tr) / len(mem | tr) if (mem | tr) else 1.0)
            row.append(float(np.mean(js)))
        eps_sweep[eps] = row
        print(f"  {eps:6.3f}  " + "  ".join(f"{v:11.2f}" for v in row))

    print("\n  sampled vs exact interaction (200 random coalitions per pair): " +
          ", ".join(f"{k} r={v:.3f}" for k, v in corr.items()))

    blob = json.load(open("results_exp19.json"))
    blob["eps_sweep"] = eps_sweep
    json.dump(blob, open("results_exp19.json", "w"))


if __name__ == "__main__":
    main()
