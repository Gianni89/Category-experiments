"""
Experiment 18 -- One resource with several uses, or linked resources?
=====================================================================

Round 4, précis question D: what modification to the toy model would most
clearly distinguish
    ONE RESOURCE   PET is one learned organisation that both classifies
                   (is this a pet?) and supplies the role / consequence
                   structure used by Under(PET, X)
    LINKED         PET-classifier and PET-role are separate learned structures
                   associated only through the shared label/concept
and is cross-role transfer a serious discriminator (question C)?

DESIGN.  The same inputs (outputs of unary jars SIZE, AQUATIC, DANGER, FUR),
the same two uses:
    classify   is this kind a pet?                  (all kinds)
    role       consequences of keeping it as a pet  (pets only)
Three architectures differing ONLY in how much learned internal structure the
two uses share (hidden units, error-driven, 12 per use):
    linked     separate pools for the two uses                (sharing 0)
    partial    6 shared units + 6 private units per use       (sharing 0.5)
    one        a single pool of 12 used by both               (sharing 1)

After joint training, three revisions, each starting from the trained state:
  R1  classification-only revision: two new non-pet kinds (small, dangerous,
      furless: scorpion; aquatic: piranha) are learned as NOT pets; the role
      use receives no training at all.
      Transfer = how much the ROLE use's outputs change.
  R2  role-only revision: a new consequence rule (large animals are riskier
      to handle: risk = danger + 0.3 size), learned from pet-keeping episodes;
      the classify use receives no training.
      Transfer = how much the CLASSIFY use's outputs change.
  U   upstream control: no learning at all; the DANGER jar feeding both uses
      is recalibrated (its output halved).  Both uses change in every
      architecture -- shared INPUTS produce "transfer" without shared vehicles.

PREDICTIONS, stated before running.
  1. R1 and R2 transfer is exactly zero for 'linked', intermediate for
     'partial', largest for 'one': transfer grades with sharing.
  2. U produces comparable changes in all three: transfer from shared inputs
     does not discriminate.
  3. Transfer in 'one' is mostly INTERFERENCE: the untrained use gets worse
     on what it had learned, not better.
"""

from __future__ import annotations

import json
import sys
import numpy as np

QUICK = "--quick" in sys.argv
N_SEEDS = 3 if QUICK else 10
T1 = 12000 if QUICK else 24000
T2 = 4000 if QUICK else 6000
H = 12

KINDS = {
    "wolf": (0.70, 0, 0.90, 1), "fox": (0.40, 0, 0.60, 1), "cat": (0.30, 0, 0.40, 1),
    "dog": (0.50, 0, 0.30, 1), "rabbit": (0.15, 0, 0.00, 1), "hamster": (0.08, 0, 0.00, 1),
    "mouse": (0.05, 0, 0.05, 1), "cow": (0.90, 0, 0.15, 1), "horse": (0.90, 0, 0.10, 1),
    "shark": (0.90, 1, 0.95, 0), "pike": (0.35, 1, 0.60, 0), "goldfish": (0.05, 1, 0.00, 0),
    "tuna": (0.60, 1, 0.20, 0), "koi": (0.30, 1, 0.00, 0),
}
NEW_NEG = {"scorpion": (0.03, 0, 0.90, 0), "piranha": (0.10, 1, 0.80, 0)}
PETS = ["dog", "cat", "hamster", "rabbit", "goldfish", "koi"]
NOISE = 0.04


def cons(v, rule=0):
    risk = v[2] + (0.3 * v[0] if rule == 1 else 0.0)
    return np.array([v[0] * (1 + 2 * v[1]), risk])


class Net:
    def __init__(self, seed, sharing):
        r = np.random.default_rng(seed)
        ns = int(round(H * sharing)); npv = H - ns
        self.ns, self.npv = ns, npv
        # pools: shared, private-classify, private-role
        self.P = {k: (r.normal(0, 0.8, (n, 4)), r.normal(0, 0.3, n)) for k, n in
                  [("s", ns), ("c", npv), ("r", npv)]}
        self.u = r.normal(0, 0.1, H); self.c0 = 0.0
        self.V = r.normal(0, 0.1, (2, H)); self.r0 = np.zeros(2)

    def hidden(self, x, use, mask=None):
        parts = []
        for k in ["s", use]:
            W, b = self.P[k]
            parts.append(np.tanh(W @ x + b))
        h = np.concatenate(parts)
        if mask is not None:
            h = h * mask
        return h

    def classify(self, x, mask=None):
        h = self.hidden(x, "c", mask)
        return 1 / (1 + np.exp(-(self.u @ h + self.c0)))

    def role(self, x, mask=None):
        return self.V @ self.hidden(x, "r", mask) + self.r0

    def _backhidden(self, x, use, gh, lr):
        i = 0
        for k in ["s", use]:
            W, b = self.P[k]
            n = len(b)
            if n:
                h = np.tanh(W @ x + b)
                g = gh[i:i + n] * (1 - h ** 2)
                W += lr * np.outer(g, x); b += lr * g
            i += n

    def step_classify(self, x, y, lr=0.05):
        h = self.hidden(x, "c"); p = 1 / (1 + np.exp(-(self.u @ h + self.c0)))
        e = y - p
        gh = e * self.u
        self.u += lr * e * h; self.c0 += lr * e
        self._backhidden(x, "c", gh, lr)

    def step_role(self, x, y, lr=0.05):
        h = self.hidden(x, "r"); e = y - (self.V @ h + self.r0)
        gh = e @ self.V
        self.V += lr * np.outer(e, h); self.r0 += lr * e
        self._backhidden(x, "r", gh, lr)


def obs(k, rng, table=None):
    t = table or {**KINDS, **NEW_NEG}
    return np.array(t[k], float) + rng.normal(0, NOISE, 4)


def snapshot(net, rng_seed, recal=False):
    rng = np.random.default_rng(rng_seed)
    out = {}
    for k in list(KINDS) + list(NEW_NEG):
        xs = [obs(k, rng) for _ in range(30)]
        if recal:
            xs = [x * np.array([1, 1, 0.5, 1]) for x in xs]
        out[k] = dict(p=float(np.mean([net.classify(x) for x in xs])),
                      role=np.mean([net.role(x) for x in xs], 0).tolist())
    return out


def coupling(net, seed):
    """How far the two uses draw on the same internal units: for each hidden
    unit, its ablation effect on each use (normalised), summed min()."""
    rng = np.random.default_rng(seed)
    xs = [obs(k, rng) for k in KINDS for _ in range(5)]
    ec, er = [], []
    for j in range(H):
        m = np.ones(H); m[j] = 0
        ec.append(np.mean([abs(net.classify(x) - net.classify(x, m)) for x in xs]))
        er.append(np.mean([np.abs(net.role(x) - net.role(x, m)).sum() for x in xs]))
    ec, er = np.array(ec), np.array(er)
    # unit j is at position j of the hidden vector for BOTH uses only if shared;
    # private units occupy the same slot index but are different units.
    ns = net.ns
    shared_part = np.minimum(ec[:ns] / (ec.sum() + 1e-12), er[:ns] / (er.sum() + 1e-12)).sum()
    return float(shared_part)


def run(seed, sharing):
    net = Net(seed, sharing)
    rng = np.random.default_rng(1_800 + seed)
    kinds = list(KINDS)
    for t in range(T1):
        if t % 2 == 0:
            k = kinds[int(rng.integers(len(kinds)))]
            net.step_classify(obs(k, rng), float(k in PETS))
        else:
            k = PETS[int(rng.integers(len(PETS)))]
            net.step_role(obs(k, rng), cons(np.array(KINDS[k], float)))
    before = snapshot(net, 7_000 + seed)
    coup = coupling(net, 7_500 + seed)

    import copy
    # R1: classification-only revision with two new non-pets
    n1 = copy.deepcopy(net); r1 = np.random.default_rng(2_800 + seed)
    kinds1 = kinds + list(NEW_NEG)
    for _ in range(T2):
        k = kinds1[int(r1.integers(len(kinds1)))]
        n1.step_classify(obs(k, r1), float(k in PETS))
    after1 = snapshot(n1, 7_000 + seed)
    # R2: role-only revision with a new consequence rule
    n2 = copy.deepcopy(net); r2 = np.random.default_rng(3_800 + seed)
    for _ in range(T2):
        k = PETS[int(r2.integers(len(PETS)))]
        n2.step_role(obs(k, r2), cons(np.array(KINDS[k], float), rule=1))
    after2 = snapshot(n2, 7_000 + seed)
    # U: upstream recalibration, no learning
    afterU = snapshot(net, 7_000 + seed, recal=True)

    ks = list(KINDS) + list(NEW_NEG)
    d_role = lambda a, b: float(np.mean([np.abs(np.array(a[k]["role"]) - b[k]["role"]).sum() for k in ks]))
    d_cls = lambda a, b: float(np.mean([abs(a[k]["p"] - b[k]["p"]) for k in ks]))
    # interference: change in role error on the TRAINED pets (old rule) after R1
    err = lambda s: float(np.mean([np.abs(np.array(s[k]["role"]) - cons(np.array(KINDS[k], float))).sum() for k in PETS]))
    cls_err = lambda s: float(np.mean([abs(s[k]["p"] - float(k in PETS)) for k in KINDS]))
    return dict(seed=seed, sharing=sharing, coupling=coup,
                R1_role_change=d_role(after1, before), R1_cls_change=d_cls(after1, before),
                R1_role_err_before=err(before), R1_role_err_after=err(after1),
                R2_cls_change=d_cls(after2, before), R2_role_change=d_role(after2, before),
                R2_cls_err_before=cls_err(before), R2_cls_err_after=cls_err(after2),
                U_role_change=d_role(afterU, before), U_cls_change=d_cls(afterU, before),
                new_neg_p_after_R1={k: after1[k]["p"] for k in NEW_NEG})


def main():
    out = []
    for sharing in [0.0, 0.5, 1.0]:
        for seed in range(N_SEEDS):
            r = run(seed, sharing); out.append(r)
            print(f"  sharing {sharing:.1f} seed {seed}: coupling {r['coupling']:.2f}  R1 role change {r['R1_role_change']:.3f} "
                  f"(role err {r['R1_role_err_before']:.3f}->{r['R1_role_err_after']:.3f})  R2 cls change {r['R2_cls_change']:.3f}  "
                  f"U role/cls {r['U_role_change']:.3f}/{r['U_cls_change']:.3f}", flush=True)
    json.dump(dict(n_seeds=N_SEEDS, T1=T1, T2=T2, runs=out), open("results_exp18.json", "w"))
    print("\nSUMMARY (mean over seeds)")
    for sharing in [0.0, 0.5, 1.0]:
        rs = [r for r in out if r["sharing"] == sharing]
        f = lambda k: np.mean([r[k] for r in rs])
        print(f"  sharing {sharing:.1f}: coupling {f('coupling'):.2f} | R1 (classify-only) -> role change {f('R1_role_change'):.3f}, "
              f"pet role error {f('R1_role_err_before'):.3f}->{f('R1_role_err_after'):.3f} | R2 (role-only) -> classify change "
              f"{f('R2_cls_change'):.3f}, classify error {f('R2_cls_err_before'):.3f}->{f('R2_cls_err_after'):.3f} | "
              f"upstream U: role {f('U_role_change'):.3f}, classify {f('U_cls_change'):.3f}")


if __name__ == "__main__":
    main()
