"""
Experiment 14 -- Can the jars dynamics ACQUIRE PET's role face?
==============================================================

Reviewer round 3, question 4.  In exp11 PET's role face (a map from a kept
animal's attributes to the consequences of keeping it) was fitted by ridge
regression with the right interaction term supplied by hand (hidden assumption
H4).  Here the role face is learned online from pet-keeping episodes by the
same kind of dynamics as the rest of the model: error-driven weights,
recruitment of new resources when error persists, counterfactual pruning under
a cost.  Nothing about the functional form is given, except the VOCABULARY of
resources that recruitment may draw on -- which is the manipulated variable.

WORLD.  Kinds as in exp11 (size, aquatic, danger, fur).  Episodes: a pet
(dog, cat, hamster, rabbit, goldfish, koi) is kept; the learner sees its jar
outputs (attributes + noise) and then the consequences:
    housing = size * (1 + 2 * aquatic),   risk = danger   (+ small noise)
No shark, pike, tuna or wolf is ever kept.

LEARNERS (the role face; all online, all with recruitment and pruning)
  local            recruits LOCAL units (Gaussian bumps in jar-output space,
                   centre and precision learned, capacity cost) -- the jar
                   units of demonstrations 1-4 -- with a linear readout
  linear+local     the unary jars' outputs also feed the readout directly;
                   recruitment adds local units
  linear+product   recruitment adds a PRODUCT of two jars' outputs (a gain /
                   relational resource of the v_a^T L v_b kind, restricted to
                   pairs), chosen by its correlation with the residual error
                   (cascade-correlation style), pruned if it stops paying
  mixed            recruitment chooses, on the same residual-correlation test,
                   between a new local unit and any product
  given-form       the exp11 reference: ridge regression with size*aquatic
                   supplied by hand

TESTS.  Consequences of keeping animals never kept:
  PET SHARK (true 2.70, 0.95), and a novel set: pike, tuna, wolf, fox.

PREDICTIONS, stated before running.
  1. local fails on PET SHARK housing: local units cannot extrapolate beyond
     the pets they were recruited on (prediction near the koi value or lower).
  2. linear+local improves risk (linear in DANGER) but not housing.
  3. linear+product recruits size x aquatic in most seeds and extrapolates
     housing well.
  4. mixed: open.  If it prefers local units, the resource vocabulary alone
     does not settle the matter; the selection signal must also favour
     products.
"""

from __future__ import annotations

import itertools
import json
import sys
import numpy as np

QUICK = "--quick" in sys.argv
T = 4000 if QUICK else 8000
N_SEEDS = 3 if QUICK else 10

KINDS = {
    "wolf": (0.70, 0, 0.90, 1), "fox": (0.40, 0, 0.60, 1), "cat": (0.30, 0, 0.40, 1),
    "dog": (0.50, 0, 0.30, 1), "rabbit": (0.15, 0, 0.00, 1), "hamster": (0.08, 0, 0.00, 1),
    "shark": (0.90, 1, 0.95, 0), "pike": (0.35, 1, 0.60, 0), "goldfish": (0.05, 1, 0.00, 0),
    "tuna": (0.60, 1, 0.20, 0), "koi": (0.30, 1, 0.00, 0),
}
PETS = ["dog", "cat", "hamster", "rabbit", "goldfish", "koi"]
NOVEL = ["pike", "tuna", "wolf", "fox"]
NAMES = ["size", "aquatic", "danger", "fur"]
PAIRS = list(itertools.combinations(range(4), 2))
NOISE, YNOISE = 0.04, 0.02


def consequences(v):
    return np.array([v[0] * (1 + 2 * v[1]), v[2]])


def observe(k, rng):
    return np.array(KINDS[k], float) + rng.normal(0, NOISE, 4)


class RoleFace:
    def __init__(self, seed, linear, local, product, lam_unit=0.0015, thr=0.035,
                 eta=0.05, eta_w=0.02, eta_b=0.02, lam_b=0.002, grace=800, nmax=12):
        self.rng = np.random.default_rng(seed)
        self.linear, self.local_ok, self.product_ok = linear, local, product
        self.W = np.zeros((2, 4)); self.c = np.zeros(2)
        self.units = []          # dicts: w, b, v, born
        self.prods = []          # dicts: pair, v, born
        self.lam_unit, self.thr, self.eta, self.eta_w, self.eta_b, self.lam_b = lam_unit, thr, eta, eta_w, eta_b, lam_b
        self.grace, self.nmax = grace, nmax
        self.buf, self.t = [], 0

    # ---- forward
    def feats(self, x, skip=None):
        out = self.c.copy()
        if self.linear:
            out = out + self.W @ x
        for i, u in enumerate(self.units):
            if skip == ("u", i):
                continue
            out = out + u["v"] * np.exp(-np.sum(np.exp(u["b"]) * (x - u["w"]) ** 2))
        for i, p in enumerate(self.prods):
            if skip == ("p", i):
                continue
            a, b = p["pair"]; out = out + p["v"] * x[a] * x[b]
        return out

    def predict(self, x):
        return self.feats(x)

    # ---- learning
    def step(self, x, y):
        e = y - self.feats(x)
        self.c += self.eta * e
        if self.linear:
            self.W += self.eta * np.outer(e, x)
        for u in self.units:
            beta = np.exp(u["b"]); d = x - u["w"]
            a = np.exp(-np.sum(beta * d ** 2))
            g = float(e @ u["v"]) * a
            u["v"] += self.eta * e * a
            u["w"] += self.eta_w * g * 2 * beta * d
            u["b"] += self.eta_b * (g * (-beta * d ** 2) - self.lam_b * beta)
            u["b"] = np.clip(u["b"], -6, 4)
        for p in self.prods:
            a, b = p["pair"]; p["v"] += self.eta * e * x[a] * x[b]
        self.buf.append((x, y)); self.buf = self.buf[-400:]
        self.t += 1
        if self.t % 200 == 0 and len(self.buf) >= 200:
            self.maybe_recruit()
        if self.t % 400 == 0:
            self.prune()

    def mse(self, skip=None):
        return float(np.mean([np.sum((y - self.feats(x, skip)) ** 2) for x, y in self.buf[-300:]]))

    def maybe_recruit(self):
        recent = self.buf[-200:]
        res = np.array([y - self.feats(x) for x, y in recent])
        if np.mean(np.abs(res)) < self.thr or len(self.units) + len(self.prods) >= self.nmax:
            return
        X = np.array([x for x, _ in recent])
        cands = []
        if self.product_ok:
            used = {p["pair"] for p in self.prods}
            for pr in PAIRS:
                if pr in used:
                    continue
                f = X[:, pr[0]] * X[:, pr[1]]
                cands.append((self.score(f, res), ("p", pr)))
        if self.local_ok:
            worst = X[int(np.argmax(np.sum(res ** 2, 1)))]
            f = np.exp(-np.sum(0.8 * (X - worst) ** 2, 1))
            cands.append((self.score(f, res), ("u", worst)))
        if not cands:
            return
        _, (kind, arg) = max(cands, key=lambda c: c[0])
        if kind == "p":
            self.prods.append(dict(pair=arg, v=np.zeros(2), born=self.t))
        else:
            self.units.append(dict(w=arg + self.rng.normal(0, 0.02, 4), b=np.full(4, np.log(0.8)),
                                   v=np.zeros(2), born=self.t))

    @staticmethod
    def score(f, res):
        f = f - f.mean()
        if np.std(f) < 1e-9:
            return 0.0
        return float(np.sum(np.abs(f @ (res - res.mean(0)))) / (np.linalg.norm(f) + 1e-12))

    def prune(self):
        base = self.mse()
        for kind, lst in [("u", self.units), ("p", self.prods)]:
            i = 0
            while i < len(lst):
                if self.t - lst[i]["born"] >= self.grace and self.mse((kind, i)) - base < self.lam_unit:
                    lst.pop(i); base = self.mse()
                else:
                    i += 1


def given_form(seed):
    rng = np.random.default_rng(100 + seed)
    rf = lambda v: np.array([v[0], v[1], v[2], v[3], v[0] * v[1], 1.0])
    X = np.array([rf(observe(k, rng)) for k in PETS for _ in range(30)])
    Y = np.array([consequences(np.array(KINDS[k], float)) for k in PETS for _ in range(30)])
    Wr = np.linalg.solve(X.T @ X + 1e-3 * np.eye(6), X.T @ Y)
    return lambda v: rf(v) @ Wr


CONFIGS = {
    "local": dict(linear=False, local=True, product=False),
    "linear+local": dict(linear=True, local=True, product=False),
    "linear+product": dict(linear=True, local=False, product=True),
    "mixed": dict(linear=True, local=True, product=True),
    # ADDED AFTER THE PILOT, declared: in the pilot the product learner recruited
    # size x aquatic first in every seed but went on to recruit spurious products
    # (danger x fur, ...) that fit the pets and wrecked risk extrapolation.  One
    # stricter setting of the recruitment threshold and unit cost was tried.
    "linear+product-strict": dict(linear=True, local=False, product=True, lam_unit=0.004, thr=0.05),
    "mixed-strict": dict(linear=True, local=True, product=True, lam_unit=0.004, thr=0.05),
}


def evaluate(pred, seed):
    rng = np.random.default_rng(900 + seed)
    res = {}
    for k in ["shark"] + NOVEL + PETS:
        res[k] = np.mean([pred(observe(k, rng)) for _ in range(40)], 0).tolist()
    return res


def run(name, seed):
    if name == "given-form":
        pred = given_form(seed); info = {}
    else:
        L = RoleFace(seed, **CONFIGS[name])
        rng = np.random.default_rng(1_400 + seed)
        for _ in range(T):
            k = PETS[int(rng.integers(len(PETS)))]
            x = observe(k, rng)
            y = consequences(np.array(KINDS[k], float)) + rng.normal(0, YNOISE, 2)
            L.step(x, y)
        pred = L.predict
        info = dict(n_units=len(L.units), products=[f"{NAMES[a]}x{NAMES[b]}" for a, b in (p["pair"] for p in L.prods)])
    ev = evaluate(pred, seed)
    true = {k: consequences(np.array(KINDS[k], float)).tolist() for k in ev}
    err_novel = float(np.mean([np.abs(np.array(ev[k]) - true[k]) for k in NOVEL]))
    err_pets = float(np.mean([np.abs(np.array(ev[k]) - true[k]) for k in PETS]))
    return dict(learner=name, seed=seed, pred=ev, true=true, err_novel=err_novel, err_pets=err_pets, **info)


def main():
    out = []
    for name in list(CONFIGS) + ["given-form"]:
        for seed in range(N_SEEDS):
            r = run(name, seed); out.append(r)
            print(f"  {name:15s} seed {seed}: SHARK {np.round(r['pred']['shark'],2)}  novel err {r['err_novel']:.3f}  "
                  f"pets err {r['err_pets']:.3f}  {r.get('products','')} units {r.get('n_units','')}", flush=True)
    json.dump(dict(T=T, n_seeds=N_SEEDS, runs=out), open("results_exp14.json", "w"))
    print("\nSUMMARY  PET SHARK true [2.70, 0.95]")
    for name in list(CONFIGS) + ["given-form"]:
        rs = [r for r in out if r["learner"] == name]
        sh = np.mean([r["pred"]["shark"] for r in rs], 0)
        prods = {}
        for r in rs:
            for p in r.get("products", []):
                prods[p] = prods.get(p, 0) + 1
        print(f"  {name:15s} SHARK {np.round(sh,2)}  novel-kind abs err {np.mean([r['err_novel'] for r in rs]):.3f}  "
              f"trained-pet abs err {np.mean([r['err_pets'] for r in rs]):.3f}  products recruited (seeds): {prods}")


if __name__ == "__main__":
    main()
