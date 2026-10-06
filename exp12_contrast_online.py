"""
Experiment 12 -- Does contrast history persist in the vehicle when the contrast is absent?
========================================================================================

Reviewer round 3, question 2.  The direct test of Post 1's contrast claim
against its named rival, the GENERATIVE / INDEPENDENT-CATEGORY view.

DESIGN.  Two groups per replicate.  Identical positive stream (same draws, same
order, same trial schedule); only ONE contrast class differs:
    NEAR  right-hand contrast centred at hue +1.2
    FAR   right-hand contrast centred at hue +3.0
Positives at (0, 0); two input dimensions (hue, and a second dimension).  Three
further contrast classes are identical in both groups: left (-2.5, 0), up
(0, +2.5) and down (0, -2.5).
  [Changed after the pilot, declared: the pilot had only the right-hand
  contrast.  In the FAR group some jar learners then represented the positive
  class purely as "not the contrast" (a contrast jar plus a bias), leaving no
  positive-category representation to measure.  Surrounding contrasts make a
  positive-category jar the cheaper solution in both groups.  The ALCOVE
  learning rates were also lowered and attention bounded after it diverged in
  the pilot, and the covering map made finer (0.25 spacing) because at 0.5
  its peak could only fall on a grid node.]  After learning, learning is frozen and the
contrast is REMOVED from every test: each measure reads only the positive
category's own representation.

CONTRAST-FREE MEASURES
  centre         where the category's own representation peaks on the hue axis
                 (the model analogue of reproducing the typical member)
  typicality     own-representation strength at hue +0.5 minus at hue -0.5
                 (positive = the side toward where the contrast was)
  discrimination sensitivity of the own representation to a small hue change,
                 relative to the same change on the second dimension, averaged
                 over positive items (within-category discrimination)
plus, for comparison, one contrast-PRESENT measure
  boundary       where the full classifier switches on the hue axis

LEARNERS
  jar        Pool2 (local units, error-driven centre and precision, capacity
             cost, recruitment and pruning); the positive category's own
             representation is the positively-weighted part of its readout
  exemplar   ALCOVE-style: fixed covering map of exemplar nodes, learned
             dimensional attention, association weights; own representation
             is the positive output node
  generative each category is a Gaussian fitted to its OWN members only;
             classification compares the two densities at decision time; own
             representation is the positive density

PREDICTIONS, stated before running.
  1. generative: NO group difference on any contrast-free measure (by
     construction), but a group difference in the contrast-present boundary.
     Contrast is online at decision without being internal to the vehicle.
  2. jar: NEAR vs FAR differ on all three contrast-free measures -- centre
     displaced away from the contrast, typicality lower on the contrast side,
     hue discrimination sharpened.
  3. exemplar: typicality and discrimination differ (association weights,
     attention); centre shift smaller than the jar's or absent.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from pool import Pool2

QUICK = "--quick" in sys.argv
T = 4000 if QUICK else 8000
N_SEEDS = 3 if QUICK else 10
SIG = 0.40
CONTRAST = {"near": 1.2, "far": 3.0}
HGRID = np.arange(-2.0, 2.0 + 1e-9, 0.01)


FIXED = [(-2.5, 0.0), (0.0, 2.5), (0.0, -2.5)]


def streams(seed):
    """Positive draws, contrast noise and trial kinds -- shared by both groups.
    kind: 0 = positive (p = 0.5), 1 = right contrast, 2-4 = fixed contrasts."""
    rng = np.random.default_rng(12_000 + seed)
    u = rng.random(T)
    kind = np.where(u < 0.5, 0, np.where(u < 0.7, 1, 2 + rng.integers(0, 3, T)))
    pos = rng.normal(0, SIG, (T, 2))
    con_noise = rng.normal(0, SIG, (T, 2))
    return kind, pos, con_noise


def item(t, group, kind, pos, con_noise):
    k = kind[t]
    if k == 0:
        return pos[t], 1.0, 0
    centre = (CONTRAST[group], 0.0) if k == 1 else FIXED[k - 2]
    return con_noise[t] + np.array(centre), 0.0, int(k)


# ----------------------------------------------------------------- learners
class Jar:
    def __init__(self, seed):
        self.P = Pool2(2, 1, seed, mode="local")

    def train(self, s, y, k):
        self.P.step(s, np.array([y]), np.array([True]))

    def freeze(self):
        self.P.frozen = True

    def own(self, S):
        """Positive category's own representation: positively weighted units only."""
        out = []
        v = np.clip(self.P.V[0], 0, None)
        for s in S:
            out.append(float(v @ self.P.act(s)))
        return np.array(out)

    def decide(self, S):
        return np.array([self.P.predict(s)[0][0] for s in S])


class Exemplar:
    """ALCOVE (Kruschke 1992), city-block, covering map, humble teacher."""

    def __init__(self, seed, c=2.0, lw=0.005, la=0.0003, amax=4.0):
        h = np.arange(-3.0, 6.01, 0.25); d = np.arange(-3.0, 3.01, 0.25)
        self.H = np.array([(a, b) for a in h for b in d])
        self.alpha = np.array([0.5, 0.5]); self.W = np.zeros((2, len(self.H)))
        self.c, self.lw, self.la, self.amax = c, lw, la, amax

    def hid(self, s):
        return np.exp(-self.c * np.abs(self.H - s) @ self.alpha)

    def train(self, s, y, k):
        h = self.hid(s); o = self.W @ h
        tgt = np.array([y, 1 - y])
        teach = np.where(tgt > 0, np.maximum(1, o), np.minimum(-1, o))
        err = teach - o
        self.W += self.lw * np.outer(err, h)
        back = (err @ self.W) * h
        self.alpha -= self.la * (back[:, None] * self.c * np.abs(self.H - s)).sum(0)
        self.alpha = np.clip(self.alpha, 0, self.amax)

    def freeze(self):
        pass

    def own(self, S):
        return np.array([self.W[0] @ self.hid(s) for s in S])

    def decide(self, S):
        out = []
        for s in S:
            o = self.W @ self.hid(s); e = np.exp(2.0 * (o - o.max()))
            out.append(e[0] / e.sum())
        return np.array(out)


class Generative:
    """Each category -- the positive class and each contrast class -- is a
    Gaussian fitted to its own members only."""

    def __init__(self, seed):
        self.X = {}

    def train(self, s, y, k):
        self.X.setdefault(k, []).append(s)

    def freeze(self):
        self.mu = {k: np.mean(v, 0) for k, v in self.X.items()}
        self.var = {k: np.var(v, 0) + 1e-6 for k, v in self.X.items()}

    def dens(self, S, k):
        S = np.atleast_2d(S)
        return np.exp(-0.5 * np.sum((S - self.mu[k]) ** 2 / self.var[k], 1)) / np.sqrt(np.prod(2 * np.pi * self.var[k]))

    def own(self, S):
        return self.dens(S, 0)

    def decide(self, S):
        a = self.dens(S, 0) * 0.5
        b = sum(self.dens(S, k) * (0.2 if k == 1 else 0.1) for k in self.X if k != 0)
        return a / (a + b + 1e-300)


LEARNERS = {"jar": Jar, "exemplar": Exemplar, "generative": Generative}


def measures(L, probes):
    line = np.stack([HGRID, np.zeros_like(HGRID)], 1)
    own_line = L.own(line)
    centre = float(HGRID[int(np.argmax(own_line))])
    typ_pos, typ_neg = L.own(np.array([[0.5, 0.0], [-0.5, 0.0]]))
    scale = float(own_line.max()) + 1e-12
    typicality = float((typ_pos - typ_neg) / scale)
    eps = 0.05
    dh = np.abs(L.own(probes + [eps, 0]) - L.own(probes - [eps, 0]))
    dd = np.abs(L.own(probes + [0, eps]) - L.own(probes - [0, eps]))
    discrimination = float(np.mean(dh) / (np.mean(dd) + 1e-12))
    dec = L.decide(np.stack([np.arange(0, 3.0, 0.01), np.zeros(300)], 1))
    below = np.where(dec < 0.5)[0]
    boundary = float(np.arange(0, 3.0, 0.01)[below[0]]) if len(below) else float("nan")
    return dict(centre=centre, typicality=typicality, discrimination=discrimination, boundary=boundary)


def run(name, seed, group):
    kind, pos, con = streams(seed)
    L = LEARNERS[name](seed)
    for t in range(T):
        s, y, k = item(t, group, kind, pos, con)
        L.train(s, y, k)
    L.freeze()
    rng = np.random.default_rng(55_000 + seed)
    probes = rng.normal(0, SIG, (200, 2))
    # training-set accuracy with contrast present, for the record
    S = np.array([item(t, group, kind, pos, con)[0] for t in range(T - 1000, T)])
    Y = np.array([item(t, group, kind, pos, con)[1] for t in range(T - 1000, T)])
    acc = float(np.mean((L.decide(S) > 0.5) == (Y > 0.5)))
    return dict(learner=name, seed=seed, group=group, acc=acc, **measures(L, probes))


def main():
    out = []
    for name in LEARNERS:
        for seed in range(N_SEEDS):
            for g in ["near", "far"]:
                r = run(name, seed, g); out.append(r)
            a, b = out[-2], out[-1]
            print(f"  {name:10s} seed {seed}: centre {a['centre']:+.2f}/{b['centre']:+.2f}  "
                  f"typ {a['typicality']:+.3f}/{b['typicality']:+.3f}  disc {a['discrimination']:.2f}/{b['discrimination']:.2f}  "
                  f"boundary {a['boundary']:.2f}/{b['boundary']:.2f}  acc {a['acc']:.2f}/{b['acc']:.2f}", flush=True)
    json.dump(dict(T=T, n_seeds=N_SEEDS, contrast=CONTRAST, runs=out), open("results_exp12.json", "w"))
    print("\nSUMMARY  (NEAR minus FAR, mean +/- s.e. over seeds; contrast ABSENT except 'boundary')")
    for name in LEARNERS:
        row = []
        for m in ["centre", "typicality", "discrimination", "boundary"]:
            d = np.array([r[m] for r in out if r["learner"] == name and r["group"] == "near"]) - \
                np.array([r[m] for r in out if r["learner"] == name and r["group"] == "far"])
            row.append(f"{m} {d.mean():+.3f}+/-{d.std(ddof=1)/np.sqrt(len(d)):.3f}")
        print(f"  {name:10s} " + "  ".join(row))


if __name__ == "__main__":
    main()
