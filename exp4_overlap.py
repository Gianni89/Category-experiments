"""
Experiment 4 -- Overlapping jars.
=================================

The reviewer's "next genuinely hard test": can one input simultaneously recruit
several jars -- RED, ROUND, LARGE, ... -- with different histories, utilities,
edges and downstream roles, and do those jars stay stable and independently
revisable?  Or does the capacity mechanism force everything into one dominant
organisation?

THE WORLD.  Each episode has three independent binary features carried on three
channels (hue, shape, size) plus a pure-noise channel.  Channel value =
+/-1.2 + N(0, 0.45).  One of the eight feature combinations -- green,
elongated, large -- is NEVER shown during training.

THE TARGETS (each with its own action and its own feedback):
  T1 ripe      depends on hue            online from the start
  T2 rolls     depends on shape          online from 1/4 of the run
  T3 portion   depends on size           online from 1/2 of the run
  T4 edible    red AND round             online from 1/2 of the run
In the final quarter T1's boundary MOVES (ripe iff hue > 0.8, i.e. only the
reddest items) -- a revision to one jar's edge, not merely to a belief.

TWO LEARNERS, identical except for the competition regime:
  global   softmax over units: one input predominantly recruits one unit.
           This is the report's architecture.
  local    independent sigmoid units: an input can fully recruit several.

Everything else is shared: per-channel precision under a capacity cost
K = sum beta, error-driven centre and precision updates (one error signal
passed back along each unit's own outgoing weights -- ALCOVE-style; no
backpropagation through any settling loop), recruitment of a new unit when a
target's error stays high, and counterfactual pruning under lambda.

PREDICTIONS, stated before running.
  1. local: units become channel-selective (precision concentrates on one
     channel; the capacity cost strips the rest), a typical input recruits
     several units at once, and far fewer units are maintained than combinations.
     global: units become conjunctive cells, roughly one per trained combination.
  2. On the NEVER-SEEN combination, local is right on all three feature targets;
     global is wrong on about one in three, because it must file the novel case
     under the nearest trained conjunction.
  3. When T1's edge moves, local revises the hue jars and leaves the shape and
     size jars and their targets essentially untouched; global disturbs other
     targets, because its cells are shared across targets.
  4. The unit-by-target load-bearing matrix (ablate a unit with learning frozen,
     measure each target) is close to block-diagonal for local and dense for global.
"""

from __future__ import annotations

import json
import sys
import numpy as np

QUICK = "--quick" in sys.argv
T_TOTAL = 12000 if QUICK else 32000
N_SEEDS = 3 if QUICK else 8
Q = T_TOTAL // 4
ONSET = {0: 0, 1: Q, 2: 2 * Q, 3: 2 * Q}
SHIFT_AT = 3 * Q
TARGET_NAMES = ["ripe (hue)", "rolls (shape)", "portion (size)", "edible (red & round)"]
HELD_OUT = (0, 0, 1)            # green, elongated, large -- never trained
D, M, NMAX = 4, 4, 16
AMP, NOISE = 1.2, 0.45


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))


def softmax(v):
    v = v - v.max(); e = np.exp(v); return e / e.sum()


# ---------------------------------------------------------------------------
# world
# ---------------------------------------------------------------------------

def sample(rng, t, allow_held_out=False):
    while True:
        f = tuple(int(v) for v in rng.integers(0, 2, 3))
        if allow_held_out or f != HELD_OUT:
            break
    s = np.zeros(D)
    s[:3] = (2 * np.array(f) - 1) * AMP + rng.normal(0, NOISE, 3)
    s[3] = rng.normal(0, 1.0)
    return s, f


def targets(s, f, t):
    ripe = (s[0] > 0.8) if t >= SHIFT_AT else (f[0] == 1)
    return np.array([float(ripe), float(f[1]), float(f[2]), float(f[0] and f[1])])


def online(t):
    return np.array([t >= ONSET[m] for m in range(M)])


# ---------------------------------------------------------------------------
# learner
# ---------------------------------------------------------------------------

class Pool:
    def __init__(self, mode, seed, lam_b=0.004, lam_unit=0.01):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((NMAX, D))
        self.b = np.full((NMAX, D), np.log(0.8))
        self.alive = np.zeros(NMAX, bool)
        self.born = np.zeros(NMAX, int)
        self.V = np.zeros((M, NMAX))
        self.c = np.zeros(M)
        self.lam_b, self.lam_unit = lam_b, lam_unit
        self.eta_V, self.eta_w, self.eta_b, self.eta_att = 0.10, 0.03, 0.04, 0.004
        self.offset = 3.0            # sigmoid units respond ~0.95 at their centre
        self.gain = 1.0              # softmax temperature for the global mode
        self.b_min, self.b_max = -6.0, 3.0
        self.t = 0
        self.buf, self.errbuf = [], {m: [] for m in range(M)}
        self.frozen = False

    # -- fast dynamics ------------------------------------------------------
    def act(self, s, alive=None):
        alive = self.alive if alive is None else alive
        beta = np.exp(self.b)
        u = -np.sum(beta * (s[None, :] - self.w) ** 2, axis=1)
        if self.mode == "local":
            x = sigmoid(u + self.offset)
            return np.where(alive, x, 0.0), u
        x = softmax(np.where(alive, self.gain * u, -1e9))
        return np.where(alive, x, 0.0), u

    def predict(self, s, alive=None):
        x, _ = self.act(s, alive)
        return sigmoid(self.V @ x + self.c), x

    # -- slow dynamics ------------------------------------------------------
    def recruit(self, s):
        free = np.where(~self.alive)[0]
        if not len(free):
            return None
        k = int(free[0])
        self.alive[k] = True
        self.born[k] = self.t
        self.w[k] = s + self.rng.normal(0, 0.05, D)
        self.b[k] = np.log(0.8)
        self.V[:, k] = 0.0
        return k

    def step(self, s, f):
        t = self.t
        tgt = targets(s, f, t)
        on = online(t)
        p, x = self.predict(s)
        e = np.where(on, tgt - p, 0.0)
        wrong = (np.abs(tgt - p) > 0.5) & on
        for m in range(M):
            if on[m]:
                self.errbuf[m].append((float(wrong[m]), s))
                if len(self.errbuf[m]) > 300:
                    self.errbuf[m] = self.errbuf[m][-200:]

        if not self.frozen and self.alive.any():
            # readouts (delta rule, per target)
            self.V += self.eta_V * np.outer(e, x) * self.alive[None, :]
            self.c += self.eta_V * e
            # error passed back along each unit's own outgoing weights
            g = e @ self.V
            if self.mode == "local":
                delta = g * x * (1 - x)
            else:
                delta = x * (g - np.sum(x * g))
            beta = np.exp(self.b)
            diff = s[None, :] - self.w
            self.w += self.eta_w * (delta[:, None] * 2 * beta * diff) * self.alive[:, None]
            # a weak pull toward the cases a unit responds to keeps units on the data
            self.w += self.eta_att * (x[:, None] * diff) * self.alive[:, None]
            # precision: error-driven sharpening per channel, minus the capacity cost
            grad_b = delta[:, None] * (-beta * diff ** 2)
            self.b += self.eta_b * (grad_b - self.lam_b * beta) * self.alive[:, None]
            np.clip(self.b, self.b_min, self.b_max, out=self.b)

        # structure: recruit when a target's error persists, prune what does not pay
        if not self.frozen:
            if t % 250 == 0:
                for m in range(M):
                    eb = self.errbuf[m][-150:]
                    if on[m] and len(eb) >= 100 and np.mean([w_ for w_, _ in eb]) > 0.10:
                        bad = [s_ for w_, s_ in eb if w_ > 0]
                        if bad:
                            self.recruit(bad[int(self.rng.integers(len(bad)))])
                            break
            if not self.alive.any():
                self.recruit(s)
            self.buf.append((s, f, t))
            if len(self.buf) > 600:
                self.buf = self.buf[-400:]
            if t % 500 == 0 and t > 0:
                self.prune()
        self.t += 1

    def accuracy(self, cases, alive=None):
        ok, n = 0.0, 0
        for s, f, tt in cases:
            on = online(self.t)
            p, _ = self.predict(s, alive)
            tg = targets(s, f, self.t)
            ok += np.sum(((p > 0.5) == (tg > 0.5)) & on); n += on.sum()
        return ok / max(n, 1)

    def prune(self):
        cases = self.buf[-300:]
        base = self.accuracy(cases)
        for k in np.where(self.alive)[0]:
            if self.t - self.born[k] < 1500 or self.alive.sum() <= 1:
                continue
            trial = self.alive.copy(); trial[k] = False
            if base - self.accuracy(cases, trial) < self.lam_unit:
                self.alive[k] = False
                self.V[:, k] = 0.0
                base = self.accuracy(cases)


# ---------------------------------------------------------------------------
# measurement (all with learning frozen)
# ---------------------------------------------------------------------------

def probe_set(seed, n, held_out=False):
    rng = np.random.default_rng(77_000 + seed)
    out = []
    while len(out) < n:
        s, f = sample(rng, 0, allow_held_out=True)
        if (f == HELD_OUT) == held_out:
            out.append((s, f))
    return out


def measure(P, seed):
    was = P.frozen; P.frozen = True
    t = P.t
    on = online(t)
    trained = probe_set(seed, 400)
    novel = probe_set(seed, 200, held_out=True)

    def acc_per_target(cases, alive=None):
        A = np.zeros(M); n = 0
        for s, f in cases:
            p, _ = P.predict(s, alive)
            A += ((p > 0.5) == (targets(s, f, t) > 0.5)); n += 1
        return (A / n)

    acc_tr = acc_per_target(trained)
    acc_nv = acc_per_target(novel)
    X = np.array([P.predict(s)[1] for s, _ in trained])
    claimed = np.sum(X > 0.5, axis=1)
    coact = float(np.mean(claimed))
    multi = float(np.mean(claimed >= 2))
    triple = [i for i, (s_, f_) in enumerate(trained) if f_ == (1, 1, 1)]
    coact_111 = float(np.mean(claimed[triple])) if triple else None
    alive = np.where(P.alive)[0]

    # channel selectivity of each unit (share of precision on its top feature channel)
    beta = np.exp(P.b[alive][:, :3])
    sel = (beta.max(1) / beta.sum(1)).tolist() if len(alive) else []
    top_channel = beta.argmax(1).tolist() if len(alive) else []

    # unit x target load-bearing: ablate with learning frozen
    base = np.array([P.predict(s)[0] for s, _ in trained])
    LB = np.zeros((len(alive), M))
    for i, k in enumerate(alive):
        a2 = P.alive.copy(); a2[k] = False
        pk = np.array([P.predict(s, a2)[0] for s, _ in trained])
        LB[i] = np.mean(np.abs(pk - base), axis=0)
    LBo = LB[:, on] if on.any() else LB

    # participation: on how many jars does THIS input's classification depend?
    # (ablate each unit, learning frozen; count units whose removal moves any
    # online target's probability by more than 0.25 for this input)
    sub = trained[:150]
    base_sub = base[:150]
    moved = np.zeros((len(sub), len(alive)), bool)
    for i, k in enumerate(alive):
        a2 = P.alive.copy(); a2[k] = False
        pk = np.array([P.predict(s_, a2)[0] for s_, _ in sub])
        moved[:, i] = np.any((np.abs(pk - base_sub) > 0.25) & on[None, :], axis=1)
    participation = float(np.mean(moved.sum(1))) if len(alive) else 0.0
    focus = (LBo.max(1) / (LBo.sum(1) + 1e-12)).tolist() if len(alive) else []

    P.frozen = was
    return dict(t=t, acc_trained=acc_tr.tolist(), acc_novel=acc_nv.tolist(),
                coactivation=coact, multi_claimed=multi, coact_red_round_large=coact_111,
                participation=participation,
                n_units=int(len(alive)),
                selectivity=sel, top_channel=top_channel,
                LB=LB.tolist(), LB_focus=focus, online=on.tolist(),
                params={int(k): dict(w=P.w[k].tolist(), b=P.b[k].tolist()) for k in alive})


def run(mode, seed):
    if mode == "local-nocost":
        P = Pool("local", seed, lam_b=0.0)
    else:
        P = Pool(mode, seed)
    env = np.random.default_rng(1_000 + seed)
    curve = []
    snaps = {}
    for t in range(T_TOTAL):
        s, f = sample(env, t)
        P.step(s, f)
        if t % 400 == 0:
            recent = {m: np.mean([w_ for w_, _ in P.errbuf[m][-200:]]) if P.errbuf[m] else None
                      for m in range(M)}
            curve.append(dict(t=t, n_units=int(P.alive.sum()),
                              err={m: (None if v is None else float(v)) for m, v in recent.items()}))
        W = T_TOTAL - SHIFT_AT
        marks = {SHIFT_AT - W - 1: "baseline_start", SHIFT_AT - 1: "pre_shift", T_TOTAL - 1: "end"}
        if t + 1 in marks:
            snaps[marks[t + 1]] = measure(P, seed)
    return dict(curve=curve, snaps=snaps)


def drift(pre, end, units_by_channel):
    """Mean parameter change for units grouped by their dominant channel."""
    out = {}
    for ch, ks in units_by_channel.items():
        ds = []
        for k in ks:
            get = lambda snap: snap["params"].get(k, snap["params"].get(str(k)))
            a, b = get(pre), get(end)
            if a is None or b is None:
                continue
            ds.append(float(np.linalg.norm(np.array(a["w"]) - np.array(b["w"]))
                            + np.linalg.norm(np.array(a["b"]) - np.array(b["b"]))))
        out[ch] = float(np.mean(ds)) if ds else None
    return out


def main():
    results = {}
    for mode in ["local", "global", "local-nocost"]:
        runs = []
        for seed in range(N_SEEDS):
            r = run(mode, seed)
            pre, end = r["snaps"]["pre_shift"], r["snaps"]["end"]
            by_ch = {}
            for k, ch in zip(pre["params"].keys(), pre["top_channel"]):
                by_ch.setdefault(int(ch), []).append(k)
            base0 = r["snaps"]["baseline_start"]
            d_shift = drift(pre, end, by_ch)
            d_base = drift(base0, pre, by_ch)
            r["drift_by_channel"] = d_shift
            r["drift_baseline"] = d_base
            r["drift_excess"] = {k: (None if d_shift[k] is None or d_base.get(k) is None
                                     else d_shift[k] - d_base[k]) for k in d_shift}
            runs.append(r)
            print(f"  {mode} seed {seed}: units {pre['n_units']:2d} coact {pre['coactivation']:.2f} "
                  f"trained {np.round(pre['acc_trained'],2)} novel {np.round(pre['acc_novel'],2)} "
                  f"| after shift {np.round(end['acc_trained'],2)}", flush=True)
        results[mode] = runs

    with open("results_exp4.json", "w") as f:
        json.dump(dict(targets=TARGET_NAMES, T_total=T_TOTAL, onset=ONSET, shift_at=SHIFT_AT,
                       held_out=HELD_OUT, n_seeds=N_SEEDS, results=results), f)

    print("\nSUMMARY (mean over seeds, measured just before the edge shift)")
    for mode in ["local", "global", "local-nocost"]:
        pre = [r["snaps"]["pre_shift"] for r in results[mode]]
        end = [r["snaps"]["end"] for r in results[mode]]
        mean = lambda L, k: np.mean([x[k] for x in L], axis=0)
        print(f"\n  {mode}")
        print(f"    units maintained      {mean(pre,'n_units'):.1f}")
        print(f"    units per input       {mean(pre,'coactivation'):.2f}   "
              f"claimed by >=2 jars {mean(pre,'multi_claimed'):.2f}   "
              f"red-round-large recruits {np.mean([x['coact_red_round_large'] for x in pre]):.2f}")
        print(f"    channel selectivity   {np.mean([np.mean(x['selectivity']) for x in pre]):.2f}  (1/3 = none, 1 = one channel)")
        print(f"    LB focus on one target {np.mean([np.mean(x['LB_focus']) for x in pre]):.2f}")
        print(f"    acc trained           {np.round(mean(pre,'acc_trained'),3)}")
        print(f"    acc NEVER-SEEN combo  {np.round(mean(pre,'acc_novel'),3)}")
        print(f"    acc after T1 edge shift {np.round(mean(end,'acc_trained'),3)}")
        ex = [r["drift_excess"] for r in results[mode]]
        for ch in [0, 1, 2]:
            v = [e[ch] for e in ex if e.get(ch) is not None]
            if v:
                print(f"    excess drift after shift, units on channel {ch}: {np.mean(v):+.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
