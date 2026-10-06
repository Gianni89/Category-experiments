"""
Experiment 15 -- Why can't the learner carve an edge inside an existing jar?
===========================================================================

Reviewer round 3, question 5.  In exp10 the own-edge compound CHOICE = very red
(hue > 1.4) AND round was never learned: accuracy stayed at the always-"no"
baseline.  Three candidate explanations:

  (a) a contingent flaw in recruitment or initialisation
  (b) a consequence of the unity assumption (a jar's shared parameters couple
      its edges, so a sub-boundary cannot be carved inside it)
  (c) evidence that subordinate / hierarchical jars need new machinery

CONDITIONS (all otherwise exp10's own-edge case, recruitment on)
  default     exp10 as run
  miss        recruitment triggered by the MISS rate on the compound's positive
              cases instead of the overall error rate.  (CHOICE is rare -- about
              8% of items -- so an overall error rate can never exceed the 10%
              trigger even when every positive is missed.)
  juvenile    a newly recruited unit is exempt from the pull toward the centre
              of its inputs for its first 1,500 trials
              [REDEFINED AFTER THE PILOT, declared: the pilot version also
              started new units narrow (precision 3.0 on every channel).  That
              made them hyper-specific on the irrelevant noise and size channels,
              so they almost never fired.  The narrow start was dropped.]
  miss+juv    both
  oracle      no discovery needed: at onset a unit is placed where CHOICE lives
              (hue 2.0, shape +1.2) with narrow hue tuning, then everything runs
              as default.  Tests MAINTENANCE: can the architecture hold such an
              edge once it exists?
  frequent    default learner, but CHOICE items are made common (30% of trials)
              -- the base-rate hypothesis from the world side
ADDED AFTER THE PILOT (declared), because the pilot showed that frequent
succeeded and the oracle failed, pointing at rarity rather than discovery:
  valued      miss trigger + juvenile, and a missed CHOICE costs 5x a false
              alarm in the learning signal (a rare distinction that matters a
              lot -- payoff asymmetry rather than frequency)
  oracle-kept oracle unit exempt from the centre pull and from pruning:
              separates capture and pruning from readout learning
ADDED AFTER THE FULL RUN (declared): oracle-noprune and oracle-anchored give
the oracle unit only one of the two protections each.  Run with --only.

READING THE OUTCOMES (stated before running)
  * oracle succeeds and default fails  -> the edge is representable and
    maintainable; the failure is in DISCOVERY -- (a), not (b).
  * oracle fails too                    -> the architecture cannot hold a
    sub-boundary inside a jar's territory -- (b) or (c).
  * miss or frequent succeeds           -> the specific flaw is the error-rate
    trigger's blindness to rare targets.
  * only miss+juv succeeds              -> both trigger and capture matter.
PREDICTION: oracle succeeds; miss+juv succeeds in most seeds; miss alone and
juvenile alone partly; default fails.
"""

from __future__ import annotations

import json
import sys
import numpy as np

from pool import Pool2

QUICK = "--quick" in sys.argv
T = 12000 if QUICK else 24000
N_SEEDS = 3 if QUICK else 8
D, AMP, NOISE = 4, 1.2, 0.45
ONSET = T // 2
M = 4


class Pool3(Pool2):
    def __init__(self, *a, trigger="rate", juvenile=False, pos_weight=1.0, **k):
        super().__init__(*a, **k)
        self.trigger, self.juvenile, self.pos_weight = trigger, juvenile, pos_weight
        self.mbuf = {m: [] for m in range(self.M)}
        self.protected = set()          # exempt from pruning
        self.anchored = set()           # exempt from the centre pull

    def prune(self):
        cases = self.buf[-300:]
        base = self.accuracy(cases, self.alive)
        for k in np.where(self.alive)[0]:
            if k in self.protected or self.t - self.born[k] < self.grace or self.alive.sum() <= 1:
                continue
            trial = self.alive.copy(); trial[k] = False
            if base - self.accuracy(cases, trial) < self.lam_unit:
                self.alive[k] = False; self.V[:, k] = 0.0
                base = self.accuracy(cases, self.alive)

    def step(self, s, tgt, on):
        # identical to Pool2.step except (i) attraction can be withheld from
        # juvenile units and (ii) the recruitment trigger can use the miss rate
        alive_now = self.alive.copy()
        p, x = self.predict(s, alive_now)
        e = np.where(on, tgt - p, 0.0)
        if self.pos_weight != 1.0 and tgt[3] > 0.5:
            e[3] *= self.pos_weight
        wrong = (np.abs(tgt - p) > 0.5) & on
        for m in range(self.M):
            if on[m]:
                self.errbuf[m].append((float(wrong[m]), s))
                if len(self.errbuf[m]) > 300:
                    self.errbuf[m] = self.errbuf[m][-200:]
                if tgt[m] > 0.5:
                    self.mbuf[m].append((float(wrong[m]), s))
                    self.mbuf[m] = self.mbuf[m][-60:]
        if not self.frozen and self.alive.any():
            self.V += self.eta_V * np.outer(e, x) * self.alive[None, :]
            self.c += self.eta_V * e
            gsig = e @ self.V
            delta = gsig * x * (1 - x)
            u, d = self.drive(s)
            beta = np.exp(self.b)
            self.w += self.eta_w * (delta[:, None] * 2 * beta * d) * alive_now[:, None]
            att = alive_now.copy()
            if self.juvenile:
                att &= (self.t - self.born) >= self.grace
            for k in self.anchored:
                att[k] = False
            self.w += self.eta_att * (x[:, None] * d) * att[:, None]
            gb = delta[:, None] * (-beta * d ** 2)
            self.b += self.eta_b * (gb - self.lam_b * beta) * alive_now[:, None]
            np.clip(self.b, self.b_min, self.b_max, out=self.b)
        if not self.frozen:
            if self.do_recruit and self.t % 250 == 0:
                for m in range(self.M):
                    if not on[m]:
                        continue
                    if self.trigger == "miss":
                        eb = self.mbuf[m]
                        ok = len(eb) >= 20
                    else:
                        eb = self.errbuf[m][-150:]
                        ok = len(eb) >= 100
                    if ok and np.mean([w_ for w_, _ in eb]) > self.err_thresh:
                        bad = [s_ for w_, s_ in eb if w_ > 0]
                        if bad:
                            self.recruit_at(bad[int(self.rng.integers(len(bad)))])
                            if self.trigger == "miss":
                                self.mbuf[m] = []
                            break
            if not self.alive.any():
                self.recruit_at(s)
            self.buf.append((s, tgt, on))
            if len(self.buf) > 600:
                self.buf = self.buf[-400:]
            if self.do_prune and self.t % 500 == 0 and self.t > 0:
                self.prune()
        self.t += 1


def item(rng, frequent=False, t=0):
    if frequent and t >= ONSET and rng.random() < 0.30:
        while True:
            h = rng.normal(AMP, NOISE)
            if h > 1.4:
                break
        f = np.array([1, 1, int(rng.integers(0, 2))])
        s = np.zeros(D); s[0] = h
        s[1:3] = (2 * f[1:] - 1) * AMP + rng.normal(0, NOISE, 2)
        s[3] = rng.normal(0, 1)
        return s, f
    f = rng.integers(0, 2, 3)
    s = np.zeros(D)
    s[:3] = (2 * f - 1) * AMP + rng.normal(0, NOISE, 3)
    s[3] = rng.normal(0, 1)
    return s, f


def targets(s, f):
    return np.array([float(f[0]), float(f[1]), float(f[2]), float(s[0] > 1.4 and f[1] == 1)])


CONDS = {
    "default": dict(),
    "miss": dict(trigger="miss"),
    "juvenile": dict(juvenile=True),
    "miss+juv": dict(trigger="miss", juvenile=True),
    "oracle": dict(oracle=True),
    "frequent": dict(frequent=True),
    "valued": dict(trigger="miss", juvenile=True, pos_weight=5.0),
    "oracle-kept": dict(oracle=True, keep=True),
    # added after the full run, declared: which of the two protections matters?
    "oracle-noprune": dict(oracle=True, keep_prune=True),
    "oracle-anchored": dict(oracle=True, keep_att=True),
}


def run(cond, seed):
    cfg = dict(CONDS[cond])
    oracle = cfg.pop("oracle", False); frequent = cfg.pop("frequent", False); keep = cfg.pop("keep", False)
    keep_prune = cfg.pop("keep_prune", False) or keep; keep_att = cfg.pop("keep_att", False) or keep
    P = Pool3(D, M, seed, mode="local", **cfg)
    env = np.random.default_rng(1_000 + seed)
    for t in range(T):
        if t == ONSET and oracle:
            k = P.recruit_at(np.array([2.0, AMP, 0.0, 0.0]))
            P.b[k] = np.log(np.array([3.0, 1.5, 0.05, 0.05]))
            if keep_prune:
                P.protected.add(int(k))
            if keep_att:
                P.anchored.add(int(k))
        s, f = item(env, frequent, t)
        on = np.array([True, True, True, t >= ONSET])
        P.step(s, targets(s, f), on)
    P.frozen = True
    rng = np.random.default_rng(66_000 + seed)
    cases = [(s, targets(s, f), np.ones(M, bool)) for s, f in (item(rng) for _ in range(1500))]
    full = P.acc_by_target(cases)
    pos = [c for c in cases if c[1][3] > 0.5]
    recall = float(np.mean([P.predict(s)[0][3] > 0.5 for s, _, _ in pos]))
    neg = [c for c in cases if c[1][3] < 0.5]
    false_alarm = float(np.mean([P.predict(s)[0][3] > 0.5 for s, _, _ in neg]))
    # edge sharpness: among red AND round items only, how well does the CHOICE
    # output separate hue > 1.4 from hue < 1.4 (AUC)?  A learner that merely says
    # "yes" to red round things scores 0.5 here, whatever its recall.
    rr = [(P.predict(s)[0][3], t[3]) for s, t, _ in cases if t[0] > 0.5 and t[1] > 0.5]
    sp = np.array([a for a, y in rr if y > 0.5]); sn = np.array([a for a, y in rr if y < 0.5])
    auc = float(np.mean(sp[:, None] > sn[None, :]) + 0.5 * np.mean(sp[:, None] == sn[None, :]))
    units = []
    for k in np.where(P.alive)[0]:
        a = P.alive.copy(); a[k] = False
        drop = float(full[3] - P.acc_by_target(cases, a)[3])
        beta = np.exp(P.b[k][:3])
        units.append(dict(k=int(k), born=int(P.born[k]), centre=P.w[k][:3].round(2).tolist(),
                          share=(beta / beta.sum()).round(2).tolist(), drop=drop))
    carrier = max(units, key=lambda u: u["drop"])
    return dict(cond=cond, seed=seed, acc=full.tolist(), recall=recall, false_alarm=false_alarm, edge_auc=auc, n_units=len(units),
                new_units=sum(u["born"] >= ONSET for u in units), carrier=carrier, units=units)


def main():
    if "--only" in sys.argv:
        names = sys.argv[sys.argv.index("--only") + 1].split(",")
        prev = json.load(open("results_exp15.json"))
        runs = [r for r in prev["runs"] if r["cond"] not in names]
        for cond in names:
            for seed in range(N_SEEDS):
                r = run(cond, seed); runs.append(r)
                print(f"  {cond:15s} seed {seed}: recall {r['recall']:.2f} FA {r['false_alarm']:.3f} edge AUC {r['edge_auc']:.2f}", flush=True)
            rs = [r for r in runs if r["cond"] == cond]
            print(f"  {cond:15s} edge AUC {np.mean([r['edge_auc'] for r in rs]):.2f} recall {np.mean([r['recall'] for r in rs]):.2f} "
                  f"FA {np.mean([r['false_alarm'] for r in rs]):.3f} solved {np.mean([r['recall'] > 0.8 and r['false_alarm'] < 0.05 for r in rs]):.0%}")
        prev["runs"] = runs
        json.dump(prev, open("results_exp15.json", "w"))
        return
    base = 1 - 0.25 * 0.5 * (1 - __import__("math").erf((1.4 - AMP) / (NOISE * 2 ** 0.5)))
    out = []
    for cond in CONDS:
        for seed in range(N_SEEDS):
            r = run(cond, seed); out.append(r)
            c = r["carrier"]
            print(f"  {cond:9s} seed {seed}: CHOICE acc {r['acc'][3]:.3f} recall {r['recall']:.2f} FA {r['false_alarm']:.2f} "
                  f"edge AUC {r['edge_auc']:.2f}  base jars "
                  f"{np.round(r['acc'][:3],3)}  new units {r['new_units']}  carrier centre {c['centre']} "
                  f"share {c['share']} drop {c['drop']:.3f}", flush=True)
    json.dump(dict(T=T, onset=ONSET, n_seeds=N_SEEDS, baseline_always_no=base, runs=out),
              open("results_exp15.json", "w"))
    print(f"\nSUMMARY (always-'no' baseline {base:.3f})")
    for cond in CONDS:
        rs = [r for r in out if r["cond"] == cond]
        solved = np.mean([r["recall"] > 0.8 and r["false_alarm"] < 0.05 for r in rs])
        print(f"  {cond:9s} CHOICE acc {np.mean([r['acc'][3] for r in rs]):.3f}  recall {np.mean([r['recall'] for r in rs]):.2f}  "
              f"FA {np.mean([r['false_alarm'] for r in rs]):.3f}  edge AUC {np.mean([r['edge_auc'] for r in rs]):.2f}  "
              f"solved (recall > 0.8, FA < 0.05) {solved:.0%}  base jars min {np.min([min(r['acc'][:3]) for r in rs]):.3f}")


if __name__ == "__main__":
    main()
