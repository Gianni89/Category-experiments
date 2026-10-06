"""
pool.py -- a generalised version of the overlap learner in exp4_overlap.py.

Same principles as exp4 (independent 'local' units or a softmax 'global'
competition; per-channel precision under a capacity cost; error passed back
one layer along each unit's own outgoing weights; recruitment when a target's
error persists; counterfactual pruning under lambda), with switches for the
follow-up experiments:

  targets        supplied by the caller each step (any number)
  directions     each unit also learns ONE preferred direction in input space,
                 so it can become selective for a dimension that is oblique to
                 the channel basis (rank-1 learned metric, still unit-local)
  unit_dropout   units fail at random during learning (makes redundancy pay)
  prune/recruit  switch structure change on or off

exp4_overlap.py is left untouched so its published results stay reproducible.
"""

from __future__ import annotations

import itertools
import numpy as np


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))


def softmax(v):
    v = v - v.max(); e = np.exp(v); return e / e.sum()


class Pool2:
    def __init__(self, D, M, seed, mode="local", lam_b=0.004, lam_unit=0.01, nmax=16,
                 directions=False, unit_dropout=0.0, prune=True, recruit=True,
                 err_thresh=0.10, grace=1500, prune_under_dropout=False):
        self.D, self.M, self.mode = D, M, mode
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros((nmax, D))
        self.b = np.full((nmax, D), np.log(0.8))
        self.dirs = directions
        self.a = self.rng.normal(0, 1, (nmax, D))
        self.a /= np.linalg.norm(self.a, axis=1, keepdims=True)
        self.g = np.full(nmax, np.log(0.5))        # log precision along the learned direction
        self.alive = np.zeros(nmax, bool)
        self.born = np.zeros(nmax, int)
        self.V = np.zeros((M, nmax)); self.c = np.zeros(M)
        self.lam_b, self.lam_unit = lam_b, lam_unit
        self.eta_V, self.eta_w, self.eta_b, self.eta_att, self.eta_a = 0.10, 0.03, 0.04, 0.004, 0.02
        self.offset, self.b_min, self.b_max = 3.0, -6.0, 3.0
        self.unit_dropout, self.do_prune, self.do_recruit = unit_dropout, prune, recruit
        self.err_thresh, self.grace = err_thresh, grace
        # judge a unit's worth under the conditions the system actually runs in
        # (with unreliable components), rather than with every unit guaranteed present
        self.prune_under_dropout = prune_under_dropout
        self.t = 0
        self.buf, self.errbuf = [], {m: [] for m in range(M)}
        self.frozen = False

    # -- fast dynamics ------------------------------------------------------
    def drive(self, s):
        d = s[None, :] - self.w
        u = -np.sum(np.exp(self.b) * d ** 2, axis=1)
        if self.dirs:
            proj = np.sum(self.a * d, axis=1)
            u = u - np.exp(self.g) * proj ** 2
        return u, d

    def act(self, s, alive=None):
        alive = self.alive if alive is None else alive
        u, _ = self.drive(s)
        if self.mode == "local":
            return np.where(alive, sigmoid(u + self.offset), 0.0)
        return np.where(alive, softmax(np.where(alive, u, -1e9)), 0.0)

    def predict(self, s, alive=None):
        x = self.act(s, alive)
        return sigmoid(self.V @ x + self.c), x

    # -- slow dynamics ------------------------------------------------------
    def recruit_at(self, s):
        free = np.where(~self.alive)[0]
        if not len(free):
            return None
        k = int(free[0])
        self.alive[k] = True; self.born[k] = self.t
        self.w[k] = s + self.rng.normal(0, 0.05, self.D)
        self.b[k] = np.log(0.8)
        self.g[k] = np.log(0.5)
        a = self.rng.normal(0, 1, self.D); self.a[k] = a / np.linalg.norm(a)
        self.V[:, k] = 0.0
        return k

    def step(self, s, tgt, on):
        alive_now = self.alive.copy()
        if self.unit_dropout > 0 and not self.frozen:
            alive_now &= self.rng.random(len(alive_now)) > self.unit_dropout
        p, x = self.predict(s, alive_now)
        e = np.where(on, tgt - p, 0.0)
        wrong = (np.abs(tgt - p) > 0.5) & on
        for m in range(self.M):
            if on[m]:
                self.errbuf[m].append((float(wrong[m]), s))
                if len(self.errbuf[m]) > 300:
                    self.errbuf[m] = self.errbuf[m][-200:]

        if not self.frozen and self.alive.any():
            self.V += self.eta_V * np.outer(e, x) * self.alive[None, :]
            self.c += self.eta_V * e
            gsig = e @ self.V
            delta = gsig * x * (1 - x) if self.mode == "local" else x * (gsig - np.sum(x * gsig))
            u, d = self.drive(s)
            beta = np.exp(self.b)
            gw = 2 * beta * d
            if self.dirs:
                proj = np.sum(self.a * d, axis=1)
                gam = np.exp(self.g)
                gw = gw + 2 * gam[:, None] * proj[:, None] * self.a
                # direction: gradient of u wrt a is -2*gam*proj*d ; keep unit norm
                ga = -2 * gam[:, None] * proj[:, None] * d
                self.a += self.eta_a * delta[:, None] * ga * alive_now[:, None]
                self.a /= np.linalg.norm(self.a, axis=1, keepdims=True) + 1e-12
                gg = delta * (-gam * proj ** 2)
                self.g += self.eta_b * (gg - self.lam_b * gam) * alive_now
                np.clip(self.g, self.b_min, self.b_max, out=self.g)
            self.w += self.eta_w * (delta[:, None] * gw) * alive_now[:, None]
            self.w += self.eta_att * (x[:, None] * d) * alive_now[:, None]
            gb = delta[:, None] * (-beta * d ** 2)
            self.b += self.eta_b * (gb - self.lam_b * beta) * alive_now[:, None]
            np.clip(self.b, self.b_min, self.b_max, out=self.b)

        if not self.frozen:
            if self.do_recruit and self.t % 250 == 0:
                for m in range(self.M):
                    eb = self.errbuf[m][-150:]
                    if on[m] and len(eb) >= 100 and np.mean([w_ for w_, _ in eb]) > self.err_thresh:
                        bad = [s_ for w_, s_ in eb if w_ > 0]
                        if bad:
                            self.recruit_at(bad[int(self.rng.integers(len(bad)))])
                            break
            if not self.alive.any():
                self.recruit_at(s)
            self.buf.append((s, tgt, on))
            if len(self.buf) > 600:
                self.buf = self.buf[-400:]
            if self.do_prune and self.t % 500 == 0 and self.t > 0:
                self.prune()
        self.t += 1

    def accuracy(self, cases, alive=None):
        ok, n = 0.0, 0
        for s, tgt, on in cases:
            p, _ = self.predict(s, alive)
            ok += np.sum(((p > 0.5) == (tgt > 0.5)) & on); n += on.sum()
        return ok / max(n, 1)

    def _acc_operating(self, cases, alive, reps=6):
        if not (self.prune_under_dropout and self.unit_dropout > 0):
            return self.accuracy(cases, alive)
        rng = np.random.default_rng(self.t)
        return float(np.mean([self.accuracy(cases, alive & (rng.random(len(alive)) > self.unit_dropout))
                              for _ in range(reps)]))

    def prune(self):
        cases = self.buf[-300:]
        base = self._acc_operating(cases, self.alive)
        for k in np.where(self.alive)[0]:
            if self.t - self.born[k] < self.grace or self.alive.sum() <= 1:
                continue
            trial = self.alive.copy(); trial[k] = False
            if base - self._acc_operating(cases, trial) < self.lam_unit:
                self.alive[k] = False
                self.V[:, k] = 0.0
                base = self._acc_operating(cases, self.alive)

    # -- measurement (learning frozen) -------------------------------------
    def acc_by_target(self, cases, alive=None):
        A = np.zeros(self.M); n = np.zeros(self.M)
        for s, tgt, on in cases:
            p, _ = self.predict(s, alive)
            A += ((p > 0.5) == (tgt > 0.5)) * on; n += on
        return A / np.maximum(n, 1)

    def selectivity(self, n_feat=None, basis=None):
        """Share of a unit's tuning on its top feature direction.

        Without `basis`, tuning is read off the first `n_feat` raw channels.
        With `basis` (rows = true feature directions), tuning is measured along
        those directions instead.
        """
        out = []
        for k in np.where(self.alive)[0]:
            if basis is None:
                prof = np.exp(self.b[k])[:n_feat]
            else:
                M = np.diag(np.exp(self.b[k]))
                if self.dirs:
                    M = M + np.exp(self.g[k]) * np.outer(self.a[k], self.a[k])
                prof = np.array([f @ M @ f for f in basis])
            prof = np.asarray(prof, float)
            out.append((int(k), float(prof.max() / prof.sum()), int(prof.argmax())))
        return out


def shapley(value_fn, players, max_exact=9, n_perm=400, seed=0):
    """Shapley values for a small set of players.

    value_fn(frozenset) -> performance with ONLY those players present.
    Exact for up to `max_exact` players; permutation sampling beyond that.
    """
    from math import factorial
    n = len(players)
    cache = {}

    def v(S):
        S = frozenset(S)
        if S not in cache:
            cache[S] = value_fn(S)
        return cache[S]

    phi = {p: 0.0 for p in players}
    if n <= max_exact:
        for p in players:
            others = [q for q in players if q != p]
            for r in range(len(others) + 1):
                for S in itertools.combinations(others, r):
                    wgt = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                    phi[p] += wgt * (v(set(S) | {p}) - v(S))
        return phi, v
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        order = list(rng.permutation(players))
        S = set()
        prev = v(S)
        for p in order:
            S.add(p)
            cur = v(S)
            phi[p] += (cur - prev) / n_perm
            prev = cur
    return phi, v
