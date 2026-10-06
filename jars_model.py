"""
jars_model.py
=============

A small simulated learner for the "jars" formalism (Post 3.5 review).

The point of this model is NOT to be a theory of human concepts. It is to make
three claims of the proposed formalism precise enough to be wrong:

  1. Contrast-shaped boundaries.  Identical positive exemplars + different
     contrast neighbourhoods -> different classifiers.  The model should
     *derive* this from a capacity constraint, not merely permit it.

  2. Two timescales.  "Constitutive" and "modifiable" are different
     measurements, not two names for causal importance.  Constitution is
     fast-timescale counterfactual dependence with slow structure HELD FIXED;
     plasticity is slow-timescale drift.  The doc's Section 11 test conflates them.

  3. Selection under cost.  A distinction is retained iff its measured value
     exceeds its capacity cost, where value is scored against feedback that is
     generated independently of whether the distinction was retained.

ARCHITECTURE
------------
State S_t of a learner:

    w   (N, n_app)   unit centres in appearance-channel space
    b   (N,)         log precision of each unit  (beta_k = exp(b_k))
    a   (N,)         unit bias
    L   (N, N)       lateral relations  -- INSIDE the fast settling loop
    R   (N, C)       jar->consequence relations -- outside the loop by default
    Wc  (C, n_ctx)   context->consequence cue weights
    g   (C,)         gate in [0,1]: how much consequence c feeds BACK into settling
    A   (n_act, N)   action readout, reads the settled classification only

FAST dynamics (structure fixed): settle() iterates the recurrent network to a
fixed point.  This is classification of the current case.

SLOW dynamics (structure changes): local learning rules -- competitive centre
updates, error-driven precision, anti-Hebbian lateral inhibition, delta-rule
readouts, a reward-driven gate, and unit birth/death under a capacity cost.

Migration of a relation from R (pure readout / "belief") into the settling loop
(g_c > 0) is the model's mechanism for the doc's Section 14 permeability claim.

All learning rules are LOCAL, per the doc's own constraint on F (Section 5):
no backpropagation through the settling loop anywhere in this file.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def softmax(v: np.ndarray) -> np.ndarray:
    v = v - np.max(v)
    e = np.exp(v)
    return e / np.sum(e)


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two distributions (base 2, in [0,1])."""
    p = np.clip(p, eps, None); p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda u, v: float(np.sum(u * (np.log2(u) - np.log2(v))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------

@dataclass
class Params:
    n_app: int = 2          # appearance channels (unit matching happens here)
    n_ctx: int = 0          # context channels (drive consequence units)
    N: int = 8              # unit pool size
    C: int = 0              # number of consequence units
    n_act: int = 2          # number of actions

    # fast dynamics
    T_fast: int = 8
    damp: float = 0.6
    gamma: float = 0.8      # recurrent gain

    # slow dynamics (learning rates)
    eta_w: float = 0.020    # competitive centre update
    eta_w_err: float = 0.010  # discriminative repulsion on error
    eta_b: float = 0.010    # precision
    eta_L: float = 0.020    # lateral
    eta_A: float = 0.060    # action readout
    eta_R: float = 0.060    # consequence readout
    eta_Wc: float = 0.060   # context cue weights
    eta_g: float = 0.150    # gate

    # capacity costs
    lam_b: float = 0.020    # unit cost of holding precision (the lambda on K)
    lam_L: float = 0.0006   # lateral decay (relations persist unless unused)
    cost_g: float = 0.010   # cost of holding a relation in the settling loop
    lam_unit: float = 0.02  # accuracy a unit must be worth to be kept

    # bounds
    b_min: float = 0.0
    b_max: float = 4.0
    b0: float = 0.4

    # Assignment sharpness is a structural property of the learning process
    # ("which unit does this episode belong to"), and is deliberately NOT the
    # same quantity as the learned precision b, which shapes classification.
    # Conflating them makes centres collapse to the grand mean whenever the
    # task is easy enough that precision decays to its floor.
    tau_assign: float = 0.30
    L_min: float = -6.0
    L_max: float = 2.0

    # gate machinery
    gate_period: int = 40   # trials between gate updates
    gate_batch: int = 24    # counterfactual probes per gate update

    # unit birth/death
    allow_structure_change: bool = False
    prune_period: int = 400
    prune_buffer: int = 200
    recruit_vigilance: float = 0.40
    recruit_err_window: int = 120
    recruit_err_thresh: float = 0.30

    eps_explore: float = 0.05

    # Architecture switches for robustness studies.  Defaults reproduce the
    # results in the report exactly.
    kernel: str = "gauss"        # "gauss": -beta*|d|^2   "laplace": -beta*|d|_1
    cost_form: str = "linear"    # K = sum beta ("linear"), sum beta^2 ("quadratic"),
                                 #     sum log beta ("log": constant pressure on b)


# ----------------------------------------------------------------------------
# the learner
# ----------------------------------------------------------------------------

class Learner:
    """An adaptive system with two explicitly separated timescales."""

    def __init__(self, p: Params, rng: np.random.Generator, n_init_units: int = 2):
        self.p = p
        self.rng = rng

        N, C = p.N, p.C
        self.w = rng.normal(0.0, 0.35, size=(N, p.n_app))
        self.b = np.full(N, p.b0, dtype=float)
        self.a = np.zeros(N)
        self.L = np.zeros((N, N))
        self.R = np.zeros((N, C))
        self.Wc = np.zeros((C, p.n_ctx)) if p.n_ctx else np.zeros((C, 0))
        self.g = np.zeros(C)
        self.A = np.zeros((p.n_act, N))

        self.alive = np.zeros(N, dtype=bool)
        self.alive[:n_init_units] = True

        # per-relation learning rates (lets us dissociate plasticity from constitution)
        self.eta_R_c = np.full(C, p.eta_R)

        # bookkeeping
        self.t = 0
        self._gate_gain = np.zeros(C)
        self._gate_n = np.zeros(C)
        self._err_hist: list[float] = []
        self._buffer: list[tuple] = []
        self.frozen = False          # when True, slow dynamics are switched off
        self.trace: dict = {}

    def init_centres_from(self, items: np.ndarray):
        """Seed the live units from experienced episodes rather than from noise.

        Units in this architecture are recruited from experience; starting them
        all at the origin makes structure formation depend on luck, which adds
        variance without adding content.
        """
        idx = self.rng.choice(len(items), size=int(self.alive.sum()), replace=False)
        self.w[self.alive] = items[idx][:, :self.p.n_app] + \
            self.rng.normal(0, 0.02, size=(int(self.alive.sum()), self.p.n_app))

    # -- fast dynamics --------------------------------------------------------

    def settle(self, s: np.ndarray, g: np.ndarray | None = None,
               R: np.ndarray | None = None, L: np.ndarray | None = None,
               alive: np.ndarray | None = None):
        """Run the fast dynamics to a fixed point with structure held fixed.

        Returns (x, y): x = distribution over jar units (the classification),
        y = consequence activations.
        """
        p = self.p
        g = self.g if g is None else g
        R = self.R if R is None else R
        L = self.L if L is None else L
        alive = self.alive if alive is None else alive

        app = s[:p.n_app]
        ctx = s[p.n_app:]

        if p.kernel == "laplace":
            d2 = np.sum(np.abs(app[None, :] - self.w), axis=1)
        else:
            d2 = np.sum((app[None, :] - self.w) ** 2, axis=1)
        u = -np.exp(self.b) * d2 + self.a
        u = np.where(alive, u, -1e9)

        # What context says about each consequence, independently of how the
        # case is currently being classified.
        y_cue = sigmoid(self.Wc @ ctx) if p.n_ctx else np.full(p.C, 0.5)

        x = softmax(u)
        for _ in range(p.T_fast):
            if p.C:
                # What the current classification predicts, and the disagreement
                # between that and context.  Feedback is driven by the PREDICTION
                # ERROR, not by the consequence value: a relation whose grouping
                # the classification already respects exerts no force.  (A raw
                # value-feedback term R diag(g) R^T is positive semi-definite and
                # so sharpens settling whatever the relation says, which would
                # reward useless relations for a generic confidence boost.)
                y_td = sigmoid(R.T @ x)
                fb = R @ (g * (y_cue - y_td))
            else:
                fb = 0.0
            h = u + p.gamma * (L @ x + fb)
            h = np.where(alive, h, -1e9)
            x = (1.0 - p.damp) * x + p.damp * softmax(h)
        x = x / x.sum()

        y = sigmoid(R.T @ x) if p.C else np.zeros(0)
        return x, y

    def responsibility(self, app: np.ndarray) -> np.ndarray:
        """Which unit does this episode belong to, for bookkeeping purposes.

        Fixed sharpness: this is the learning process deciding where to file an
        episode, not the classifier deciding what the episode is.
        """
        d2 = np.sum((app[None, :] - self.w) ** 2, axis=1)
        u = np.where(self.alive, -d2 / (2.0 * self.p.tau_assign ** 2), -1e9)
        return softmax(u)

    def act(self, x: np.ndarray, explore: bool = True) -> int:
        z = self.A @ x
        if explore and self.rng.random() < self.p.eps_explore:
            return int(self.rng.integers(self.p.n_act))
        return int(np.argmax(z))

    # -- slow dynamics --------------------------------------------------------

    def step(self, s: np.ndarray, correct_action: int, cons_obs: np.ndarray | None = None):
        """One experience.  Returns a small record of what happened."""
        p = self.p
        x, y = self.settle(s)
        a_hat = self.act(x, explore=not self.frozen)
        r = 1.0 if a_hat == correct_action else 0.0

        if not self.frozen:
            self._learn(s, x, y, a_hat, correct_action, r, cons_obs)

        self.t += 1
        self._err_hist.append(1.0 - r)
        if len(self._err_hist) > 4000:
            self._err_hist = self._err_hist[-2000:]
        self._buffer.append((s.copy(), correct_action))
        if len(self._buffer) > p.prune_buffer * 3:
            self._buffer = self._buffer[-p.prune_buffer * 2:]

        event = self.maybe_change_structure()
        return dict(x=x, y=y, action=a_hat, reward=r, event=event)

    def _learn(self, s, x, y, a_hat, correct_action, r, cons_obs):
        p = self.p
        app = s[:p.n_app]
        ctx = s[p.n_app:]
        err = 1.0 - r

        # --- centres: competitive attraction (assignment responsibility),
        #     discriminative repulsion on error (settled classification).
        resp = self.responsibility(app)
        dw = p.eta_w * resp[:, None] * (app[None, :] - self.w)
        if err > 0:
            # Repulsion is along the unit direction, not the full displacement,
            # so a unit that is already far away cannot be flung further and
            # become permanently dead.
            d = app[None, :] - self.w
            nrm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
            dw -= p.eta_w_err * x[:, None] * (d / nrm)
        self.w += dw * self.alive[:, None]

        # --- precision: gradient ascent on U = V - lam*K with K = sum_k beta_k.
        #     Errors while active push precision up; holding precision costs
        #     in proportion to the precision held.  Equilibrium is therefore
        #     beta_k ~ E[x_k * err] / lam_b -- graded in the error rate, rather
        #     than a threshold that saturates as soon as the task gets easy.
        if p.cost_form == "quadratic":
            pen = p.lam_b * 2.0 * np.exp(2.0 * self.b)
        elif p.cost_form == "log":
            pen = p.lam_b * np.ones_like(self.b)
        else:
            pen = p.lam_b * np.exp(self.b)
        self.b += p.eta_b * (x * err - pen) * self.alive
        np.clip(self.b, p.b_min, p.b_max, out=self.b)

        # --- lateral: co-active units that co-occur with error inhibit each other
        if err > 0:
            outer = np.outer(x, x)
            np.fill_diagonal(outer, 0.0)
            self.L -= p.eta_L * outer
        self.L *= (1.0 - p.lam_L)          # relations decay unless renewed
        np.fill_diagonal(self.L, 0.0)
        np.clip(self.L, p.L_min, p.L_max, out=self.L)

        # --- action readout (supervised by the independently generated feedback)
        self.A[a_hat] -= p.eta_A * err * x
        self.A[correct_action] += p.eta_A * err * x

        # --- consequence readouts and context cues (delta rule).
        #     R and Wc are two independent predictors of the same observable:
        #     R is the jar->consequence relation, Wc the context cue.
        if p.C and cons_obs is not None:
            self.R += self.eta_R_c[None, :] * np.outer(x, cons_obs - y)
            if p.n_ctx:
                y_cue = sigmoid(self.Wc @ ctx)
                self.Wc += p.eta_Wc * np.outer(cons_obs - y_cue, ctx)

        # --- gate: does putting relation c into the settling loop pay?
        if p.C:
            self._gate_probe(s, correct_action)
            if self.t % p.gate_period == 0:
                self._gate_update()

    def _gate_probe(self, s, correct_action):
        """Counterfactual: would settling with / without relation c in the loop
        have produced the correct action?  Independent feedback, so this is not
        'retained because retained'."""
        p = self.p
        if self.rng.random() > (p.gate_batch / p.gate_period):
            return
        c = int(self.rng.integers(p.C))

        g_on = self.g.copy(); g_on[c] = max(self.g[c], 0.60)
        g_off = self.g.copy(); g_off[c] = 0.0

        x_on, _ = self.settle(s, g=g_on)
        x_off, _ = self.settle(s, g=g_off)
        r_on = 1.0 if int(np.argmax(self.A @ x_on)) == correct_action else 0.0
        r_off = 1.0 if int(np.argmax(self.A @ x_off)) == correct_action else 0.0

        self._gate_gain[c] += (r_on - r_off)
        self._gate_n[c] += 1

    def _gate_update(self):
        p = self.p
        for c in range(p.C):
            if self._gate_n[c] >= 3:
                gain = self._gate_gain[c] / self._gate_n[c]
                self.g[c] += p.eta_g * (gain - p.cost_g)
        self.g = np.clip(self.g, 0.0, 1.0)
        self._gate_gain[:] = 0.0
        self._gate_n[:] = 0.0

    # -- structure change (unit birth / death) --------------------------------

    def maybe_change_structure(self):
        """Split (recruit) and prune under U = V - lambda*K.

        V is measured counterfactually against held-out recent feedback; K is
        the unit count.  A unit survives iff removing it costs more accuracy
        than lam_unit.
        """
        p = self.p
        if not p.allow_structure_change or self.frozen:
            return None
        event = None

        if self.t % p.prune_period == 0 and self.t > 0 and len(self._buffer) >= p.prune_buffer:
            buf = self._buffer[-p.prune_buffer:]
            base = self._buffer_accuracy(buf, self.alive)
            # prune: remove any unit not worth its cost
            for k in np.where(self.alive)[0]:
                if self.alive.sum() <= 1:
                    break
                trial_alive = self.alive.copy(); trial_alive[k] = False
                acc = self._buffer_accuracy(buf, trial_alive)
                if base - acc < p.lam_unit:
                    self.alive[k] = False
                    self.w[k] = self.rng.normal(0, 0.35, size=p.n_app)
                    self.b[k] = p.b0
                    self.A[:, k] = 0.0
                    self.R[k, :] = 0.0
                    self.L[k, :] = 0.0
                    self.L[:, k] = 0.0
                    event = ("prune", int(k), self.t)

            # recruit: if recent error is high and a free unit exists, spawn one
            # at the centroid of recently-mis-handled inputs
            recent_err = float(np.mean(self._err_hist[-p.recruit_err_window:])) \
                if len(self._err_hist) >= p.recruit_err_window else 0.0
            free = np.where(~self.alive)[0]
            if recent_err > p.recruit_err_thresh and len(free) > 0:
                bad = [b_[0][:p.n_app] for b_ in buf[-p.recruit_err_window:]]
                errs = self._buffer_errors(buf[-p.recruit_err_window:], self.alive)
                bad = [v for v, e in zip(bad, errs) if e > 0]
                if len(bad) >= 8:
                    k = int(free[0])
                    self.alive[k] = True
                    self.w[k] = np.mean(np.array(bad), axis=0) + self.rng.normal(0, 0.02, p.n_app)
                    self.b[k] = p.b0
                    event = ("recruit", k, self.t)
        return event

    def _buffer_accuracy(self, buf, alive) -> float:
        ok = 0
        for s, ca in buf:
            x, _ = self.settle(s, alive=alive)
            ok += int(np.argmax(self.A @ x) == ca)
        return ok / max(len(buf), 1)

    def _buffer_errors(self, buf, alive) -> list:
        out = []
        for s, ca in buf:
            x, _ = self.settle(s, alive=alive)
            out.append(0.0 if int(np.argmax(self.A @ x)) == ca else 1.0)
        return out

    # -- measurements ---------------------------------------------------------

    def classify_profile(self, probes: np.ndarray) -> np.ndarray:
        """Application profile mu: the classification of each probe (rows sum to 1)."""
        return np.array([self.settle(s)[0] for s in probes])

    def action_profile(self, probes: np.ndarray) -> np.ndarray:
        return np.array([int(np.argmax(self.A @ self.settle(s)[0])) for s in probes])

    def load_bearing(self, probes: np.ndarray, kind: str, idx) -> float:
        """CONSTITUTION, fast timescale.

        Ablate one relation with all slow dynamics held fixed, and measure how
        much the classification of the very next cases changes.  Mean JS
        divergence over the probe set, in [0,1].
        """
        base = self.classify_profile(probes)
        if kind == "R":
            R2 = self.R.copy(); R2[:, idx] = 0.0
            alt = np.array([self.settle(s, R=R2)[0] for s in probes])
        elif kind == "L":
            i, j = idx
            L2 = self.L.copy(); L2[i, j] = 0.0; L2[j, i] = 0.0
            alt = np.array([self.settle(s, L=L2)[0] for s in probes])
        else:
            raise ValueError(kind)
        return float(np.mean([js_divergence(p_, q_) for p_, q_ in zip(base, alt)]))
