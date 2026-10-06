"""
Experiment 11 -- A minimal operator extension: relations and jar-on-jar application.
====================================================================================

Reviewer questions 12-13.  Build the smallest extension that supports
  * unary jars                         (PET, FISH, SHARK as classifiers)
  * an asymmetric binary relation      (CHASES)
  * novel ordered argument reversal
  * jar-on-jar application             Under(PET, FISH)
  * an unseen combination              Under(PET, SHARK)
and report what hidden assumptions had to be added.  Performance is not the
point.

SIMPLIFICATION, declared.  Entities are represented by the outputs of four
already-learned unary jars -- SIZE, AQUATIC, DANGER, FUR -- approximated here by
noisy attribute values.  Learning those jars is demonstrations 1-4; here they
are taken as given.

PART 1 -- CHASES(a, b): a chases b iff a is dangerous, a is bigger than b, and
they share a habitat.  Asymmetric by construction.
  bilinear        sigma(v_a' L v_b + u'v_a + w'v_b + c), L unconstrained (RESCAL-like)
  ordered-linear  sigma(u'v_a + w'v_b + c): has ordered slots but no interaction
  symmetric       L = L', u = w: bilinear but blind to argument order
  bag             sigma(q'(v_a + v_b) + c): an unordered association
Tests: held-out ordered pairs whose REVERSE was trained; pairs involving a kind
never seen in any CHASES pair.

PART 2 -- Under(MOD, HEAD).  PET is given TWO faces: a classifier face (is this
a pet?) and a ROLE face, a learned map from a kept animal's attributes to the
consequences of keeping it (housing cost, handling risk).
Under(PET, X) = PET's role face applied to X's attributes.
  Under-operator      the above
  fuzzy intersection  min(PET(x), X(x)) -- membership in both
  prototype average   consequences of the average of the PET and X prototypes
  nearest compound    consequences of the most similar STORED pet (memorised compounds)
  generic readout     a small MLP trained on seen pet compounds, input = attributes
Tests: PET SHARK consequences (true housing 2.7, risk 0.95 -- far outside any
pet seen); the ranking of fish kinds under PET FISH (is the goldfish the most
typical PET FISH though it is neither the most typical pet nor the most typical fish?).
"""

from __future__ import annotations

import json
import sys
import numpy as np

QUICK = "--quick" in sys.argv
N_SEEDS = 3 if QUICK else 10

KINDS = {  # size, aquatic, danger, fur
    "wolf":     (0.70, 0, 0.90, 1), "fox":    (0.40, 0, 0.60, 1), "cat":     (0.30, 0, 0.40, 1),
    "dog":      (0.50, 0, 0.30, 1), "mouse":  (0.05, 0, 0.00, 1), "rabbit":  (0.15, 0, 0.00, 1),
    "hamster":  (0.08, 0, 0.00, 1), "horse":  (0.90, 0, 0.10, 1), "cow":     (0.90, 0, 0.10, 1),
    "shark":    (0.90, 1, 0.95, 0), "pike":   (0.35, 1, 0.60, 0), "goldfish":(0.05, 1, 0.00, 0),
    "tuna":     (0.60, 1, 0.20, 0), "minnow": (0.03, 1, 0.00, 0), "koi":     (0.30, 1, 0.00, 0),
}
PETS = ["dog", "cat", "hamster", "rabbit", "goldfish", "koi"]
FISH = ["shark", "pike", "goldfish", "tuna", "minnow", "koi"]
NOISE = 0.04


def profile(k, rng):
    v = np.array(KINDS[k], float)
    return v + rng.normal(0, NOISE, 4)


def sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------- part 1
def chases(a, b):
    A, B = KINDS[a], KINDS[b]
    return float(A[2] > 0.5 and A[0] > B[0] + 0.1 and A[1] == B[1])


def feats(model, va, vb):
    if model == "bilinear":
        return np.concatenate([np.outer(va, vb).ravel(), va, vb, [1.0]])
    if model == "ordered-linear":
        return np.concatenate([va, vb, [1.0]])
    if model == "symmetric":
        o = np.outer(va, vb); o = 0.5 * (o + o.T)
        return np.concatenate([o.ravel(), va + vb, [1.0]])
    if model == "bag":
        return np.concatenate([va + vb, [1.0]])
    raise ValueError(model)


def fit_logistic(X, y, l2=1e-3, iters=3000, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(X @ w)
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
    return w


def part1(seed):
    rng = np.random.default_rng(seed)
    kinds = list(KINDS)
    novel = ["pike", "fox"]                     # never in any CHASES training pair
    known = [k for k in kinds if k not in novel]
    pairs = [(a, b) for a in known for b in known if a != b]
    rng.shuffle(pairs)
    # hold out one direction of some pairs whose reverse stays in training
    held = []
    train = []
    seen = set()
    for a, b in pairs:
        if (b, a) in seen and len(held) < 25 and rng.random() < 0.5:
            held.append((a, b))
        else:
            train.append((a, b)); seen.add((a, b))
    novel_pairs = [(a, b) for a in kinds for b in kinds if a != b and (a in novel or b in novel)]
    out = {}
    for model in ["bilinear", "ordered-linear", "symmetric", "bag"]:
        X, y = [], []
        for rep in range(20):
            for a, b in train:
                X.append(feats(model, profile(a, rng), profile(b, rng))); y.append(chases(a, b))
        w = fit_logistic(np.array(X), np.array(y))

        def acc(ps):
            ok = []
            for a, b in ps:
                for rep in range(10):
                    ok.append((sig(feats(model, profile(a, rng), profile(b, rng)) @ w) > 0.5) == chases(a, b))
            return float(np.mean(ok))

        # reversal pairs where the answer actually differs from the trained direction
        flip = [(a, b) for a, b in held if chases(a, b) != chases(b, a)]
        out[model] = dict(train=acc(train), reversed_held_out=acc(held),
                          reversed_where_answer_flips=acc(flip) if flip else None,
                          novel_kind=acc(novel_pairs), n_flip=len(flip))
    return out


# ---------------------------------------------------------------- part 2
def pet_consequences(v):
    """The world's rules for what keeping an animal as a pet involves."""
    size, aq, danger = v[0], v[1], v[2]
    return np.array([size * (1 + 2 * aq), danger])      # housing cost, handling risk


def role_features(v):
    # the role face must be able to represent interactions among the argument's
    # attributes (housing depends on size x aquatic): a hidden assumption, logged
    return np.array([v[0], v[1], v[2], v[3], v[0] * v[1], 1.0])


def fit_ridge(X, Y, l2=1e-3):
    return np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ Y)


def mlp_fit(X, Y, seed, hidden=16, iters=6000, lr=0.05):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 0.5, (X.shape[1], hidden)); b1 = np.zeros(hidden)
    W2 = r.normal(0, 0.5, (hidden, Y.shape[1])); b2 = np.zeros(Y.shape[1])
    for _ in range(iters):
        H = np.maximum(0, X @ W1 + b1); P = H @ W2 + b2
        G = 2 * (P - Y) / len(X)
        gW2 = H.T @ G; gb2 = G.sum(0)
        GH = (G @ W2.T) * (H > 0)
        gW1 = X.T @ GH; gb1 = GH.sum(0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return lambda x: np.maximum(0, x @ W1 + b1) @ W2 + b2


def part2(seed):
    rng = np.random.default_rng(100 + seed)
    kinds = list(KINDS)
    # classifier faces
    Xc = np.array([np.append(profile(k, rng), 1) for k in kinds for _ in range(30)])
    lab_pet = np.array([float(k in PETS) for k in kinds for _ in range(30)])
    lab_fish = np.array([float(k in FISH) for k in kinds for _ in range(30)])
    lab_shark = np.array([float(k == "shark") for k in kinds for _ in range(30)])
    w_pet, w_fish, w_shark = (fit_logistic(Xc, l) for l in (lab_pet, lab_fish, lab_shark))
    PET = lambda v: sig(np.append(v, 1) @ w_pet)
    FISHj = lambda v: sig(np.append(v, 1) @ w_fish)
    SHARKj = lambda v: sig(np.append(v, 1) @ w_shark)

    # role face of PET: learned only from animals actually kept as pets
    Xr = np.array([role_features(profile(k, rng)) for k in PETS for _ in range(30)])
    Yr = np.array([pet_consequences(np.array(KINDS[k], float)) for k in PETS for _ in range(30)])
    M_pet = fit_ridge(Xr, Yr)
    role = lambda v: role_features(v) @ M_pet

    # generic readout: an MLP on the same pet data, raw attributes in
    Xm = np.array([profile(k, rng) for k in PETS for _ in range(30)])
    mlp = mlp_fit(Xm, Yr, seed)

    shark_inst = [profile("shark", rng) for _ in range(50)]
    true = pet_consequences(np.array(KINDS["shark"], float))
    proto_pet = np.mean([KINDS[k] for k in PETS], 0)
    proto_shark = np.array(KINDS["shark"], float)
    nearest = min(PETS, key=lambda k: np.linalg.norm(np.array(KINDS[k]) - proto_shark))

    res = {}
    res["Under-operator"] = np.mean([role(v) for v in shark_inst], 0).tolist()
    memb = [min(PET(v), SHARKj(v)) for v in shark_inst]
    res["fuzzy intersection"] = dict(membership=float(np.mean(memb)),
                                     consequences=None)   # nothing falls under both
    res["prototype average"] = pet_consequences((proto_pet + proto_shark) / 2).tolist()
    res["nearest stored compound"] = dict(compound=f"PET {nearest.upper()}",
                                          consequences=pet_consequences(np.array(KINDS[nearest], float)).tolist())
    res["generic readout"] = np.mean([mlp(v) for v in shark_inst], 0).tolist()

    # PET FISH: rank fish kinds
    def rank_under(k):
        vs = [profile(k, rng) for _ in range(30)]
        # feasibility under the PET role: low predicted housing cost and risk
        feas = np.mean([sig(4 * (0.8 - role(v)[0])) * sig(4 * (0.5 - role(v)[1])) for v in vs])
        return float(np.mean([FISHj(v) for v in vs]) * feas)
    under_rank = sorted(FISH, key=rank_under, reverse=True)
    pet_rank = sorted(FISH, key=lambda k: -np.mean([PET(profile(k, rng)) for _ in range(30)]))
    typ_fish = sorted(FISH, key=lambda k: np.linalg.norm(np.array(KINDS[k]) - np.mean([KINDS[f] for f in FISH], 0)))
    typ_pet = sorted(PETS, key=lambda k: np.linalg.norm(np.array(KINDS[k]) - proto_pet))
    return dict(true_shark=true.tolist(), shark=res, pet_fish_under=under_rank,
                fish_by_pet_membership=pet_rank, most_typical_fish=typ_fish, most_typical_pet=typ_pet)


def main():
    p1 = [part1(s) for s in range(N_SEEDS)]
    p2 = [part2(s) for s in range(N_SEEDS)]
    json.dump(dict(n_seeds=N_SEEDS, part1=p1, part2=p2), open("results_exp11.json", "w"))

    print("PART 1 -- CHASES (accuracy, mean over seeds)")
    for model in ["bilinear", "ordered-linear", "symmetric", "bag"]:
        g = lambda k: np.mean([r[model][k] for r in p1 if r[model][k] is not None])
        print(f"  {model:15s} trained {g('train'):.3f}  reversed held-out {g('reversed_held_out'):.3f}  "
              f"reversal where answer flips {g('reversed_where_answer_flips'):.3f}  novel kind {g('novel_kind'):.3f}")

    print("\nPART 2 -- Under(PET, SHARK): predicted [housing cost, handling risk]; true", p2[0]["true_shark"])
    for k in ["Under-operator", "prototype average", "generic readout"]:
        v = np.mean([r["shark"][k] for r in p2], 0)
        print(f"  {k:24s} {np.round(v, 2)}")
    print(f"  {'fuzzy intersection':24s} membership in PET-and-SHARK "
          f"{np.mean([r['shark']['fuzzy intersection']['membership'] for r in p2]):.3f} -> no consequences")
    nc = p2[0]["shark"]["nearest stored compound"]
    print(f"  {'nearest stored compound':24s} {nc['compound']} -> {np.round(nc['consequences'], 2)}")
    print("\nPART 2 -- PET FISH ranking of fish kinds")
    print("  Under(PET, FISH):        ", p2[0]["pet_fish_under"])
    print("  fish ranked by PET alone:", p2[0]["fish_by_pet_membership"])
    print("  most typical FISH:       ", p2[0]["most_typical_fish"])
    print("  most typical PET:        ", p2[0]["most_typical_pet"])
    top = [r["pet_fish_under"][0] for r in p2]
    print("  top PET FISH across seeds:", {k: top.count(k) for k in set(top)})


if __name__ == "__main__":
    main()
