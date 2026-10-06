"""Figure for the round-7 follow-up.

fig27 exp26 contribution to accuracy against contribution to discriminability, and what each
      candidate criterion gets right
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from make_figures import style, tidy

CAPS = ["A", "B", "X"]
CRITERIA = ["contribution", "+ content", "+ solo profile", "+ discriminability"]


def fig27(mode):
    t = style(mode)
    d = json.load(open("results_exp26.json"))
    runs = d["runs"]
    names = runs[0]["names"]
    eps, ceps, aeps = d["eps"], d["content_eps"], d["align_eps"]

    def m_(i, field, cap):
        vals = [r["components"][i][field][cap] for r in runs]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    show = ["a1", "u1", "shared", "gflat", "modulator", "pure gain", "inert"]
    nice = {"a1": "a1\ncooperative", "u1": "u1\nconjunctive", "shared": "shared\nfeature",
            "gflat": "gflat\nconstant", "modulator": "modulator\nrich gain",
            "pure gain": "pure gain", "inert": "inert"}
    ids = [names.index(n) for n in show]

    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (a) accuracy against discriminability
    ax = axs[0]
    x = np.arange(len(show))
    best = lambda i, f: max(m_(i, f, c) for c in CAPS)
    ax.bar(x - 0.19, [best(i, "phi") for i in ids], width=0.34, color=t["s2"],
           label="contribution to accuracy  $\\varphi$", zorder=3)
    ax.bar(x + 0.19, [best(i, "psi") for i in ids], width=0.34, color=t["s1"],
           label="contribution to discriminability  $\\psi$", zorder=3)
    ax.axhline(eps, color=t["muted"], lw=1.0, ls=":")
    ax.text(len(show) - 0.45, eps + 0.006, "$\\varepsilon$", color=t["muted"], fontsize=8, ha="right")
    ax.axhline(0, color=t["axis"], lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([nice[n] for n in show], fontsize=7)
    ax.set_ylabel("Shapley value for the component's\nbest capacity (accuracy units / AUC)")
    ax.set_title("generic support raises accuracy without\nchanging which cases rank above which",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    tidy(ax, t)

    # (b) what each criterion gets right
    ax = axs[1]
    for k, nm in enumerate(show):
        i = names.index(nm)
        built = {c for c in CAPS if i in runs[0]["truth"][c]}
        c0 = {c for c in CAPS if m_(i, "phi", c) >= eps}
        c1 = {c for c in c0 if m_(i, "marginal", c) >= ceps}
        c2 = {c for c in c1 if abs(m_(i, "solo", c)) >= aeps}
        c3 = {c for c in c0 if m_(i, "psi", c) >= eps}
        for j, got in enumerate([c0, c1, c2, c3]):
            ok = got == built
            ax.add_patch(Rectangle((j, len(show) - 1 - k), 0.92, 0.92,
                                   color=t["s3"] if ok else t["s2"], alpha=0.85 if ok else 0.9))
            ax.text(j + 0.46, len(show) - 1 - k + 0.46, "✓" if ok else "✗", ha="center",
                    va="center", fontsize=11, color=t["surface"], fontweight="bold")
    ax.set_xlim(-0.1, 4.0); ax.set_ylim(-0.1, len(show))
    ax.set_xticks([j + 0.46 for j in range(4)])
    ax.set_xticklabels(CRITERIA, fontsize=7.5, rotation=18, ha="right")
    ax.set_yticks([len(show) - 1 - k + 0.46 for k in range(len(show))])
    ax.set_yticklabels(show, fontsize=7.5)
    ax.set_title("does the criterion recover what was built in?", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.savefig(f"figs/fig27_discriminability_light.png".replace("light", mode),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for mode in ["light", "dark"]:
        fig27(mode)
    print("wrote fig27 (light and dark)")
