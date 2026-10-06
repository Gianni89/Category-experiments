"""Figure for the round-8 follow-up.

fig28 exp27 generic support that improves ordering, and structure that moves the profile but
      not the ranking
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import style, tidy


def fig28(mode):
    t = style(mode)
    d = json.load(open("results_exp27.json"))
    p1, p2 = d["part1"], d["part2"]
    names = p1[0]["names"]
    eps = d["eps"]

    show = ["a1", "shared", "pure gain", "denoiser", "inert"]
    nice = {"a1": "a1\nmember", "shared": "shared\nfeature", "pure gain": "pure gain\ngeneric",
            "denoiser": "denoiser\ngeneric", "inert": "inert"}
    ids = [names.index(n) for n in show]

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.4), gridspec_kw=dict(width_ratios=[1.3, 1]))

    ax = axs[0]
    x = np.arange(len(show))
    best = lambda i, k: max(float(np.mean([r[k][m][i] for r in p1])) for m in range(3))
    unr = lambda i: float(np.mean([r["psi_unrouted"][i] for r in p1]))
    ax.bar(x - 0.27, [best(i, "phi") for i in ids], width=0.25, color=t["s2"],
           label="contribution to accuracy  $\\varphi$", zorder=3)
    ax.bar(x, [best(i, "psi") for i in ids], width=0.25, color=t["s1"],
           label="contribution to discriminability  $\\psi$", zorder=3)
    ax.bar(x + 0.27, [unr(i) for i in ids], width=0.25, color=t["s3"],
           label="$\\psi$ for capacities that do not route it", zorder=3)
    ax.axhline(eps, color=t["muted"], lw=1.0, ls=":")
    ax.text(len(show) - 0.45, eps + 0.005, "$\\varepsilon$", color=t["muted"], fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([nice[n] for n in show], fontsize=7.5)
    ax.set_ylabel("Shapley value (accuracy units / AUC)")
    ax.set_title("the denoiser earns the largest discriminability score\nin the system — "
                 "and earns it off its own readout too", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    tidy(ax, t)

    ax = axs[1]
    n2 = p2[0]["names"]
    show2 = ["d1", "d2", "shaper", "inert"]
    ids2 = [n2.index(n) for n in show2]
    x = np.arange(len(show2))
    gb = lambda i, k: float(np.mean([r[k][i] for r in p2]))
    ax.bar(x - 0.19, [gb(i, "psi_binary") for i in ids2], width=0.34, color=t["s1"],
           label="contribution to the binary ranking", zorder=3)
    ax.bar(x + 0.19, [gb(i, "psi_within") for i in ids2], width=0.34, color=t["s2"],
           label="contribution to structure within the category", zorder=3)
    ax.axhline(0, color=t["axis"], lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(show2, fontsize=8)
    ax.set_ylabel("Shapley value")
    ax.set_ylim(-0.25, 0.95)
    ax.set_title("a resource can carry a category's internal structure\nand be invisible to AUC",
                 fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    tidy(ax, t)

    fig.tight_layout()
    fig.savefig(f"figs/fig28_denoiser_and_profile_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for mode in ["light", "dark"]:
        fig28(mode)
    print("wrote fig28 (light and dark)")
