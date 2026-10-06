"""Figures for round 5.

fig18 coupling matrices for the six constructed systems, with the organisation we built in
fig19 what the two variants recover: partition scores, algorithm agreement, membership
fig20 single ablation against the Shapley value in the redundant and interaction cases
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from make_figures import style, tidy

CASES = ["separate", "shared-gain", "distributed", "overlap", "redundant", "interaction"]
TITLE = {"separate": "two separate capacities", "shared-gain": "plus a shared global gain",
         "distributed": "one distributed capacity", "overlap": "a shared component",
         "redundant": "interchangeable duplicates", "interaction": "a suppressor, no part of A"}


def save(fig, name, mode):
    fig.savefig(f"figs/{name}_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load():
    return json.load(open("results_exp19.json"))


def seq_cmap(t):
    """Sequential ramp from the surface colour to s1, for magnitudes."""
    return LinearSegmentedColormap.from_list("mag", [t["surface"], t["s1"]])


def div_cmap(t):
    """Diverging ramp for signed interactions: s2 negative, surface zero, s1 positive."""
    return LinearSegmentedColormap.from_list("sgn", [t["s2"], t["surface"], t["s1"]])


def labels_for(r):
    """Short component labels: the capacity each one belongs to by construction."""
    caps = r["capacities"]
    lab = []
    for i in range(r["n"]):
        tags = [m for m in caps if i in r["truth"][m]]
        if i == r["special"] and not tags:
            lab.append("k")
        elif i in r["inert"]:
            lab.append("·")
        else:
            lab.append("".join(tags).lower() if tags else "·")
    return lab


def fig18(mode):
    t = style(mode)
    d = load()
    fig, axs = plt.subplots(2, 3, figsize=(9.4, 7.2))
    cm = seq_cmap(t)
    for ax, case in zip(axs.ravel(), CASES):
        r = [x for x in d["runs"] if x["case"] == case and x["seed"] == 0][0]
        C = np.array(r["coupling"])
        im = ax.imshow(C, cmap=cm, vmin=0, vmax=0.16)
        lab = labels_for(r)
        ax.set_xticks(range(r["n"])); ax.set_yticks(range(r["n"]))
        ax.set_xticklabels(lab, fontsize=7); ax.set_yticklabels(lab, fontsize=7)
        # outline the organisation we built in
        for m, col in zip(r["capacities"], [t["s1"], t["s3"]]):
            for i in r["truth"][m]:
                for j in r["truth"][m]:
                    if i != j:
                        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               edgecolor=col, linewidth=1.1, zorder=4))
        ax.set_title(TITLE[case], pad=6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.subplots_adjust(bottom=0.17, top=0.865, hspace=0.30, wspace=0.25)
    cax = fig.add_axes([0.30, 0.105, 0.42, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("coupling  $C_{ij}=\\sum_m |I^{(m)}_{ij}|$,  in accuracy", fontsize=8)
    cb.outline.set_visible(False)
    fig.legend(handles=[Line2D([], [], color=t["s1"], lw=1.4, label="capacity A, as built"),
                        Line2D([], [], color=t["s3"], lw=1.4, label="capacity B, as built")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005), fontsize=8)
    fig.suptitle("Coalition-sensitive coupling in six systems whose organisation is known",
                 fontsize=10.5, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, "labels name the capacity each component was built into:  "
             "a, b, ab for both, k for the suppressor, · for inert",
             ha="center", fontsize=7.6, color=t["muted"])
    save(fig, "fig18_coupling_matrices", mode)


def fig19(mode):
    t = style(mode)
    d = load()
    fig, axs = plt.subplots(1, 3, figsize=(9.4, 3.1),
                            gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.0]))

    # (a) recovery score per algorithm
    ax = axs[0]
    x = np.arange(len(CASES))
    for k, (alg, col) in enumerate(zip(["greedy", "louvain", "threshold"], [t["s1"], t["s2"], t["s3"]])):
        vals = []
        for case in CASES:
            rs = [r for r in d["runs"] if r["case"] == case]
            if rs[0]["disjoint_truth"]:
                vals.append(np.mean([r["variant1"][alg]["ari"] for r in rs]))
            else:
                vals.append(np.mean([min(r["variant1"][alg]["best_jaccard"].values()) for r in rs]))
        ax.bar(x + (k - 1) * 0.27, vals, width=0.25, color=col, label=alg, zorder=3)
    ax.axhline(0.8, color=t["muted"], lw=1.0, ls="--", zorder=2)
    ax.text(len(CASES) - 0.45, 0.815, "pass", color=t["muted"], fontsize=7, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(CASES, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("recovery of the built-in\norganisation (ARI or Jaccard)")
    ax.set_title("variant 1: one partition of the graph", fontsize=9)
    ax.legend(loc="upper center", ncol=3, fontsize=7)
    tidy(ax, t)

    # (b) agreement between algorithms and across edge thresholds
    ax = axs[1]
    agree = [np.mean([r["algorithms_agree"] for r in d["runs"] if r["case"] == case]) for case in CASES]
    stable = [np.mean([r["stable_under_sweep"] for r in d["runs"] if r["case"] == case]) for case in CASES]
    ax.barh(x - 0.2, agree, height=0.36, color=t["s1"], label="algorithms agree", zorder=3)
    ax.barh(x + 0.2, stable, height=0.36, color=t["s2"], label="stable over edge threshold", zorder=3)
    ax.set_yticks(x); ax.set_yticklabels(CASES, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlim(0, 1.3); ax.set_xticks([0, 0.5, 1.0]); ax.set_xlabel("fraction of runs")
    ax.set_title("is the structure in the data,\nor in the algorithm?", fontsize=9)
    ax.legend(loc="center right", fontsize=7)
    tidy(ax, t, grid="x")

    # (c) variant 2: how integrated the recovered members turn out to be
    ax = axs[2]
    integ = [np.mean([r["variant2"]["A"]["integration"] for r in d["runs"] if r["case"] == case])
             for case in CASES]
    sgn = [np.mean([r["variant2"]["A"]["integration_signed"] for r in d["runs"] if r["case"] == case])
           for case in CASES]
    ax.bar(x - 0.18, integ, width=0.33, color=t["s3"], label="mean within-set $|I|$", zorder=3)
    ax.bar(x + 0.18, sgn, width=0.33, color=t["s2"], label="mean within-set $I$ (signed)", zorder=3)
    ax.axhline(0, color=t["axis"], lw=0.9)
    ax.axhline(0.01, color=t["muted"], lw=1.0, ls=":", zorder=2)
    ax.text(-0.45, 0.013, "$\\delta$", color=t["muted"], fontsize=8, ha="left")
    ax.set_xticks(x); ax.set_xticklabels(CASES, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylim(-0.065, 0.135); ax.set_ylabel("integration of capacity A's\nmembers (accuracy)")
    ax.set_title("variant 2: membership recovered\nexactly everywhere (J = 1.00)", fontsize=9)
    ax.legend(loc="lower left", fontsize=7)
    tidy(ax, t)

    fig.tight_layout()
    save(fig, "fig19_recovery", mode)


def fig20(mode):
    t = style(mode)
    d = load()
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for ax, case in zip(axs, ["redundant", "interaction"]):
        r = [x for x in d["runs"] if x["case"] == case and x["seed"] == 0][0]
        keep = sorted(set(r["truth"]["A"]) | ({r["special"]} if r["special"] is not None else set()))
        single = [r["single"]["A"][i] for i in keep]
        phi = [r["shapley"]["A"][i] for i in keep]
        x = np.arange(len(keep))
        lab = [("k" if i == r["special"] else f"a{k+1}") for k, i in enumerate(keep)]
        ax.bar(x - 0.19, single, width=0.34, color=t["s2"], label="single ablation", zorder=3)
        ax.bar(x + 0.19, phi, width=0.34, color=t["s1"], label="Shapley value", zorder=3)
        ax.axhline(0, color=t["axis"], lw=0.9)
        ax.axhline(0.02, color=t["muted"], lw=1.0, ls=":", zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
        ax.set_ylabel("contribution to capacity A\n(accuracy)")
        ax.set_title(TITLE[case])
        tidy(ax, t)
    axs[0].legend(loc="upper left", fontsize=7.5)
    axs[1].text(0.03, 0.06, "dotted line: the membership\nthreshold $\\varepsilon=0.02$",
                transform=axs[1].transAxes, ha="left", fontsize=7, color=t["muted"])
    fig.tight_layout()
    save(fig, "fig20_single_vs_shapley", mode)


if __name__ == "__main__":
    for mode in ["light", "dark"]:
        fig18(mode); fig19(mode); fig20(mode)
    print("wrote fig18, fig19, fig20 (light and dark)")
