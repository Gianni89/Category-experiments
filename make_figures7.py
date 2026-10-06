"""Figures for the round-6 follow-ups.

fig25 exp24 a shared discriminative resource against a rich generic modulator
fig26 exp25 the lineage repair is graded, and relations can carry it
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import style, tidy

CAPS = ["A", "B", "C"]


def save(fig, name, mode):
    fig.savefig(f"figs/{name}_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig25(mode):
    t = style(mode)
    d = json.load(open("results_exp24.json"))
    runs = d["runs"]
    names = runs[0]["names"]
    show = ["a1", "shared", "modulator", "gflat", "in1"]
    label = {"a1": "a1\n(specific)", "shared": "shared\n(feature)", "modulator": "modulator\n(generic)",
             "gflat": "gflat\n(flat)", "in1": "in1\n(inert)"}
    ids = [names.index(n) for n in show]

    def avg(i, field, cap):
        return float(np.mean([r["components"][i][field][cap] for r in runs]))

    def solo(i, cap, key):
        return float(np.mean([r["components"][i]["solo"][cap][key] for r in runs]))

    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2))
    x = np.arange(len(show))

    # (a) contribution
    ax = axs[0]
    for k, (c, col) in enumerate(zip(CAPS, [t["s1"], t["s3"], t["s2"]])):
        ax.bar(x + (k - 1) * 0.27, [avg(i, "phi", c) for i in ids], width=0.25, color=col,
               label=f"capacity {c}", zorder=3)
    ax.axhline(d["eps"], color=t["muted"], lw=1.0, ls=":")
    ax.text(len(show) - 0.45, d["eps"] + 0.004, "$\\varepsilon$", color=t["muted"], fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([label[n] for n in show], fontsize=7)
    ax.set_ylabel("Shapley contribution (accuracy)")
    ax.set_title("contribution admits both", fontsize=9)
    ax.legend(loc="upper center", ncol=3, fontsize=6.8)
    ax.set_ylim(0, 0.175)
    tidy(ax, t)

    # (b) content, identical for the two candidates by construction
    ax = axs[1]
    for k, (c, col) in enumerate(zip(CAPS, [t["s1"], t["s3"], t["s2"]])):
        ax.bar(x + (k - 1) * 0.27, [avg(i, "info", c) for i in ids], width=0.25, color=col, zorder=3)
    ax.axhline(d["content_eps"], color=t["muted"], lw=1.0, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([label[n] for n in show], fontsize=7)
    ax.set_ylabel("discriminative content  $|AUC-0.5|$")
    ax.set_title("content cannot separate them:\nthe two have the same activity", fontsize=9)
    ax.set_ylim(0, 0.62)
    ax.annotate("", xy=(1.0, 0.33), xytext=(2.0, 0.33),
                arrowprops=dict(arrowstyle="<->", color=t["ink2"], lw=1.1))
    ax.text(1.5, 0.355, "identical", ha="center", fontsize=7, color=t["ink2"])
    tidy(ax, t)

    # (c) the solo profile
    ax = axs[2]
    for k, (c, col) in enumerate(zip(CAPS, [t["s1"], t["s3"], t["s2"]])):
        ax.bar(x + (k - 1) * 0.27, [solo(i, c, "alignment") for i in ids], width=0.25,
               color=col, zorder=3)
    ax.axhline(0.10, color=t["muted"], lw=1.0, ls=":")
    ax.axhline(0, color=t["axis"], lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([label[n] for n in show], fontsize=7)
    ax.set_ylabel("solo profile's alignment\nwith the category")
    ax.set_title("the solo profile does:\na multiplier alone changes nothing", fontsize=9)
    ax.set_ylim(-0.08, 0.78)
    tidy(ax, t)

    fig.tight_layout()
    save(fig, "fig25_shared_vs_generic", mode)


def fig26(mode):
    t = style(mode)
    d = json.load(open("results_exp25.json"))
    sweep = d["sweep"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw=dict(width_ratios=[1.3, 1]))

    # (a) the sweep
    ax = axs[0]
    get = lambda w, k: float(np.mean([r[k] for r in d["part1"] if r["w_B"] == w]))
    xs = [1 - w for w in sweep]            # how skewed the survivor is toward A
    ax.plot(xs, [get(w, "plain_overlap") for w in sweep], "o-", color=t["muted"], ms=4,
            label="overlap, criterion as written")
    ax.plot(xs, [get(w, "carried_differential") for w in sweep], "o-", color=t["s1"], ms=4,
            label="carried differential membership")
    ax.axhline(d["tau"], color=t["s2"], lw=1.0, ls="--")
    ax.text(0.02, d["tau"] + 0.012, "$\\tau$", color=t["s2"], fontsize=8)
    for w in sweep:
        ok = get(w, "holds_binary") > 0.5
        ax.plot([1 - w], [-0.035], marker="^" if ok else "v", ms=5,
                color=t["s3"] if ok else t["s2"], clip_on=False)
    ax.set_xlabel("how far the survivor is skewed toward the capacity\n"
                  "(0 = equally shared, 1 = wholly specific)")
    ax.set_ylabel("share of the later organisation\ncarried over")
    ax.set_ylim(-0.06, 0.32)
    ax.set_title("the repair is graded; the binary version is not", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.5, -0.055, "markers below: binary specificity verdict", fontsize=6.6,
            color=t["muted"], ha="center")
    tidy(ax, t)

    # (b) part 2
    ax = axs[1]
    p2 = d["part2"]
    vals = [np.mean([r["carried_membership_differential"] for r in p2]),
            np.mean([r["carried_relational_differential"] for r in p2])]
    ax.bar([0, 1], vals, width=0.5, color=[t["s2"], t["s1"]], zorder=3)
    ax.axhline(d["tau"], color=t["muted"], lw=1.0, ls="--")
    ax.text(1.42, d["tau"] + 0.02, "$\\tau$", color=t["muted"], fontsize=8, ha="right")
    for k, v in enumerate(vals):
        ax.text(k, v + 0.03, "BREAKS" if v < d["tau"] else "holds", ha="center", fontsize=8,
                color=t["s2"] if v < d["tau"] else t["s1"], fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["membership\ndifferential", "relational\ndifferential"],
                                              fontsize=8)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("carried differentiating organisation")
    iA = np.mean([r["I_A_s1s2"] for r in p2]); iB = np.mean([r["I_B_s1s2"] for r in p2])
    ax.set_title(f"equal membership, different relations\n"
                 f"(survivors' interaction: {iA:+.2f} for A, {iB:+.2f} for B)", fontsize=9)
    tidy(ax, t)

    fig.suptitle("What has to carry over for one organisation to continue another",
                 fontsize=10.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig26_lineage_specificity", mode)


if __name__ == "__main__":
    for mode in ["light", "dark"]:
        fig25(mode); fig26(mode)
    print("wrote fig25 and fig26 (light and dark)")
