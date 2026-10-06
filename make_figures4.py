"""Figures for round 4.

fig16 module persistence through turnover vs wholesale replacement (exp17)
fig17 one resource vs linked resources: transfer and its confound (exp18)
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import style, tidy


def save(fig, name, mode):
    fig.savefig(f"figs/{name}_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig16(mode):
    t = style(mode)
    d = json.load(open("results_exp17.json"))
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for wholesale, lab, col in [(False, "gradual turnover", t["s1"]), (True, "whole module removed at 10k", t["s2"])]:
        rs = [r for r in d["runs"] if r["wholesale"] == wholesale]
        tt = np.array([x["t"] for x in rs[0]["trace"]]) / 1000
        acc = np.array([[x["acc"][0] for x in r["trace"]] for r in rs])
        start = np.array([[x["load_from_start"] for x in r["trace"]] for r in rs])
        ov = np.array([[x["overlap_load"] for x in r["trace"]] for r in rs])
        for ax, y in zip(axs, [acc, start, ov]):
            m = y.mean(0); e = y.std(0) / np.sqrt(len(y))
            ax.plot(tt, m, color=col, label=lab)
            ax.fill_between(tt, m - e, m + e, color=col, alpha=0.15, lw=0)
        for r in rs:
            for x in r["trace"][1:]:
                if x["overlap_load"] < 1e-3:
                    axs[2].plot(x["t"] / 1000, 0, "v", color=col, ms=5, zorder=5)
    axs[0].set_ylim(0.4, 1.02); axs[0].set_title("accuracy on RIPE\n(the capacity)", loc="left", fontsize=8.3)
    axs[1].set_ylim(-0.03, 1.05); axs[1].set_title("share of the module's load carried\nby its members at the start", loc="left", fontsize=8.3)
    axs[2].set_ylim(-0.05, 1.05); axs[2].set_title("share carried by members of the module\none checkpoint earlier (▼ = chain break)", loc="left", fontsize=8.3)
    for ax in axs:
        ax.set_xlabel("thousand trials"); tidy(ax, t)
    axs[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Same capacity throughout; the jar persists only where the chain of continuations is unbroken",
                 x=0.01, ha="left", fontsize=9.2, fontweight="bold", color=t["ink"])
    fig.tight_layout()
    save(fig, "fig16_module_persistence", mode)


def fig17(mode):
    t = style(mode)
    d = json.load(open("results_exp18.json"))
    arch = [(0.0, "linked\n(sharing 0)"), (0.5, "partial\n(sharing 0.5)"), (1.0, "one resource\n(sharing 1)")]
    panels = [("coupling", "measured coupling of the\ntwo uses (shared load)"),
              ("R1_role_change", "classify-only revision:\nchange in ROLE outputs"),
              ("R2_cls_change", "role-only revision:\nchange in CLASSIFY outputs"),
              ("U_role_change", "upstream DANGER jar recalibrated,\nno learning: change in ROLE outputs")]
    fig, axs = plt.subplots(1, 4, figsize=(9.6, 2.9))
    for ax, (k, title) in zip(axs, panels):
        vals = []
        for i, (s, lab) in enumerate(arch):
            v = np.array([r[k] for r in d["runs"] if r["sharing"] == s])
            col = t["muted"] if k.startswith("U_") else t["s1"]
            ax.bar(i, v.mean(), 0.64, color=col, zorder=3)
            ax.errorbar(i, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color=t["ink2"], lw=0.8, capsize=2, zorder=4)
            vals.append(v.mean())
        top = max(vals) * 1.25 + 1e-6
        ax.set_ylim(0, top)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.03 * top, f"{v:.3f}" if v < 0.1 else f"{v:.2f}", ha="center", fontsize=6.7, color=t["ink2"])
        ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in arch], fontsize=6.4)
        ax.set_title(title, loc="left", fontsize=7.8)
        tidy(ax, t)
    fig.suptitle("Transfer grades with shared structure, but shared inputs transfer more",
                 x=0.01, ha="left", fontsize=9.2, fontweight="bold", color=t["ink"])
    fig.tight_layout()
    save(fig, "fig17_one_resource", mode)


if __name__ == "__main__":
    import os, sys
    os.makedirs("figs", exist_ok=True)
    which = sys.argv[1:] or ["fig16", "fig17"]
    for mode in ["light", "dark"]:
        for f in which:
            globals()[f](mode)
    print("ok")
