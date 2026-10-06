"""Figures for the third review round (spine v4).

fig11 contrast history with the contrast removed (Q2, exp12)
fig12 drift of the realiser at stable accuracy (Q3, exp13)
fig13 learning PET's role face (Q4, exp14)
fig14 the own-edge diagnosis (Q5, exp15)
fig15 far-edge coupling within and across jars (Q1, exp16)
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import style, tidy, band


def save(fig, name, mode):
    fig.savefig(f"figs/{name}_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig11(mode):
    t = style(mode)
    d = json.load(open("results_exp12.json"))
    L = [("jar", "jar (Pool2)", t["s1"]), ("exemplar", "exemplar (ALCOVE)", t["s2"]),
         ("generative", "generative", t["s3"])]
    ms = [("centre", "centre shift\n(hue units)"), ("typicality", "typicality asymmetry\n(toward − away)"),
          ("discrimination", "hue discrimination\n(ratio to 2nd dim)"), ("boundary", "boundary\n(contrast PRESENT)")]
    fig, axs = plt.subplots(1, 4, figsize=(9.0, 3.0))
    for ax, (m, lab) in zip(axs, ms):
        vals = []
        for i, (k, name, col) in enumerate(L):
            n = np.array([r[m] for r in d["runs"] if r["learner"] == k and r["group"] == "near"])
            f = np.array([r[m] for r in d["runs"] if r["learner"] == k and r["group"] == "far"])
            diff = n - f
            ax.bar(i, diff.mean(), 0.66, color=col, zorder=3, label=name)
            ax.errorbar(i, diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)), color=t["ink2"], lw=0.8, capsize=2, zorder=4)
            vals.append(diff.mean())
        lo, hi = min(0, min(vals)), max(0, max(vals))
        r_ = hi - lo
        ax.set_ylim(lo - 0.18 * r_, hi + 0.18 * r_)
        for i, v in enumerate(vals):
            txt = "0 (by\nconstruction)" if (L[i][0] == "generative" and m != "boundary") else f"{v:+.2f}"
            ax.text(i, v + (0.04 * r_ if v >= 0 else -0.04 * r_), txt, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.6, color=t["ink2"])
        ax.axhline(0, color=t["axis"], lw=0.8)
        ax.set_xticks([]); ax.set_title(lab, loc="left", fontsize=8)
        if m == "boundary":
            ax.set_facecolor(t["grid"])
        tidy(ax, t)
    axs[0].set_ylabel("NEAR minus FAR group")
    axs[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(2.4, -0.05), fontsize=7.4)
    fig.suptitle("Same positives, different contrast; tested with the contrast removed (first three panels)",
                 x=0.01, y=1.02, ha="left", fontsize=9.2, fontweight="bold", color=t["ink"])
    fig.subplots_adjust(wspace=0.45, top=0.82, bottom=0.14)
    save(fig, "fig11_contrast_online", mode)


def fig12(mode):
    t = style(mode)
    d = json.load(open("results_exp13.json"))
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.7))
    conds = [(False, True, "one jar per job, noisy", t["s1"], "-"),
             (True, True, "redundant jars, noisy", t["s2"], "-"),
             (True, False, "redundant jars, no noise", t["s2"], (0, (3, 2)))]
    for red, noisy, lab, col, ls in conds:
        rs = [r for r in d["runs"] if r["redundant"] == red and r["noisy"] == noisy]
        tt = np.array([x["t"] for x in rs[0]["trace"]]) / 1000
        acc = np.array([[x["acc"][0] for x in r["trace"]] for r in rs])
        old = np.array([[x["old_load"] for x in r["trace"]] for r in rs])
        rsa = np.array([[x["rsa"] for x in r["trace"]] for r in rs])
        for ax, y in zip(axs, [acc, old, rsa]):
            ax.plot(tt, y.mean(0), color=col, ls=ls, label=lab)
            ax.fill_between(tt, y.mean(0) - y.std(0) / np.sqrt(len(y)), y.mean(0) + y.std(0) / np.sqrt(len(y)),
                            color=col, alpha=0.15, lw=0)
    axs[0].set_ylim(0.9, 1.005); axs[0].set_title("accuracy on RIPE", loc="left", fontsize=8.3)
    axs[1].set_ylim(-0.02, 0.55); axs[1].set_title("load still carried by the\nunits present at the start", loc="left", fontsize=8.3)
    axs[2].set_ylim(0, 1.02); axs[2].set_title("similarity of the population\ncode to the start (RSA)", loc="left", fontsize=8.3)
    for ax in axs:
        ax.set_xlabel("thousand trials after steady state"); tidy(ax, t)
    axs[1].legend(loc="center right", fontsize=7)
    fig.suptitle("Accuracy holds while the machinery carrying it is replaced", x=0.01, ha="left",
                 fontsize=9.2, fontweight="bold", color=t["ink"])
    fig.tight_layout()
    save(fig, "fig12_drift", mode)


def fig13(mode):
    t = style(mode)
    d = json.load(open("results_exp14.json"))
    order = [("local", "local jars"), ("linear+local", "jars +\nlocal units"), ("linear+product", "jars +\nproducts"),
             ("linear+product-strict", "products,\nstricter cost*"), ("given-form", "form given\nby hand")]
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 2.9))
    for j, (ax, lab, true) in enumerate(zip(axs, ["housing cost", "handling risk"], [2.70, 0.95])):
        for i, (k, name) in enumerate(order):
            v = np.array([r["pred"]["shark"][j] for r in d["runs"] if r["learner"] == k])
            col = t["muted"] if k == "given-form" else t["s1"]
            ax.bar(i, v.mean(), 0.64, color=col, zorder=3)
            ax.text(i, v.mean() + 0.04 * true, f"{v.mean():.2f}", ha="center", fontsize=6.8, color=t["ink2"])
        ax.axhline(true, color=t["s2"], lw=1.2, ls="--", zorder=5)
        ax.text(len(order) - 0.45, true * 1.02, "true", color=t["s2"], fontsize=7, ha="right", va="bottom")
        ax.set_xticks(range(len(order))); ax.set_xticklabels([n for _, n in order], fontsize=6.6)
        ax.set_title(f"PET SHARK: predicted {lab}", loc="left", fontsize=8.3)
        ax.set_ylim(0, true * 1.25)
        tidy(ax, t)
    axs[0].text(0, -0.36, "* stricter setting chosen after the pilot", transform=axs[0].transAxes,
                color=t["muted"], fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig13_role_face", mode)


def fig14(mode):
    t = style(mode)
    d = json.load(open("results_exp15.json"))
    order = [("default", "default"), ("miss", "miss-rate\ntrigger"), ("juvenile", "juvenile\nprotection"),
             ("miss+juv", "both"), ("valued", "both + misses\ncost 5x*"), ("oracle", "unit placed\nby hand"),
             ("oracle-anchored", "placed, no\ncentre pull**"), ("oracle-noprune", "placed, never\npruned**"),
             ("oracle-kept", "placed, both\nprotections*"), ("frequent", "CHOICE made\ncommon")]
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    for i, (k, name) in enumerate(order):
        v = np.array([r["edge_auc"] for r in d["runs"] if r["cond"] == k])
        ax.bar(i, v.mean(), 0.64, color=t["s1"], zorder=3)
        ax.errorbar(i, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), color=t["ink2"], lw=0.8, capsize=2, zorder=4)
        ax.text(i, v.mean() + 0.025, f"{v.mean():.2f}", ha="center", fontsize=6.8, color=t["ink2"])
    ax.axhline(0.5, color=t["axis"], lw=0.8, ls=":")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([n for _, n in order], fontsize=6.7)
    ax.set_ylim(0.4, 1.06)
    ax.set_ylabel("edge sharpness (AUC)\namong red round items")
    ax.set_title("Carving 'very red' inside RED: what stops it?", loc="left")
    ax.text(0, -0.36, "* added after the pilot   ** added after the full run", transform=ax.transAxes, color=t["muted"], fontsize=6.8)
    tidy(ax, t)
    save(fig, "fig14_own_edge", mode)


def fig15(mode):
    t = style(mode)
    d = json.load(open("results_exp16.json"))
    x = d["deltas"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=False)
    for ax, key, title in [(axs[0], "reach_A_left", "A's own far (left) edge"),
                           (axs[1], "reach_B_left", "the left edge of a DIFFERENT jar, B")]:
        for shared, lab, col in [(False, "separate precisions", t["s1"]), (True, "one shared precision (global gain)", t["s2"])]:
            rs = [r for r in d["runs"] if r["shared"] == shared]
            m = np.array([[r[key] for r in rs if r["delta"] == dd] for dd in x])
            band(ax, x, m.mean(1), m.std(1) / np.sqrt(m.shape[1]), col, lab)
            ax.plot(x, m.mean(1), "o", color=col, ms=3.5, zorder=4)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("distance of A's right-hand contrast")
        tidy(ax, t)
    axs[0].set_ylabel("reach of the edge")
    axs[1].legend(loc="lower right", fontsize=7)
    fig.suptitle("Far-edge coupling appears across jars when they share a parameter", x=0.01, ha="left",
                 fontsize=9.2, fontweight="bold", color=t["ink"])
    fig.tight_layout()
    save(fig, "fig15_unity", mode)


if __name__ == "__main__":
    import os, sys
    os.makedirs("figs", exist_ok=True)
    which = sys.argv[1:] or ["fig11", "fig12", "fig13", "fig14", "fig15"]
    for mode in ["light", "dark"]:
        for f in which:
            globals()[f](mode)
    print("ok")
