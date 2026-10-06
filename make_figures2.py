"""Reader-facing figures for spine v3 and the second review round.

fig5  far-edge fork (spine section 8)
fig6  life history of a jar (spine section 11)
fig7  immediate damage vs repair (reviewer Q7)
fig8  same classifier, different selection histories (Q17, exp7)
fig9  redundancy and the load-bearing measure (Q5, exp8)
fig10 oblique overlap and compound jars (Q19, Q20; exp9, exp10)

Each figure is written in a light and a dark version.
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import TOK, style, tidy, band


def save(fig, name, mode):
    fig.savefig(f"figs/{name}_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig5
def fig5(mode):
    t = style(mode)
    d = json.load(open("results_exp1R.json"))
    x = d["deltas"]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for key, lab, col in [("baseline", "shared-unit jar (one unit, one precision)", t["s1"]),
                          ("alcove", "exemplar / region learner (ALCOVE)", t["s2"])]:
        a = d["results"][key]["agg"]["reach_left"]
        band(ax, x, a["mean"], a["sem"], col, lab)
        ax.plot(x, a["mean"], "o", color=col, ms=4, zorder=4)
    ax.set_xlabel("distance of the RIGHT-hand contrast class from the positives")
    ax.set_ylabel("reach of the LEFT edge\n(left contrast never moves)")
    ax.set_title("Moving one neighbour: does the far edge move?", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
    tidy(ax, t)
    ax.text(0.01, 0.02, "same positives, same left contrast in every condition; 8 seeds, bands = ±1 s.e.",
            transform=ax.transAxes, color=t["muted"], fontsize=7)
    save(fig, "fig5_faredge", mode)


# ---------------------------------------------------------------- fig6
def fig6(mode):
    t = style(mode)
    d = json.load(open("results_exp3.json"))
    T = np.array(d["t"]); P = d["phase"]; end = 3 * P
    fig, axs = plt.subplots(3, 1, figsize=(6.0, 3.9), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.55, 0.55, 1.6], hspace=0.25))
    # row 1: feature structure present throughout
    axs[0].barh(0, end, left=0, height=0.6, color=t["s3"], alpha=0.8)
    axs[0].text(end / 2, 0, "the two sub-regions are there to be seen", ha="center", va="center",
                color="white", fontsize=7.8, fontweight="bold")
    axs[0].set_ylabel("feature\nstructure", rotation=0, ha="right", va="center")
    # row 2: payoff
    axs[1].barh(0, P, left=P, height=0.6, color=t["s2"], alpha=0.85)
    for x0, lab in [(0, "no payoff"), (2 * P, "no payoff")]:
        axs[1].text(x0 + P / 2, 0, lab, ha="center", va="center", color=t["muted"], fontsize=7.8)
    axs[1].text(1.5 * P, 0, "splitting pays", ha="center", va="center", color="white",
                fontsize=7.8, fontweight="bold")
    axs[1].set_ylabel("payoff for\ntelling them apart", rotation=0, ha="right", va="center")
    for a in axs[:2]:
        a.set_yticks([]); a.set_ylim(-0.5, 0.5)
        for s in ["left", "bottom"]:
            a.spines[s].set_visible(False)
        a.tick_params(axis="x", length=0)
    # row 3: jars
    ax = axs[2]
    band(ax, T, d["n_units_mean"], d["n_units_sem"], t["s1"])
    ax.set_ylim(1.5, 3.5); ax.set_yticks([2, 3])
    ax.set_ylabel("number of jars", rotation=0, ha="right", va="center")
    for x0 in [P, 2 * P]:
        for a in axs:
            a.axvline(x0, color=t["axis"], lw=0.8, ls=":")
    ax.annotate("jar recruited", xy=(P + 900, 3.0), xytext=(P + 900, 3.33), color=t["ink2"],
                fontsize=7.5, arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.6))
    ax.annotate("jar pruned", xy=(2 * P + 900, 2.0), xytext=(2 * P + 900, 2.45), color=t["ink2"],
                fontsize=7.5, arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.6))
    ax.set_xlabel("trials (one continuous life)")
    tidy(ax, t)
    axs[0].set_title("A jar is kept while it pays, not while the pattern is there", loc="left")
    save(fig, "fig6_lifehistory", mode)


# ---------------------------------------------------------------- fig7
def fig7(mode):
    t = style(mode)
    rows = json.load(open("results_exp2.json"))["partb"]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    labs = {0.0: ("relation never revised (plasticity 0)", t["s1"]),
            0.008: ("relation slowly revised", t["s3"]),
            0.1: ("relation quickly revised", t["s2"])}
    for eta, (lab, col) in labs.items():
        rs = [r for r in rows if r["eta"] == eta]
        m = np.array(rs[0]["marks"])
        tr = np.array([r["traj"] for r in rs])
        band(ax, m, tr.mean(0), tr.std(0) / np.sqrt(len(rs)), col, lab)
        ax.plot([0], [tr[:, 0].mean()], "o", color=col, ms=6, zorder=5)
    ax.annotate("immediate damage\n(learning frozen)\n= load-bearing", xy=(0, 0.034), xytext=(160, 0.041),
                color=t["ink2"], fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=0.6))
    ax.set_xlabel("trials of ordinary learning after the relation is cut")
    ax.set_ylabel("how differently it classifies\n(JS divergence from an uncut twin)")
    ax.set_title("Damage is one question; repair is another", loc="left")
    ax.set_ylim(0, 0.047)
    ax.legend(loc="lower left")
    tidy(ax, t)
    ax.text(0.99, 0.97, "10 seeds; bands = ±1 s.e.", transform=ax.transAxes, ha="right", va="top",
            color=t["muted"], fontsize=7)
    save(fig, "fig7_damage_repair", mode)


# ---------------------------------------------------------------- fig8
def fig8(mode):
    t = style(mode)
    d = json.load(open("results_exp7.json"))
    T1 = d["T1"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    chans = ["hue (RED)", "softness (RIPE)", "smell (SAFE)"]
    cols = [t["s2"], t["s1"], t["s3"]]
    for ax, (L, title) in zip(axs, [("A", "A: success depended on RIPE"),
                                    ("B", "B: success depended on SAFE"),
                                    ("C", "C: success depended on RED")]):
        tt = np.array(d["runs"][0]["rec"][L]["t"])
        sh = np.array([r["rec"][L]["shares"] for r in d["runs"]])
        m = sh.mean(0)
        for c in range(3):
            ax.plot(np.r_[-T1, tt], np.r_[m[0, c], m[:, c]], color=cols[c], label=chans[c])
        if L == "A":
            shS = np.array([r["rec"]["S"]["shares"] for r in d["runs"]]).mean(0)
            ax.plot(tt, shS[:, 1], ls=(0, (2, 2)), color=t["ink"], lw=1.1, label="swamp copy of A")
        ax.axvspan(-T1, 0, color=t["grid"], alpha=0.6, lw=0)
        ax.text(-T1 / 2, 0.93, "phase 1:\nall co-occur", ha="center", va="top", color=t["muted"], fontsize=7)
        ax.set_title(title, loc="left", fontsize=8.3)
        ax.set_ylim(0, 1); ax.set_xlabel("trials (phase 2 starts at 0)")
        tidy(ax, t)
    axs[0].set_ylabel("share of the jar's tuning")
    axs[0].legend(loc="center right", fontsize=6.8, bbox_to_anchor=(1.0, 0.55))
    fig.suptitle("Identical jars at the end of phase 1; what they were 'for' shows only when the world comes apart",
                 x=0.01, ha="left", fontsize=9.2, fontweight="bold", color=t["ink"])
    fig.tight_layout()
    save(fig, "fig8_glue", mode)


# ---------------------------------------------------------------- fig9
def fig9(mode):
    t = style(mode)
    d = json.load(open("results_exp8.json"))
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    groups = [("reliable units\n(1–2 hue jars)", False), ("unreliable units\n(3–6 hue jars)", True)]
    meas = [("single ablation, per jar", t["s1"]),
            ("joint ablation of all hue jars", t["s2"]),
            ("Shapley value, per jar", t["s3"]),
            ("Shapley values, summed", t["muted"])]
    w = 0.19
    for gi, (_, unrel) in enumerate(groups):
        rs = [r for r in d["runs"] if r["unreliable"] == unrel]
        single = np.mean([r["ripe"]["single"][str(k)] if str(k) in r["ripe"]["single"] else r["ripe"]["single"][k]
                          for r in rs for k in r["ripe"]["relevant"]])
        joint = np.mean([r["ripe"]["joint"] for r in rs])
        shp = [r["ripe"]["shapley"] for r in rs]
        shap = np.mean([s.get(str(k), s.get(k)) for r, s in zip(rs, shp) for k in r["ripe"]["relevant"]])
        ssum = np.mean([sum(s.get(str(k), s.get(k)) for k in r["ripe"]["relevant"]) for r, s in zip(rs, shp)])
        for mi, (v, (lab, col)) in enumerate(zip([single, joint, shap, ssum], meas)):
            ax.bar(gi + (mi - 1.5) * w, v, w * 0.92, color=col, label=lab if gi == 0 else None, zorder=3)
            ax.text(gi + (mi - 1.5) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.8, color=t["ink2"])
    ax.set_xticks([0, 1]); ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylabel("accuracy lost on RIPE")
    ax.set_ylim(0, 0.62)
    ax.axhline(0.5, color=t["axis"], lw=0.8, ls=":")
    ax.text(1.45, 0.505, "chance", color=t["muted"], fontsize=7, ha="right")
    ax.set_title("Redundancy hides load from one-at-a-time ablation", loc="left")
    ax.legend(loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.2))
    tidy(ax, t)
    save(fig, "fig9_redundancy", mode)


# ---------------------------------------------------------------- fig10
def fig10(mode):
    t = style(mode)
    d9 = json.load(open("results_exp9.json"))
    d10 = json.load(open("results_exp10.json"))
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]
    order = [("aligned-axis", "aligned\nfeatures"), ("oblique-axis", "oblique,\nchannel\nprecision"),
             ("oblique-dir", "oblique,\n+ learned\ndirection"), ("oblique-dir-cost", "… +\nstronger\ncost*"),
             ("oblique-dir-G", "… global\ncompetition")]
    for i, (k, lab) in enumerate(order):
        rs = [r for r in d9["runs"] if r["name"] == k]
        nv = np.array([np.mean(r["acc_novel"]) for r in rs])
        col = t["s1"] if k != "oblique-dir-G" else t["s2"]
        ax.bar(i, nv.mean(), 0.62, color=col, zorder=3)
        ax.errorbar(i, nv.mean(), nv.std() / np.sqrt(len(nv)), color=t["ink2"], lw=0.8, capsize=2, zorder=4)
        ax.text(i, nv.mean() + 0.025, f"{nv.mean():.2f}", ha="center", fontsize=7, color=t["ink2"])
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=6.7)
    ax.set_ylim(0.4, 1.06); ax.axhline(0.5, color=t["axis"], lw=0.8, ls=":")
    ax.set_ylabel("accuracy on the\nnever-seen combination")
    ax.set_title("Q19: overlap with oblique dimensions", loc="left")
    ax.text(0.0, -0.42, "* cost value chosen after the pilot", transform=ax.transAxes,
            color=t["muted"], fontsize=6.8)
    tidy(ax, t)

    ax = axs[1]
    cases = [("linear", "red AND round"), ("xor", "red XOR round"), ("own-edge", "very red\nAND round")]
    w = 0.36
    for i, (c, lab) in enumerate(cases):
        for j, (rec, col, name) in enumerate([(False, t["muted"], "existing jars + new readout"),
                                               (True, t["s1"], "recruitment allowed")]):
            rs = [r for r in d10["runs"] if r["case"] == c and r["recruit_after_onset"] == rec]
            v = np.array([r["acc"][3] for r in rs])
            ax.bar(i + (j - 0.5) * w, v.mean(), w * 0.92, color=col, zorder=3, label=name if i == 0 else None)
            newc = np.mean([r["carrier"]["after_onset"] for r in rs])
            if rec:
                ax.text(i + 0.5 * w, v.mean() + 0.02, f"new jar\n{newc:.0%}", ha="center", fontsize=6.5,
                        color=t["ink2"])
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in cases], fontsize=7)
    ax.set_ylim(0.4, 1.12); ax.axhline(0.5, color=t["axis"], lw=0.8, ls=":")
    # always answering "no" to the rare own-edge compound: P(hue>1.4)*P(round)
    from math import erf, sqrt
    base = 1 - 0.5 * 0.5 * (1 - 0.5 * (1 + erf((1.4 - 1.2) / (0.45 * sqrt(2)))))
    ax.plot([1.6, 2.4], [base, base], color=t["s2"], lw=1.2, ls="--", zorder=5)
    ax.text(2.42, base - 0.035, "always\n'no'", color=t["s2"], fontsize=6.5, va="top")
    ax.set_ylabel("accuracy on the compound")
    ax.set_title("Q20: when does a compound get a jar?", loc="left")
    ax.legend(loc="upper center", fontsize=6.8, ncol=2, bbox_to_anchor=(0.5, -0.2))
    tidy(ax, t)
    fig.tight_layout()
    save(fig, "fig10_oblique_compound", mode)


if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)
    for mode in ["light", "dark"]:
        for f in [fig5, fig6, fig7, fig8, fig9, fig10]:
            f(mode)
    print("ok")
