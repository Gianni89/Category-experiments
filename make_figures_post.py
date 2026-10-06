"""Reader-facing figure variants for the Post 3.5 draft.

Re-renders three figures with blog titles (no reviewer question numbers) and
copies the others under descriptive names into figs/post/.
"""

from __future__ import annotations

import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import style, tidy

OUT = "figs/post"


def save(fig, name, mode):
    fig.savefig(f"{OUT}/{name}_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def oblique(mode):
    t = style(mode)
    d = json.load(open("results_exp9.json"))
    order = [("aligned-axis", "useful features\nline up with\nthe input"),
             ("oblique-axis", "features rotated;\njars can only\nweight inputs"),
             ("oblique-dir", "features rotated;\njars can learn\na direction"),
             ("oblique-dir-G", "same, but jars\ncompete for\nevery case")]
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    for i, (k, lab) in enumerate(order):
        v = np.array([np.mean(r["acc_novel"]) for r in d["runs"] if r["name"] == k])
        col = t["s2"] if k == "oblique-dir-G" else t["s1"]
        ax.bar(i, v.mean(), 0.6, color=col, zorder=3)
        ax.errorbar(i, v.mean(), v.std() / np.sqrt(len(v)), color=t["ink2"], lw=0.8, capsize=2, zorder=4)
        ax.text(i, v.mean() + 0.025, f"{v.mean():.2f}", ha="center", fontsize=7.2, color=t["ink2"])
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for _, l in order], fontsize=7)
    ax.set_ylim(0.4, 1.06); ax.axhline(0.5, color=t["axis"], lw=0.8, ls=":")
    ax.text(-0.42, 0.505, "chance", color=t["muted"], fontsize=6.8, ha="left", va="bottom")
    ax.set_ylabel("accuracy on a combination\nnever seen in training")
    ax.set_title("Overlap is not enough: the jars must find the right dimensions", loc="left")
    tidy(ax, t)
    save(fig, "overlap_portability", mode)


def compound(mode):
    t = style(mode)
    d = json.load(open("results_exp10.json"))
    cases = [("linear", "RED AND ROUND"), ("xor", "RED XOR ROUND")]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    w = 0.36
    for i, (c, lab) in enumerate(cases):
        for j, (rec, col, name) in enumerate([(False, t["muted"], "existing jars + a new readout"),
                                               (True, t["s1"], "new jars may be recruited")]):
            rs = [r for r in d["runs"] if r["case"] == c and r["recruit_after_onset"] == rec]
            v = np.array([r["acc"][3] for r in rs])
            ax.bar(i + (j - 0.5) * w, v.mean(), w * 0.92, color=col, zorder=3, label=name if i == 0 else None)
            ax.text(i + (j - 0.5) * w, v.mean() + 0.015, f"{v.mean():.2f}", ha="center", fontsize=7, color=t["ink2"])
            if rec:
                newc = np.mean([r["carrier"]["after_onset"] for r in rs])
                ax.text(i + 0.5 * w, 0.43, f"own jar\nin {newc:.0%}\nof runs", ha="center", fontsize=6.4, color="white")
    ax.set_xticks(range(2)); ax.set_xticklabels([l for _, l in cases], fontsize=7.5)
    ax.set_ylim(0.4, 1.1); ax.axhline(0.5, color=t["axis"], lw=0.8, ls=":")
    ax.set_ylabel("accuracy on the compound")
    ax.set_title("When composition fails, a new jar forms", loc="left")
    ax.legend(loc="upper center", fontsize=7, ncol=1, bbox_to_anchor=(0.5, -0.14))
    tidy(ax, t)
    save(fig, "compound_jars", mode)


def contrast(mode):
    t = style(mode)
    d = json.load(open("results_exp12.json"))
    L = [("jar", "jar learner", t["s1"]), ("exemplar", "exemplar learner", t["s2"]),
         ("generative", "generative learner", t["s3"])]
    ms = [("centre", "where the category's\ncentre sits"), ("typicality", "lopsided\ntypicality"),
          ("discrimination", "sharper discrimination\nwithin the category"), ("boundary", "boundary, with the\ncontrast PRESENT")]
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
        lo, hi = min(0, min(vals)), max(0, max(vals)); r_ = hi - lo
        ax.set_ylim(lo - 0.2 * r_, hi + 0.2 * r_)
        for i, v in enumerate(vals):
            txt = "0" if (L[i][0] == "generative" and m != "boundary") else f"{v:+.2f}"
            ax.text(i, v + (0.05 * r_ if v >= 0 else -0.05 * r_), txt, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.8, color=t["ink2"])
        ax.axhline(0, color=t["axis"], lw=0.8)
        ax.set_xticks([]); ax.set_title(lab, loc="left", fontsize=8)
        if m == "boundary":
            ax.set_facecolor(t["grid"])
        tidy(ax, t)
    axs[0].set_ylabel("near-contrast group\nminus far-contrast group")
    axs[0].legend(loc="upper center", ncol=3, bbox_to_anchor=(2.4, -0.05), fontsize=7.4)
    fig.suptitle("Same positives, different neighbours: what is left once the neighbour is taken away?",
                 x=0.01, y=1.02, ha="left", fontsize=9.2, fontweight="bold", color=t["ink"])
    fig.subplots_adjust(wspace=0.45, top=0.8, bottom=0.14)
    save(fig, "contrast_removed", mode)


COPY = {
    "fig6_lifehistory": "life_history",
    "fig14_own_edge": "subordinate_edge",
    "fig15_unity": "unity_coupling",
    "fig7_damage_repair": "damage_vs_repair",
    "fig9_redundancy": "redundancy",
    "fig16_module_persistence": "same_jar_new_parts",
    "fig13_role_face": "pet_shark",
    "fig8_glue": "same_jar_different_content",
    "fig21_learned_membership": "learned_membership",
    "fig22_generic_machinery": "generic_machinery",
    "fig23_mediated_unity": "mediated_unity",
    "fig24_lineage_histories": "lineage_histories",
    "fig25_shared_vs_generic": "shared_vs_generic",
    "fig26_lineage_specificity": "lineage_specificity",
    "fig27_discriminability": "discriminability",
    "fig28_denoiser_and_profile": "routed_discrimination",
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for mode in ["light", "dark"]:
        for f in [oblique, compound, contrast]:
            f(mode)
        for src, dst in COPY.items():
            shutil.copy(f"figs/{src}_{mode}.png", f"{OUT}/{dst}_{mode}.png")
    print("ok")
