"""Figures for the round-5 follow-up.

fig21 exp20 the membership diagnostic on learned pools: what is stable and what flickers
fig22 exp21 generic enabling machinery against a genuinely shared component
fig23 exp22 direct unity, mediated unity, and no unity at all
fig24 exp23 three histories, two readings of the lineage criterion
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from make_figures import style, tidy

CONDS = [(False, False), (False, True), (True, False), (True, True)]
CLABEL = {(False, False): "reliable\n+ plastic", (False, True): "reliable\n+ noisy",
          (True, False): "redundant\n+ plastic", (True, True): "redundant\n+ noisy"}


def save(fig, name, mode):
    fig.savefig(f"figs/{name}_{mode}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- fig21
def fig21(mode):
    t = style(mode)
    d = json.load(open("results_exp20.json"))
    runs = d["runs"]

    def cells(red, noi):
        return [r["capacities"][c] for r in runs
                if r["redundant"] == red and r["noisy"] == noi for c in ("RIPE", "ROLLS")]

    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.1))
    x = np.arange(len(CONDS))

    # (a) membership: stable core, flickering margin, and the units left out
    ax = axs[0]
    core, flick, out = [], [], []
    for k in CONDS:
        cs = cells(*k)
        rs = [r for r in runs if r["redundant"] == k[0] and r["noisy"] == k[1]]
        core.append(np.mean([c["n_members"] - len(c["unstable_units"]) for c in cs]))
        flick.append(np.mean([len(c["unstable_units"]) for c in cs]))
        out.append(np.mean([r["n_alive"] for r in rs for _ in range(2)]) - core[-1] - flick[-1])
    ax.bar(x, core, color=t["s1"], label="member in every sample", zorder=3)
    ax.bar(x, flick, bottom=core, color=t["s2"], label="in some samples only", zorder=3)
    ax.bar(x, out, bottom=np.array(core) + np.array(flick), color=t["grid"],
           label="never a member", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([CLABEL[k] for k in CONDS], fontsize=7)
    ax.set_ylabel("live units in the pool")
    ax.set_title("membership is a stable core\nplus a thin margin", fontsize=9)
    ax.legend(loc="upper left", fontsize=6.8)
    tidy(ax, t)

    # (b) the eps knob
    ax = axs[1]
    eps = [float(e) for e in sorted(d["runs"][0]["capacities"]["RIPE"]["eps_jaccard"], key=float)]
    for k, col in zip(CONDS, [t["s1"], t["s3"], t["s2"], t["ink2"]]):
        cs = cells(*k)
        y = [np.mean([c["eps_jaccard"][str(e)] if str(e) in c["eps_jaccard"] else c["eps_jaccard"][e]
                      for c in cs]) for e in eps]
        ax.plot(eps, y, "o-" if k[0] else ("o--" if k[1] else "o:"), ms=3.5, color=col,
                label=CLABEL[k].replace("\n", " "))
    ax.axvline(0.02, color=t["muted"], lw=1.0, ls=":")
    ax.set_xscale("log"); ax.set_xticks(eps); ax.set_xticklabels([str(e) for e in eps], fontsize=7)
    ax.set_xlabel("membership threshold $\\varepsilon$")
    ax.set_ylabel("Jaccard against the\n$\\varepsilon = 0.02$ membership")
    ax.set_ylim(0.5, 1.04)
    ax.set_title("the threshold has a plateau\non learned systems too", fontsize=9)
    ax.legend(loc="lower center", fontsize=6.8)
    tidy(ax, t)

    # (c) how the members hold together
    ax = axs[2]
    absol = [np.mean([c["integration"] for c in cells(*k)]) for k in CONDS]
    sgn = [np.mean([c["integration_signed"] for c in cells(*k)]) for k in CONDS]
    ax.bar(x - 0.18, absol, width=0.33, color=t["s3"], label="mean within-set $|I|$", zorder=3)
    ax.bar(x + 0.18, sgn, width=0.33, color=t["s2"], label="mean within-set $I$ (signed)", zorder=3)
    ax.axhline(0, color=t["axis"], lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([CLABEL[k] for k in CONDS], fontsize=7)
    ax.set_ylabel("integration of the members\n(accuracy)")
    ax.set_title("where redundancy is allowed,\nunity is substitutive", fontsize=9)
    ax.legend(loc="lower left", fontsize=6.8)
    tidy(ax, t)

    fig.tight_layout()
    save(fig, "fig21_learned_membership", mode)


# --------------------------------------------------------------------- fig22
def fig22(mode):
    t = style(mode)
    d = json.load(open("results_exp21.json"))
    runs = d["runs"]
    names = runs[0]["names"]
    caps = ["A", "B", "C"]

    def phi(i, c):
        return float(np.mean([r["components"][i]["phi"][c] for r in runs]))

    def info(i):
        vals = [r["components"][i]["info"] for r in runs]
        if vals[0] is None:
            return None
        return max(float(np.mean([v[c] for v in vals])) for c in caps)

    def surv(i):
        out = []
        for m in range(3):
            v = [r["components"][i]["repair"]["survival"][m] for r in runs]
            v = [q for q in v if q is not None]
            if v:
                out.append(float(np.mean(v)))
        return None if not out else float(np.mean(out))

    show = ["a1", "b1", "c1", "s", "gflat", "gmult", "gdenoise", "in1"]
    ids = [names.index(n) for n in show]

    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.3), gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (a) contribution per capacity
    ax = axs[0]
    x = np.arange(len(show))
    for k, (c, col) in enumerate(zip(caps, [t["s1"], t["s3"], t["s2"]])):
        ax.bar(x + (k - 1) * 0.27, [phi(i, c) for i in ids], width=0.25, color=col,
               label=f"capacity {c}", zorder=3)
    ax.axhline(0.02, color=t["muted"], lw=1.0, ls=":")
    ax.text(len(show) - 0.4, 0.025, "$\\varepsilon$", color=t["muted"], fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(show, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("Shapley contribution (accuracy)")
    ax.set_title("the generic components contribute MORE\nthan the ones that carry the content",
                 fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    tidy(ax, t)

    # (b) the two clauses as a plane.  a1, b1 and c1 sit on the same point, so one stands in.
    ax = axs[1]
    plane = ["a1", "s", "gflat", "in1", "gmult", "gdenoise"]
    offs = {"a1": (0, 11), "s": (0, 11), "gflat": (0, 11), "in1": (0, 11),
            "gmult": (33, -3), "gdenoise": (36, -3)}
    noact_y = {"gmult": -0.030, "gdenoise": -0.068}
    ax.axhline(0.0, color=t["axis"], lw=0.8, zorder=1)
    for nm in plane:
        i = names.index(nm)
        inf = info(i)
        br = float(np.mean([runs[r]["components"][i]["breadth"] for r in range(len(runs))]))
        y = noact_y[nm] if inf is None else inf
        member = any(phi(i, c) >= 0.02 for c in caps)
        col = t["s1"] if (member and inf is not None and inf > 0.08) else \
            (t["s2"] if member else t["muted"])
        bx = br + (0.0 if inf is not None else (-0.05 if nm == "gmult" else 0.05))
        ax.scatter([bx], [y], s=75, color=col, zorder=4, marker="o" if inf is not None else "s")
        ax.annotate("a1, b1, c1" if nm == "a1" else nm, (bx, y), textcoords="offset points",
                    xytext=offs[nm], ha="center", fontsize=7.5, color=t["ink2"], zorder=5)
    ax.axhline(0.08, color=t["muted"], lw=1.0, ls=":")
    ax.text(1.46, 0.095, "carries discriminative content", fontsize=7, color=t["muted"], ha="right")
    ax.set_xlim(-0.12, 1.5); ax.set_ylim(-0.09, 0.62)
    ax.text(0.96, -0.049, "no activation of\ntheir own", ha="right", fontsize=6.6,
            color=t["muted"], va="center")
    ax.set_xlabel("breadth: share of capacities contributed to")
    ax.set_ylabel("discriminative content  $|AUC - 0.5|$")
    ax.set_title("two clauses separate what one cannot", fontsize=9)
    ax.text(0.10, 0.25, "blue: member on both clauses\norange: contributes, carries nothing\n"
            "grey: carries something, contributes nothing",
            fontsize=6.8, color=t["ink2"], va="top")
    tidy(ax, t, grid="both")

    fig.tight_layout()
    save(fig, "fig22_generic_machinery", mode)


# --------------------------------------------------------------------- fig23
def fig23(mode):
    t = style(mode)
    d = json.load(open("results_exp22.json"))
    runs = d["runs"]
    kinds = ["cooperative", "mediated", "external"]
    title = {"cooperative": "cooperative\ndirect integration",
             "mediated": "mediated\none common mechanism",
             "external": "external\ncombined outside the system"}

    fig, axs = plt.subplots(1, 4, figsize=(10.6, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25])
                            )
    for ax, kind in zip(axs[:3], kinds):
        r = [x for x in runs if x["kind"] == kind][0]
        names = r["names"]
        inter = np.array(r["interaction"])
        mem = r["members"]
        idx = [names.index(m) for m in mem]
        # layout: hub (if any) in the middle, the rest on a ring
        hub = r["hub"]
        pos = {}
        if r["hub_score"] > 0.9 and len(mem) > 3:          # a star, drawn as a star
            pos[hub] = (0.0, 0.0)
            ring = [m for m in mem if m != hub]
            for k, m in enumerate(ring):
                a = 2 * np.pi * k / len(ring) + np.pi / 2
                pos[m] = (np.cos(a), np.sin(a))
        elif r["n_pieces"] > 1:                            # each piece kept together
            for g, piece in enumerate(r["pieces"]):
                a = 2 * np.pi * g / r["n_pieces"] + np.pi / 2
                cx, cy = 0.95 * np.cos(a), 0.95 * np.sin(a)
                for k, m in enumerate(piece):
                    b = a + np.pi / 2 + np.pi * k / max(len(piece) - 1, 1)
                    pos[m] = (cx + 0.3 * np.cos(b), cy + 0.3 * np.sin(b))
        else:
            for k, m in enumerate(mem):
                a = 2 * np.pi * k / len(mem) + np.pi / 2
                pos[m] = (0.8 * np.cos(a), 0.8 * np.sin(a))
        mx = max(abs(inter[names.index(i), names.index(j)]) for i, j in itertools.combinations(mem, 2))
        for i, j in itertools.combinations(mem, 2):
            w = abs(inter[names.index(i), names.index(j)])
            if w > d["floor"]:
                ax.plot(*zip(pos[i], pos[j]), color=t["s1"], lw=0.6 + 3.4 * w / mx,
                        alpha=0.85, zorder=2, solid_capstyle="round")
        for m, (px, py) in pos.items():
            is_readout = m.startswith("R")
            ax.add_patch(Circle((px, py), 0.155, color=t["s3"] if is_readout else t["s2"],
                                zorder=3, ec="none"))
            ax.text(px, py, m, ha="center", va="center", fontsize=6.5, color=t["surface"], zorder=4)
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45); ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title[kind], fontsize=9)
        ax.text(0, -1.38, f"pieces: {r['n_pieces']}", ha="center", fontsize=7.5, color=t["ink2"])

    ax = axs[3]
    x = np.arange(3)
    direct = [np.mean([r["among_non_hub"] for r in runs if r["kind"] == k]) for k in kinds]
    hubsc = [np.mean([r["hub_score"] for r in runs if r["kind"] == k]) for k in kinds]
    ax.bar(x - 0.19, direct, width=0.34, color=t["s2"], label="direct integration, mean $|I|$", zorder=3)
    ax2 = ax.twinx()
    ax2.bar(x + 0.19, hubsc, width=0.34, color=t["s1"], label="hub score", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(kinds, fontsize=7.5, rotation=15, ha="right")
    ax.set_ylabel("mean $|I|$ among members\nother than the hub")
    ax2.set_ylabel("share of members the hub\ninteracts with", fontsize=8)
    ax.set_ylim(0, 0.165); ax2.set_ylim(0, 1.5)
    ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(t["axis"])
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=6.6, ncol=1)
    ax.set_title("the measure that tells them apart", fontsize=9)
    tidy(ax, t)

    fig.suptitle("Three ways six components can carry one capacity",
                 fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig23_mediated_unity", mode)


# --------------------------------------------------------------------- fig24
def fig24(mode):
    t = style(mode)
    d = json.load(open("results_exp23.json"))
    hists = ["gradual", "wholesale", "spare-shared"]
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    for row, h in enumerate(hists):
        r = [x for x in d["runs"] if x["history"] == h][0]
        y = len(hists) - 1 - row
        stages = [r["members0"]] + [st["members"] for st in r["steps"]]
        for k, mem in enumerate(stages):
            ax.text(k, y + 0.17, ", ".join(mem), ha="center", fontsize=7, color=t["ink2"])
            ax.scatter([k], [y], s=46, color=t["s1"], zorder=4)
        for st in r["steps"]:
            k = st["step"]
            okp, okr = st["plain"]["ok"], st["repaired"]["ok"]
            ax.annotate("", xy=(k - 0.08, y), xytext=(k - 0.92, y),
                        arrowprops=dict(arrowstyle="-|>", color=t["s1"] if okp else t["s2"],
                                        lw=2.0 if okp else 1.4,
                                        linestyle="-" if okp else (0, (3, 2))))
            mark = ("continues" if okp else "BREAKS") + " / " + ("continues" if okr else "BREAKS")
            col = t["s1"] if (okp and okr) else (t["s2"] if not okp else t["s3"])
            ax.text(k - 0.5, y - 0.19, mark, ha="center", fontsize=6.9, color=col)
            ax.text(k - 0.5, y - 0.34, f"overlap {st['plain']['overlap']:.2f} / "
                                       f"{st['repaired']['overlap']:.2f}",
                    ha="center", fontsize=6.4, color=t["muted"])
        ax.text(-0.75, y, h, ha="right", va="center", fontsize=9, fontweight="bold", color=t["ink"])
    ax.set_xlim(-1.9, 3.5); ax.set_ylim(-0.75, 2.75)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"stage {k}" for k in range(4)], fontsize=8)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Three histories, read by the lineage criterion as it stands / with the repair",
                 fontsize=10, fontweight="bold", pad=12)
    ax.text(3.45, -0.62, "each pair of verdicts and overlaps is as-is / repaired",
            ha="right", fontsize=7, color=t["muted"])
    fig.tight_layout()
    save(fig, "fig24_lineage_histories", mode)


if __name__ == "__main__":
    for mode in ["light", "dark"]:
        fig21(mode); fig22(mode); fig23(mode); fig24(mode)
    print("wrote fig21-fig24 (light and dark)")
