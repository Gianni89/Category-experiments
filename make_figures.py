"""Figures for the jars toy model.  Emits a light and a dark version of each."""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TOK = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 s1="#3987e5", s2="#d95926", s3="#199e70"),
}


def style(mode):
    t = TOK[mode]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["axis"], "grid.color": t["grid"],
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"], "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 8.5, "legend.fontsize": 7.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "grid.linewidth": 0.5,
        "lines.linewidth": 2.0, "legend.frameon": False,
        "figure.dpi": 150,
    })
    return t


def tidy(ax, t, grid="y"):
    ax.grid(True, axis=grid, alpha=0.6, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(t["axis"])


def band(ax, x, m, e, color, label=None, alpha=0.16):
    ax.plot(x, m, color=color, label=label, zorder=3)
    ax.fill_between(x, np.array(m) - np.array(e), np.array(m) + np.array(e),
                    color=color, alpha=alpha, linewidth=0, zorder=2)


# ---------------------------------------------------------------------------
# Figure 1 -- contrast-shaped boundaries
# ---------------------------------------------------------------------------

def fig1(mode):
    t = style(mode)
    R = json.load(open("results_exp1.json"))
    grid = np.array(R["grid"]); deltas = R["deltas"]
    runs = R["runs"]
    m = lambda k: np.array([runs[str(d)][k]["mean"] for d in deltas])
    e = lambda k: np.array([runs[str(d)][k]["sem"] for d in deltas])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.9))
    fig.suptitle("Identical positives, different neighbours: the boundaries follow the neighbours",
                 fontsize=11.5, fontweight="bold", color=t["ink"], y=0.985)

    # (a) profiles
    ax = axes[0, 0]
    show = [1.00, 2.50, 3.50]
    cols = [t["s2"], t["s3"], t["s1"]]
    for i, (d, c) in enumerate(zip(show, cols)):
        a = runs[str(d)]
        xp = np.array(a["xp_mean"])
        ax.plot(grid, xp, color=c, zorder=3)
        br = a["b_right"]["mean"]
        ax.plot([br], [np.interp(br, grid, xp)], "o", ms=5, color=c,
                mec=t["surface"], mew=1.5, zorder=5)
        ax.text(0.60, 0.94 - 0.085 * i, f"right contrast at {d:.1f}",
                transform=ax.transAxes, color=c, fontsize=7.8, fontweight="bold")
    ax.text(0.60, 0.94 - 0.085 * 3 - 0.02, "dots mark the boundary",
            transform=ax.transAxes, color=t["muted"], fontsize=7.2)
    pr = m("proto").mean()
    ax.axvline(pr, color=t["muted"], ls=(0, (3, 3)), lw=1.0, zorder=1)
    ax.annotate("prototype\n(all conditions)", xy=(pr, 0.62), xytext=(-52, 0),
                textcoords="offset points", color=t["ink2"], fontsize=7.4, ha="left")
    ax.axvline(R["left"], color=t["axis"], lw=1.0, zorder=1)
    ax.annotate("fixed left flank", xy=(R["left"], 0.5), xytext=(4, 0),
                textcoords="offset points", color=t["muted"], fontsize=7.2, rotation=90,
                va="center")
    ax.set_xlim(-3.4, 4.4); ax.set_ylim(0, 1.05)
    ax.set_xlabel("position on the varied channel"); ax.set_ylabel("activation of the positive jar")
    ax.set_title("a.  The classifier, at three contrast distances", loc="left")
    ax.plot([], [], color=t["muted"], ls=(0, (3, 3)), label="prototype")
    tidy(ax, t, grid="both")

    # (b) boundary vs prototype travel
    ax = axes[0, 1]
    ax.errorbar(deltas, m("b_right"), yerr=e("b_right"), color=t["s1"], marker="o",
                ms=5, capsize=2.5, elinewidth=1.0, mec=t["surface"], mew=1.2, zorder=4)
    ax.errorbar(deltas, m("proto"), yerr=e("proto"), color=t["s2"], marker="s",
                ms=5, capsize=2.5, elinewidth=1.0, mec=t["surface"], mew=1.2, zorder=4)
    ax.annotate("category boundary", xy=(deltas[-1], m("b_right")[-1]), xytext=(-4, 10),
                textcoords="offset points", color=t["s1"], fontsize=8, ha="right",
                fontweight="bold")
    ax.annotate("prototype", xy=(deltas[-1], m("proto")[-1]), xytext=(-4, 10),
                textcoords="offset points", color=t["s2"], fontsize=8, ha="right",
                fontweight="bold")
    trav_b = m("b_right").max() - m("b_right").min()
    trav_p = m("proto").max() - m("proto").min()
    ax.text(0.03, 0.94, f"boundary travels {trav_b/max(trav_p,1e-9):.0f}x further\n"
                        f"than the prototype ({trav_b:.2f} vs {trav_p:.2f})",
            transform=ax.transAxes, va="top", color=t["ink2"], fontsize=7.8)
    ax.set_xlabel("distance to the right-hand contrast class")
    ax.set_ylabel("position on the varied channel")
    ax.set_title("b.  What moves in this learner", loc="left")
    tidy(ax, t)

    # (c) asymmetry with zero crossing + robustness
    ax = axes[1, 0]
    rob = R.get("robust", {})
    for lam, r in rob.items():
        ax.plot(deltas, r["asym_reach"], color=t["muted"], lw=1.0, alpha=0.55, zorder=2)
    if rob:
        ax.text(0.60, 0.95, "same pattern across\nthe price of precision",
                transform=ax.transAxes, va="top", color=t["muted"], fontsize=7.2)
    ax.text(0.04, 0.30, "the dip below 1.5 is fragile: it vanishes\n"
                        "under a Laplacian kernel, so it is not\n"
                        "offered as a prediction",
            transform=ax.transAxes, va="top", color=t["ink2"], fontsize=7.0)
    ax.errorbar(deltas, m("asym_reach"), yerr=e("asym_reach"), color=t["s1"],
                marker="o", ms=5, capsize=2.5, elinewidth=1.0, mec=t["surface"],
                mew=1.2, zorder=4)
    ax.axhline(0, color=t["axis"], lw=1.0, zorder=1)
    ax.axvline(-R["left"], color=t["s3"], ls=(0, (4, 3)), lw=1.4, zorder=2)
    ax.annotate("flanks symmetric here", xy=(-R["left"], 0), xytext=(6, 28),
                textcoords="offset points", color=t["s3"], fontsize=7.6, fontweight="bold")
    ax.set_xlabel("distance to the right-hand contrast class")
    ax.set_ylabel("reach left  −  reach right")
    ax.set_title("c.  Asymmetry: zero at symmetry (generic)", loc="left")
    tidy(ax, t)

    # (d) opposite flank
    ax = axes[1, 1]
    ax.errorbar(deltas, m("reach_left"), yerr=e("reach_left"), color=t["s3"],
                marker="D", ms=4.5, capsize=2.5, elinewidth=1.0, mec=t["surface"],
                mew=1.2, zorder=4)
    rl = m("reach_left")
    ax.text(0.40, 0.30, "the LEFT flank never moved,\n"
                        f"yet the category's left reach\nshrinks by {rl.max()-rl.min():.2f} "
                        "as the right\ncontrast closes in",
            transform=ax.transAxes, va="top", color=t["ink2"], fontsize=7.8)
    ax.set_xlabel("distance to the right-hand contrast class")
    ax.set_ylabel("reach on the untouched left side")
    ax.set_title("d.  The opposite-flank effect (unit-based learners)", loc="left")
    tidy(ax, t)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"fig1_contrast_{mode}.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- two timescales
# ---------------------------------------------------------------------------

def fig2(mode):
    t = style(mode)
    R = json.load(open("results_exp2.json"))
    tcs = R["timecourse"]
    ts = np.array(tcs[0]["t"])

    def agg(key):
        A = np.array([tc[key] for tc in tcs])
        return A.mean(0), A.std(0) / np.sqrt(A.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.9))
    fig.suptitle("Classificatory load-bearing and modifiability are two axes; the Section 11 test reads the wrong one",
                 fontsize=11.5, fontweight="bold", color=t["ink"], y=0.985)

    # (a) gate
    ax = axes[0, 0]
    m, e = agg("g_danger"); band(ax, ts, m, e, t["s1"])
    ax.annotate("crosscutting relation\n(resolves the hard cases)",
                xy=(ts[-1], m[-1]), xytext=(-8, -30), textcoords="offset points",
                color=t["s1"], fontsize=7.8, ha="right", fontweight="bold")
    m2, e2 = agg("g_parallel"); band(ax, ts, m2, e2, t["s2"])
    ax.annotate("parallel relation\n(equally entrenched, no work to do)",
                xy=(ts[-1], m2[-1]), xytext=(-8, 16), textcoords="offset points",
                color=t["s2"], fontsize=7.8, ha="right", fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("trials"); ax.set_ylabel("gate: relation inside the settling loop")
    ax.set_title("a.  A belief migrates into the classifier", loc="left")
    tidy(ax, t)

    # (b) load-bearing
    ax = axes[0, 1]
    m, e = agg("lb_danger"); band(ax, ts, m, e, t["s1"], "crosscutting")
    m2, e2 = agg("lb_parallel"); band(ax, ts, m2, e2, t["s2"], "parallel")
    ax.annotate("crosscutting", xy=(ts[-1], m[-1]), xytext=(-8, 8),
                textcoords="offset points", color=t["s1"], fontsize=8,
                ha="right", fontweight="bold")
    ax.annotate("parallel", xy=(ts[-1], m2[-1]), xytext=(-8, 8),
                textcoords="offset points", color=t["s2"], fontsize=8,
                ha="right", fontweight="bold")
    ax.set_xlabel("trials")
    ax.set_ylabel("load-bearing (structure frozen)")
    ax.set_title("b.  Only then does ablating it change classification", loc="left")
    tidy(ax, t)

    # (c) the 2x2
    ax = axes[1, 0]
    FLOOR = 1e-6
    groups = {
        "in the settling loop": (t["s1"], "o"),
        "pure readout": (t["s2"], "s"),
        "lateral": (t["s3"], "^"),
    }

    def classify(row):
        if row["cls"] == "lateral":
            return "lateral"
        if row["cls"] == "coherent":
            return "in the settling loop"
        return "pure readout"

    pts = [(classify(r), max(r["pl"], FLOOR), r["lb"]) for r in R["pop"]]
    pts += [("in the settling loop", max(r["pl"], FLOOR), r["lb"]) for r in R["partb"]]
    for name, (c, mk) in groups.items():
        xs = [p[1] for p in pts if p[0] == name]
        ys = [p[2] for p in pts if p[0] == name]
        ax.scatter(xs, ys, s=26, marker=mk, c=c, alpha=0.8, linewidths=0.8,
                   edgecolors=TOK[mode]["surface"], label=name, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(2e-7, 4e-3)
    allpl = [p[1] for p in pts]; alllb = [p[2] for p in pts]
    xm = float(np.exp(np.mean(np.log(allpl))))
    ym = 0.5 * (min(alllb) + max(alllb)) * 0.35
    ax.axvline(xm, color=t["axis"], lw=0.9, zorder=1)
    ax.axhline(ym, color=t["axis"], lw=0.9, zorder=1)
    ax.set_ylim(-0.004, max(alllb) * 1.30)
    ax.text(0.005, 0.99, "load-bearing,\nrarely revised", transform=ax.transAxes,
            va="top", fontsize=7.2, color=t["ink2"])
    ax.text(0.995, 0.99, "load-bearing,\nactively revised", transform=ax.transAxes,
            va="top", ha="right", fontsize=7.2, color=t["ink2"])
    ax.text(0.005, 0.10, "dead\nstructure", transform=ax.transAxes,
            va="top", fontsize=7.2, color=t["muted"])
    ax.text(0.995, 0.32, "revised constantly,\nbears no load", transform=ax.transAxes,
            va="top", ha="right", fontsize=7.2, color=t["muted"])
    ax.text(0.005, 0.33, "left edge =\nno measured\ndrift", transform=ax.transAxes,
            va="top", fontsize=6.9, color=t["muted"])
    ax.set_xlabel("plasticity — drift per trial (log, clipped at 1e-6)")
    ax.set_ylabel("load-bearing (structure frozen)")
    ax.set_title("c.  Every cell is occupied", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.27, 0.80), labelcolor=t["ink2"])
    tidy(ax, t, grid="both")

    # (d) Part B: repair
    ax = axes[1, 1]
    etas = sorted({r["eta"] for r in R["partb"]})
    cols = [t["s1"], t["s3"], t["s2"]]
    labs = ["frozen relation", "slowly revised", "actively revised"]
    offs = [(-6, 10), (-6, -18), (-6, 12)]
    for eta, c, lab, off in zip(etas, cols, labs, offs):
        rows = [r for r in R["partb"] if r["eta"] == eta]
        marks = np.array(rows[0]["marks"], dtype=float)
        A = np.array([r["traj"] for r in rows])
        band(ax, marks, A.mean(0), A.std(0) / np.sqrt(len(rows)), c)
        ax.annotate(lab, xy=(marks[-1], A.mean(0)[-1]), xytext=off,
                    textcoords="offset points", color=c, fontsize=7.8,
                    ha="right", fontweight="bold")
    ax.axvline(0, color=t["axis"], lw=1.0)
    ax.set_ylim(-0.004, None)
    ax.text(0.30, 0.46,
            "all three are load-bearing at t = 0 —\nthe actively revised one most of all.\n"
            "The Section 11 verdict is read off\nthe right-hand edge, and inverts\nthe true ranking.",
            transform=ax.transAxes, va="top", fontsize=7.4, color=t["ink2"])
    ax.set_xlabel("trials of relearning after the relation is ablated")
    ax.set_ylabel("change in application profile")
    ax.set_title("d.  The same relation, three verdicts", loc="left")
    tidy(ax, t)

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"fig2_timescales_{mode}.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- selection under a capacity cost
# ---------------------------------------------------------------------------

def fig3(mode):
    t = style(mode)
    R = json.load(open("results_exp3.json"))
    ts = np.array(R["t"]); P = R["phase"]

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.9))
    fig.suptitle("Distinctions appear when they start paying and vanish when they stop",
                 fontsize=11.5, fontweight="bold", color=t["ink"], y=0.985)

    def phases(ax):
        ax.axvspan(P, 2 * P, color=t["s3"], alpha=0.09, lw=0, zorder=0)
        for x in (P, 2 * P):
            ax.axvline(x, color=t["axis"], lw=0.9, ls=(0, (4, 3)), zorder=1)

    # (a) unit count
    ax = axes[0, 0]
    phases(ax)
    band(ax, ts, R["n_units_mean"], R["n_units_sem"], t["s1"])
    ax.set_ylim(1.4, max(R["n_units_mean"]) + 0.7)
    ax.set_xlabel("trials"); ax.set_ylabel("number of jars maintained")
    ax.set_title("a.  Structure follows the payoff", loc="left")
    ax.text(0.015, 0.06, f"all {R['n_seeds']} learners, no exceptions",
            transform=ax.transAxes, fontsize=7.2, color=t["muted"])
    y = ax.get_ylim()[1]
    for x, lab in [(P * 0.5, "same action\nfor both"), (P * 1.5, "different\nactions"),
                   (P * 2.5, "same action\nagain")]:
        ax.text(x, y * 0.995, lab, ha="center", va="top", fontsize=7.3, color=t["ink2"])
    tidy(ax, t)

    # (b) error
    ax = axes[0, 1]
    phases(ax)
    ax.plot(ts, R["err_mean"], color=t["s2"], zorder=3)
    ax.set_xlabel("trials"); ax.set_ylabel("error rate")
    ax.set_title("b.  ...and the error it repairs", loc="left")
    tidy(ax, t)

    # (c) lambda sweep
    ax = axes[1, 0]
    lams = R["lams"]
    frac = [np.mean(R["sweep"][str(l)]) for l in lams]
    gain = R["measured_gain"]
    cols = [t["s1"] if l < gain else t["s2"] for l in lams]
    ax.bar([str(l) for l in lams], frac, color=cols, width=0.62, zorder=3)
    xs = np.interp(gain, lams, np.arange(len(lams)))
    ax.axvline(xs, color=t["s3"], lw=1.6, ls=(0, (4, 3)), zorder=4)
    ax.annotate(f"measured value of the split\n({gain:.2f} accuracy)",
                xy=(xs, 0.62), xytext=(8, 0), textcoords="offset points",
                color=t["s3"], fontsize=7.6, fontweight="bold", va="center")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("λ — the price of maintaining one more jar")
    ax.set_ylabel("fraction of learners that split")
    ax.set_title("c.  The phase boundary lands where the theory says", loc="left")
    tidy(ax, t)

    # (d) Pareto
    ax = axes[1, 1]
    par = R["pareto"]
    cands = par["cands"]; front = set(par["front"])
    never2 = set(par["by_budget"]["2"]["on_front_but_never"]) if "2" in par["by_budget"] \
        else set(par["by_budget"][2]["on_front_but_never"])
    fx = [cands[n][0] for n in cands if n in front and n not in never2]
    fy = [cands[n][1] for n in cands if n in front and n not in never2]
    dx = [cands[n][0] for n in cands if n not in front]
    dy = [cands[n][1] for n in cands if n not in front]
    order = sorted([n for n in cands if n in front], key=lambda n: cands[n][0])
    ax.plot([cands[n][0] for n in order], [cands[n][1] for n in order],
            color=t["axis"], lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.scatter(fx, fy, s=52, c=t["s1"], marker="o", edgecolors=t["surface"],
               linewidths=1.0, label="kept by both rules", zorder=4)
    ax.scatter(dx, dy, s=44, c=t["muted"], marker="x", linewidths=1.3,
               label="dominated — dropped by both", zorder=4)
    for n in never2:
        ax.scatter([cands[n][0]], [cands[n][1]], s=96, c=t["s2"], marker="*",
                   edgecolors=t["surface"], linewidths=0.8,
                   label="undominated, yet no weighting keeps it", zorder=5)
        ax.annotate("kept by a Pareto rule,\ndropped by every\nscalar utility",
                    xy=cands[n], xytext=(16, 18), textcoords="offset points",
                    color=t["s2"], fontsize=7.5, fontweight="bold")
    ax.set_xlabel("value for prediction"); ax.set_ylabel("value for action")
    ax.set_xlim(0, 1.15); ax.set_ylim(0, 1.15)
    ax.set_title("d.  Where a vector of values differs from a scalar", loc="left")
    ax.legend(loc="lower left", labelcolor=t["ink2"], fontsize=7.2)
    tidy(ax, t, grid="both")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"fig3_selection_{mode}.png", dpi=170)
    plt.close(fig)



# ---------------------------------------------------------------------------
# Figure 4 -- overlapping jars
# ---------------------------------------------------------------------------

def fig4(mode):
    t = style(mode)
    R = json.load(open("results_exp4.json"))
    res = R["results"]
    Q = R["T_total"] // 4

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.9))
    fig.suptitle("One input, several jars: overlap needs local competition, and it buys recombination",
                 fontsize=11.5, fontweight="bold", color=t["ink"], y=0.985)

    # (a) jars maintained over time
    ax = axes[0, 0]
    for i, (x, lab) in enumerate([(0, "ripe"), (Q, "+ rolls"), (2 * Q, "+ portion,\n   edible"),
                                  (3 * Q, "ripe edge\nmoves")]):
        ax.axvline(x, color=t["axis"], lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(x + 250, 0.97 if i < 3 else 0.80, lab, transform=ax.get_xaxis_transform(), va="top",
                fontsize=7.0, color=t["muted"])
    for m_, c, lab in [("local", t["s1"], "local competition"), ("global", t["s2"], "global competition")]:
        ts = np.array([r_["t"] for r_ in res[m_][0]["curve"]])
        U = np.array([[r_["n_units"] for r_ in run["curve"]] for run in res[m_]], float)
        band(ax, ts, U.mean(0), U.std(0) / np.sqrt(len(U)), c)
        ax.annotate(lab, xy=(ts[-1], U.mean(0)[-1]), xytext=(-4, 7 if m_ == "global" else -14),
                    textcoords="offset points", color=c, fontsize=7.8, ha="right", fontweight="bold")
    ax.set_ylim(0, None)
    ax.set_xlabel("trials"); ax.set_ylabel("jars maintained")
    ax.set_title("a.  Jars recruited as targets come online", loc="left")
    tidy(ax, t)

    # (b) never-seen combination
    ax = axes[0, 1]
    names = ["ripe\n(hue)", "rolls\n(shape)", "portion\n(size)"]
    xs = np.arange(3)
    for j, (m_, c, lab) in enumerate([("local", t["s1"], "local"), ("global", t["s2"], "global")]):
        A = np.array([run["snaps"]["pre_shift"]["acc_novel"][:3] for run in res[m_]])
        ax.bar(xs + (j - 0.5) * 0.36, A.mean(0), width=0.34, color=c, zorder=3,
               yerr=A.std(0) / np.sqrt(len(A)), error_kw=dict(ecolor=t["ink2"], lw=1, capsize=2.5))
        ax.text((j - 0.5) * 0.36, 0.05, lab, rotation=90, ha="center", va="bottom",
                fontsize=7.6, color=t["surface"], fontweight="bold", zorder=4)
    ax.axhline(0.5, color=t["axis"], lw=0.9, ls=(0, (3, 3)), zorder=2)
    ax.text(-0.52, 0.515, "chance", fontsize=7, color=t["muted"], ha="left", zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy on a combination never trained")
    ax.set_title("b.  Green, elongated, large — never seen", loc="left")
    tidy(ax, t)

    # (c, d) load-bearing matrices, one representative learner each
    tnames = ["ripe", "rolls", "portion", "edible"]
    for ax, m_, title in [(axes[1, 0], "local", "c.  Local: each jar carries one target"),
                          (axes[1, 1], "global", "d.  Global: each cell carries several")]:
        run = res[m_][0]["snaps"]["pre_shift"]
        LB = np.array(run["LB"])
        LBn = LB / (LB.max(1, keepdims=True) + 1e-12)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "seq", [t["surface"], t["s1"] if m_ == "local" else t["s2"]])
        ax.imshow(LBn, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(tnames)
        sel = run["selectivity"]; ch = run["top_channel"]
        chname = ["hue", "shape", "size"]
        ax.set_yticks(range(len(LB)))
        ax.set_yticklabels([f"jar {i+1} · {chname[c]} {s:.2f}" for i, (c, s) in enumerate(zip(ch, sel))],
                           fontsize=7.2)
        for i in range(LBn.shape[0]):
            for j in range(4):
                if LBn[i, j] > 0.35:
                    ax.text(j, i, f"{LBn[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                            color=t["surface"] if LBn[i, j] > 0.6 else t["ink"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("target that suffers when the jar is removed (learning frozen)")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[1, 0].text(0.0, -0.30, "row labels: the channel each jar is tuned to, and how exclusively (1 = one channel only)",
                    transform=axes[1, 0].transAxes, fontsize=6.8, color=t["muted"])

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"fig4_overlap_{mode}.png", dpi=170)
    plt.close(fig)

if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for mode in ["light", "dark"]:
        if "1" in which:
            fig1(mode)
        if "2" in which:
            fig2(mode)
        if "3" in which:
            fig3(mode)
        if "4" in which:
            fig4(mode)
    print("figures written")
