# Jars toy model — reproduction package

Code, seeds and result files for every number in the report *Simulating the Jars* and in the reply to the reviewer.

Pure Python 3.11 + NumPy (tested with NumPy 2.4). Matplotlib is needed only for the figures. No GPU and no learning framework.

```
pip install -r requirements.txt
```

## Files

| File | What it is |
|---|---|
| `jars_model.py` | The learner: fast settling with structure fixed, slow local learning, the gate, unit birth and death, and the load-bearing measurement. |
| `exp1_contrast.py` | Demonstration 1: contrast-shaped boundaries, plus a sweep over the price of precision. |
| `exp1_robustness.py` | Demonstration 1 on eight materially different learners (reviewer question 3). |
| `exp2_timescales.py` | Demonstration 2: load-bearing vs plasticity, relation migration, and the Section 11 test. |
| `exp3_selection.py` | Demonstration 3: split, merge, the λ sweep and the Pareto comparison. |
| `exp4_overlap.py` | Demonstration 4: overlapping jars, local vs global competition (reviewer question 12). |
| `make_figures.py` | Reads the result files and writes light and dark versions of figures 1–4. |
| `results_exp*.json` | The result files the report's figures and tables are drawn from. |

## Reproducing the runs

Each script has a `--quick` flag for a short pass. The full runs below are what the report uses. Times are on one CPU core.

| Command | Seeds × conditions × trials | Time | Writes |
|---|---|---|---|
| `python exp1_contrast.py` | 12 × 6 contrast distances × 9,000, plus robustness 6 × 6 × 9,000 at three λ_b | ~7 min | `results_exp1.json` |
| `python exp1_robustness.py` | 8 × 6 × 7,000 for each of 8 learners | ~12 min | `results_exp1R_all.json` |
| `python exp2_timescales.py` | 10 × 16,000, plus 10 × 3 plasticity levels | ~25 min | `results_exp2.json` |
| `python exp3_selection.py` | 10 × 21,000, plus a λ sweep of 7 × 10 × 14,000 | ~20 min | `results_exp3.json` |
| `python exp4_overlap.py` | 8 × 32,000 for each of 3 learners | ~3 min | `results_exp4.json` |
| `python make_figures.py` | — | ~20 s | `fig{1..4}_*_{light,dark}.png` |

`exp1_robustness.py` accepts learner names as arguments, so the learners can be run in parallel, e.g. `python exp1_robustness.py alcove rbf-logistic`. Each invocation writes `results_exp1R_<names>.json`. The shipped `results_exp1R.json` merges the two parallel full runs used in the reply.

## Seeds

All randomness is seeded in code, so results are deterministic given NumPy's `default_rng`. For replicate `s`:

| Stream | Seed |
|---|---|
| Learner initialisation and exploration | `s` |
| Exp 1 environment (positive stream shared across conditions) | `10000 + s` |
| Exp 2 training stream | `50000 + s` |
| Exp 2 probe set | `90000 + s` |
| Exp 2 Section 11 relearning stream | `50000 + (20000 + s)` |
| Exp 3 environment | `1000 + s` |
| Exp 3 λ-sweep gain test set | `9000 + s` |
| Exp 3 split-detection probes | `4242 + s` |
| Exp 4 environment | `1000 + s` |
| Exp 4 measurement probes | `77000 + s` |

A spot check that the architecture switches added later do not change the defaults: `exp1_contrast.run_condition(0, 2.5)` reproduces the stored values in `results_exp1.json` bit-for-bit.

## Varying the architecture rather than the parameters

* **Kernel shape:** `Params(kernel="laplace")` uses an L1 kernel instead of the Gaussian.
* **Capacity penalty:** `Params(cost_form="quadratic" | "log")` uses \(K=\sum\beta^2\) or \(K=\sum\log\beta\) instead of \(K=\sum\beta\).
* **Precision locality:** `SidePrecision` in `exp1_robustness.py` holds a separate precision for each side of each channel.
* **Readout:** `SideFixedReadout` additionally makes each unit's vote immune to error-driven erosion.
* **Category representation:** `Alcove` (Kruschke 1992, covering-map version) and `RBFLogistic` in `exp1_robustness.py`.
* **Competition regime:** `Pool(mode="local" | "global")` in `exp4_overlap.py`. `lam_b=0.0` removes the capacity cost.

Defaults reproduce the report exactly.

## Learning-rule locality

* **Experiments 1–3:** updates use only quantities local to each unit (its activation, its own parameters and a broadcast scalar error or reward). Nothing is backpropagated through the settling loop.
* **Experiment 4:** each target's error is passed back along each unit's own outgoing readout weights to update that unit's centre and precision. This is a one-layer delta rule, as in ALCOVE. There is still no backpropagation through any settling loop, but it is less local than experiments 1–3.

## Known limits

* **Scale:** three to sixteen units, two to four input channels, a handful of targets.
* **Channel basis:** assumed, not learned.
* **Overlap (experiment 4):** the easiest case — features independent and aligned with channels.
* **Migration (experiment 2):** the gate learns from action outcomes while load-bearing is measured on classification. These are different quantities but not independent ones.

## Second review round (spine v3): experiments 7–11

These use `pool.py`, a generalised version of the experiment-4 learner (`exp4_overlap.py` is untouched). Each script states its predictions in its docstring; anything added after a pilot is declared there.

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp7_glue.py` | Q17: same classifier, different selection histories | `results_exp7.json` | 3 min |
| `exp8_redundancy.py` | Q5: redundant realisation; single vs joint vs Shapley load | `results_exp8.json` | 4 min |
| `exp9_oblique.py` | Q19: overlap with oblique feature dimensions | `results_exp9.json` | 15 min |
| `exp10_compound.py` | Q20: when does a compound get its own jar? | `results_exp10.json` | 12 min |
| `exp11_operators.py` | Q12–13: asymmetric relation; `Under(PET, X)` | `results_exp11.json` | 10 s |
| `exp11b_diagnostic.py` | post hoc: does the PET SHARK result depend on the role face's functional form? | `results_exp11b.json` | 20 s |
| `make_figures2.py` | figures 5–10 (light and dark) | `figs/` | 10 s |

Every script accepts `--quick` for a short pilot, apart from `exp11b` and `make_figures2`. Seeds: exp7 environment `1000 + s`; exp8 environment `2000 + s`, test `8000 + s`; exp9 environment `1000 + s`, test `77000 + s`; exp10 environment `1000 + s`, test `66000 + s`; exp11 part 1 `s`, part 2 `100 + s`.

**Known limits of the new work**

* **exp7:** the candidate "worldly conditions" are input channels, so it says nothing about distal content.
* **exp11, the entity codes:** entities are attribute vectors standing in for jar outputs.
* **exp11, argument-to-slot routing:** this is supplied by the programmer.
* **exp11, PET's role face:** it is fitted by ridge regression, not by the jars learner.
* **exp11, PET SHARK:** the result depends on the role face including a size × aquatic term (see `exp11b`).

## Third review round (spine v4): experiments 12–16

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp12_contrast_online.py` | Q2: is contrast history internal to the vehicle when the contrast is absent? jar vs exemplar vs generative | `results_exp12.json` | 1 min |
| `exp13_drift.py` | Q3: does the realiser drift at stable accuracy? | `results_exp13.json` | 20 min |
| `exp14_role_face.py` | Q4: can the payoff dynamics acquire PET's role face? | `results_exp14.json` | 1 min |
| `exp15_own_edge.py` | Q5: why the own-edge compound fails (ten conditions; the last two via `--only oracle-noprune,oracle-anchored`) | `results_exp15.json` | 25 min |
| `exp16_unity.py` | Q1: far-edge coupling within vs across jars | `results_exp16.json` | 1 min |
| `make_figures3.py` | figures 11–15 | `figs/` | 10 s |

Seeds: exp12 streams `12000 + s`, probes `55000 + s`; exp13 environment `2000 + s`, noise `3000 + s`, test `8000 + s`; exp14 training `1400 + s`, evaluation `900 + s`; exp15 as exp10; exp16 streams `16000 + s`.

Changes made after pilots, and after one full run, are declared in each script's docstring.

## Fourth review round: experiments 17–18

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp17_module_persistence.py` | what persists through turnover, the jar or only the capacity? (Shapley-interaction coupling, modularity communities, chain of continuations; gradual vs wholesale replacement) | `results_exp17.json` | 20 min |
| `exp18_one_resource.py` | one resource vs linked resources for PET's two uses: coupling, cross-role transfer, upstream-input control | `results_exp18.json` | 1 min |
| `make_figures4.py` | figures 16–17 | `figs/` | 5 s |

Needs `networkx` (for community detection) in addition to `requirements.txt`. Seeds: exp17 as exp13 (environment `2000 + s`, test `8000 + s`); exp18 training `1800 + s`, revisions `2800 + s` / `3800 + s`, probes `7000 + s`.

## Fifth review round: experiment 19

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp19_community_recovery.py` | does the functional-community diagnostic recover organisation built in on purpose, and can it be fooled by mere interaction or by shared global machinery? Six hand-wired systems, exact coalition values over all 2ⁿ subsets, exact Shapley values and interaction indices, two readings of the unity criterion, three community algorithms, two threshold sweeps | `results_exp19.json` | 8 s |
| `make_figures5.py` | figures 18–20 | `figs/` | 5 s |

Unlike every other experiment here, exp19 does not learn: the systems are constructed so that the organisation is known in advance and the measure can be scored against it. `--quick` runs two seeds and 400 items. Seeds vary the 1,200-item evaluation sample (`19000 + s`) and nothing else. The write-up is `unity-criterion-experiment.md` / `Testing-the-Unity-Criterion.pdf`.

## Fifth round, follow-up: experiments 20–23

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp20_learned_membership.py` | does the membership diagnostic survive contact with learned systems? Stability of the recovered organisation across seeds, evaluation samples, the threshold ε, and sampled against exact attribution, on exp13's four conditions | `results_exp20.json` | 5 min |
| `exp21_generic_machinery.py` | does positive contribution absorb generic enabling machinery — a global gain, a flat always-on unit, a denoiser — into every concept? Breadth, discriminative content and content-neutral replaceability as candidate extra conditions | `results_exp21.json` | 15 min |
| `exp22_mediated_unity.py` | is pairwise interaction the only kind of unity? A capacity carried by a common classificatory mechanism against one whose components are combined only outside the system | `results_exp22.json` | 1 min |
| `exp23_lineage_shared.py` | can a surviving shared component fake a lineage? Three constructed histories, scored by the persistence criterion as it stands and with the specificity repair | `results_exp23.json` | 1 min |
| `make_figures6.py` | figures 21–24 | `figs/` | 5 s |

exp20 vectorises the coalition game over the frozen pool, which is exact in `local` mode because a unit's activation does not depend on which other units are alive. exp21 and exp23 record substantive post-pilot design changes in their docstrings, including exp23's rebuild from a learned history to a constructed one and the reason the learned version cannot work. The write-up is `unity-criterion-followup.md` / `Unity-Criterion-Follow-up.pdf`.

## Sixth round: experiments 24–25

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp24_shared_vs_generic.py` | can the membership criterion tell a genuinely shared discriminative resource from a rich generic modulator, without banning overlap? The two adversaries are built with the *same* activation function, so any measure taken from a component's own activity scores them identically | `results_exp24.json` | 1 s |
| `exp25_lineage_specificity.py` | does the lineage repair behave gradually, and can relational organisation carry a lineage when membership is fully shared? | `results_exp25.json` | 30 s |
| `make_figures7.py` | figures 25–26 | `figs/` | 3 s |

exp24 adds a third membership clause — engagement, measured as the capacity's solo application profile — and records the perturbation-based measure that failed before it. exp25 sweeps survivor specificity continuously and adds a case with zero membership differential and a large relational one. The write-up is `followup-experiments-24-25.md` / `Follow-up-Experiments-24-25.pdf`.

## Seventh round: experiment 26

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp26_conjunctive_member.py` | can classificatory engagement be stated without appeal to solo operation? A capacity realised by two components entering only through their product, so that each is constitutive while carrying no marginal information and producing no solo profile | `results_exp26.json` | 6 s |
| `make_figures8.py` | figure 27 | `figs/` | 2 s |

The answer is the Shapley value on the capacity's **discriminability** (the AUC of its application profile against its target) alongside the Shapley value on its accuracy. A constant offset and a pure gain cannot reorder cases, so they contribute to accuracy and nothing to discriminability. The clause admits conjunctive and shared members, excludes generic support, needs no new threshold, and makes the content clause redundant. Two measures that failed first are kept in the script and declared. The write-up is `engagement-stated-abstractly.md` / `Engagement-Stated-Abstractly.pdf`.

## Eighth round: experiment 27

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp27_denoiser_and_profile.py` | two adversaries for the discriminability criterion: a generic denoiser that improves ordering without any readout role, and a resource that carries within-category typicality while leaving the binary ranking untouched | `results_exp27.json` | 17 s |
| `make_figures9.py` | figure 28 | `figs/` | 2 s |

Both succeed. The denoiser earns the largest discriminability score in the system, so contribution to discriminations is not sufficient; the repair is that a member's contribution must be **routed**, measured as its discriminability score for alternative capacities built over the same resources whose readouts exclude it (0.000 for every genuine member, 0.071 for the denoiser). The shaper is invisible to binary AUC (−0.028) while carrying the category's internal structure (0.715), so AUC is the binary-task instrument rather than the definition. The write-up is `both-adversaries.md` / `Both-Adversaries.pdf`.

## Ninth round: experiment 28

| Script | Question | Output | Runtime (approx.) |
|---|---|---|---|
| `exp28_specific_modulator.py` | a counterexample to the routed criterion: a modulator that carries no signal and feeds no readout, but cleans one capacity's units alone — generic in mechanism, specific in scope | `results_exp28.json` | 18 s |

The counterexample fails: the sharpener's off-route influence is 0.010 against the global denoiser's 0.064, so the criterion admits it to the capacity it serves and excludes the denoiser from everything. It does establish that dropping contribution-to-success from membership was required rather than merely defensible — the sharpener's accuracy contribution is −0.003, so a criterion retaining that clause would exclude it. The write-up is `sign-off.md` / `Sign-Off.pdf`.

## Figures for the blog post

`make_figures_post.py` builds the set used in the post draft. It re-renders three figures with plain-English titles (the oblique-overlap, compound-jar and contrast-removed panels) and copies the rest out of `figs/` under descriptive names, writing everything to `figs/post/` in light and dark versions:

| file | from | shows |
|---|---|---|
| `life_history` | exp3 | jars follow the payoff, not the feature structure |
| `contrast_removed` | exp12 | what survives when the contrast is taken away |
| `subordinate_edge` | exp15 | ten conditions on carving "very red" inside RED |
| `unity_coupling` | exp16 | far-edge coupling within a jar and across jars |
| `damage_vs_repair` | exp2 | immediate damage against repair |
| `redundancy` | exp8 | single, joint and Shapley load |
| `same_jar_new_parts` | exp17 | gradual turnover against wholesale removal |
| `overlap_portability` | exp9 | overlap survives rotation, reuse does not |
| `compound_jars` | exp10 | AND by readout, XOR by recruitment |
| `pet_shark` | exp14 | consequences predicted for an animal never kept |
| `same_jar_different_content` | exp7 | identical jars, different worldly conditions |

Run order for a clean rebuild: the experiment scripts (each writes its own `results_*.json`), then `make_figures.py`, `make_figures2.py`, `make_figures3.py`, `make_figures4.py`, `make_figures5.py`, `make_figures6.py`, `make_figures7.py`, `make_figures8.py`, `make_figures9.py`, then `make_figures_post.py`.
