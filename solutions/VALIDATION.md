# Validation Report — Reference Solutions vs. Real NHANES

Instructor-only. Confirms each milestone's Satisfactory criteria are **achievable**
and that the intended pedagogical **payoffs land** when the notebooks are solved
end-to-end against the deployed dataset (`data/nhanes_2021_2022.csv`, N = 5,102
adults 18–79). Reference implementations: `solutions/_reference.py`. Where the
**synthetic fallback** would behave differently, it is noted.

> Jump to the **summary table** and the **design flags for Greg** at the end.

---

## Project 1 — regression, three lenses

Conditioning note: all P1 solutions z-score `age` before the polynomial expansion
(a raw `age^9` column is ~1e17 and makes `XᵀX` singular). A student who skips this
hits the singularity — a legitimate teachable discovery, not a trick.

### 1.1 Least squares — **Satisfactory: YES · payoff: PARTIAL**

- Fixture self-checks **PASS** (`OLS`, `ridge`) — the plumbing oracle is intact.
- **K-fold CV (5-fold, seed 521), val RMSE by order:**

  | order | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|
  | train | 15.933 | 15.932 | 15.921 | 15.918 | 15.913 | 15.912 | 15.910 | 15.910 | 15.909 |
  | **val** | 15.943 | 15.944 | **15.935** | 15.940 | 15.938 | 15.945 | 15.946 | 15.949 | 15.951 |

  **CV winner: order 3**, but only by ~0.01 RMSE over order 1 — *within noise* on a
  ~16 mmHg scale. age→SBP is a weak, near-linear signal (order-1 R² = 0.142,
  corr = 0.377).
- **Payoff (bias–variance) — PARTIAL.** The **overfitting** arm is clean: order 9
  has the lowest train error and the *worst* val error, and CV correctly refuses to
  reward the flexibility; ridge visibly damps the order-9 wiggle. The
  **underfitting** arm is **muted** — orders 1–3 are statistically tied because the
  true signal is essentially linear.
- **Prose check — NOT overclaiming.** The 1.1 scenario/play-artifact frame the sweep
  as *discovery* ("which combinations underfit, which overfit"), never asserting a
  winning order, so order 1 ≈ order 3 does not contradict the text. ✔️
- **Synthetic fallback differs:** the synthetic generator deliberately keeps
  age→sbp *mostly linear* too (per its rewrite), so the muted underfit arm is the
  same there. (The *older* synthetic, tuned for order-3 dominance, would have shown
  a sharper elbow — but that dataset is retired.)
- **Student effort:** moderate. ~60–80 lines (design matrix, normal-equations OLS,
  from-scratch K-fold + sweep, ridge, 3×3 grid). CV-from-scratch is the real work.

### 1.2 Maximum likelihood — **Satisfactory: YES · payoff: PARTIAL (prose flag)**

- **MLE = OLS confirmed** numerically: `max|w_mle − w_ols| ≈ 0` (`expect_close` PASS).
- `σ̂² ≈ 253.6` → `σ̂ ≈ 15.9` mmHg; fixture σ² check PASS.
- **Predictive band — correct but flat.** The MLE band is constant-width by design
  (homoscedastic σ̂). It does **not** widen anywhere; that is the intended limitation
  motivating 1.3, and the play-artifact hint already says so.
- **⚠ FLAG (prose vs. deployed data).** The play-artifact prose asserts "the age
  distribution is sparse at the upper end." On real NHANES that is **false**: density
  *peaks* at 50–70, and 70–79 (~83/yr) is comparable to 18–30 (~62/yr). The text was
  written for the **young-skewed synthetic fallback**, where it *is* true. On NHANES,
  the 1.3 edge-widening is a **leverage** effect at *both* ends, not a sparse-tail
  effect. Recommend reframing the 1.2 prose from "sparse upper tail" to
  "high-leverage extremes" (Greg's call — see flags). Does not block any criterion.
- **Student effort:** low–moderate. ~30 lines; the insight (MLE≡OLS) is conceptual,
  the code is short.

### 1.3 Bayesian — **Satisfactory: YES · payoff: YES**

- Posterior implemented from the precision form
  `A = XᵀX/σ² + I/τ²`, `m_N = S_N Xᵀy / σ²`.
- **Posterior-predictive band widens at the age-range extremes** (parameter-SD term
  `φᵀS_Nφ`), unlike 1.2's flat band — the headline 1.3 payoff. On NHANES it widens
  at **both** ends (leverage), consistent with the 1.2 flag above.
- **Collapse toward the MLE — confirmed both ways:** ‖m_N − w_MLE‖ falls
  108 → 38 → 0.85 as τ² grows 1e-2 → 1e0 → 1e4 (prior weakens); mean parameter SD
  shrinks monotonically as N grows 5% → 100%. ✔️
- **Student effort:** moderate–high. ~50–70 lines; deriving the posterior precision
  and the predictive variance is the conceptual load.

### 1.4 Synthesis — **Satisfactory: YES · payoff: YES**

- Three-lens comparison panel renders coherently: shared mean curve; MLE flat
  ±2σ̂ band; Bayesian predictive band that flares at the extremes. `max|OLS−MLE|≈0`,
  `max|MLE−posterior mean|` small (large N, weak prior).
- The argued compare-and-contrast (assumptions / outputs / relationships / choice)
  holds on the real numbers.
- **Student effort:** low code (~30 lines, integration only) + the ~500-word argument.

---

## Project 2 · Gate A — supervised

Split: 25% test, seed 521 (≈3,826 train / 1,276 test). Features = the six non-BP
columns; label = `hypertension(ds)` (prevalence 0.388). **Leakage check passed**: no
`sbp`/`dbp` reaches the feature matrix, and performance is far from perfect — so no
leakage slipped in.

### 2.1 Approximate inference — **Satisfactory: YES · payoff: YES**

- **Gradient check `expect_gradient` → PASS** (analytic grad matches the
  finite-difference of the student's own loss).
- **Newton–Raphson converges in 6 iterations**, grad norm 850 → 1e2 → 5 → 1e-2 →
  ~1e-13 (quadratic). ✔️
- **Metropolis mixes acceptably**: at proposal_sd = 0.02 the acceptance rate is
  **0.484** (inside the healthy 0.2–0.5 band); traces are fuzzy-caterpillar. The play
  artifact's scale sweep shows acceptance 0.85 (sd 0.005) → 0.01 (sd 0.1), so the
  tuning lesson lands.
- **Three-way agreement is clean** (the headline payoff): MAP, Laplace mean, and
  Metropolis mean coincide to `max |MAP − Metropolis mean| = 0.015`; Laplace vs
  Metropolis posterior SDs agree to **8.5%** (max relative). The N≈3,800 posterior is
  very nearly Gaussian, so Laplace ≈ Metropolis — exactly the point/interval/dist
  parallel to P1's three lenses.
- **Student effort:** high. ~120–160 lines (stable sigmoid + log-post/grad/Hessian,
  Newton loop, Laplace, a correct Metropolis with burn-in, three-way comparison).
  Conceptually the hardest milestone in the course.

### 2.2 Classification — **Satisfactory: YES · payoff: YES (with an honest caveat)**

- From-scratch **Pegasos SVM trains and separates above the no-skill floor**: at
  λ=1e-4 (C≈2.6), test P/R/F1 = 0.48 / 0.34 / 0.40, margin width ≈ 4.0, ≈3,059/3,826
  support vectors (heavy class overlap ⇒ wide soft margin, as expected).
- **Probabilistic vs margin is a MEANINGFUL difference**, not two identical
  boundaries: logistic (thr 0.5) F1 ≈ 0.42 vs SVM F1 ≈ 0.40, but they **disagree on
  ~20% of test patients** (agreement 0.80). The play-artifact threshold sweep moves
  logistic recall 0.36 → 0.83 as the cutoff drops 0.5 → 0.3; the C sweep narrows the
  SVM margin as C grows.
- **⚠ Performance caveat (report, not a defect): both classifiers are weak.**
  Logistic **AUC = 0.645**; accuracy at the default cutoff ≈ 0.62 vs a **majority
  baseline of 0.611**. This is *real but modest* signal — **not** near-random (AUC
  clearly > 0.5, F1 up to 0.60 at a tuned threshold) and **not** trivially perfect
  (no leakage). The cause is structural and *by design*: the label is defined from BP,
  BP is excluded to avoid leakage, and the six non-BP features correlate only weakly
  with it (max point-biserial: age 0.24, waist 0.19). The notebooks frame this
  honestly (accuracy-is-misleading note, threshold sweep), so the milestone is
  pedagogically sound — but Greg should know the supervised "win" is modest. See flags.
- **Student effort:** moderate–high. ~80–110 lines (Pegasos with bias, metrics from
  scratch, dual write-up, comparison, PR + C sweeps).

---

## Project 2 · Gate B — unsupervised

Clustering/PCA on the **standardized full cohort** (no train/test split — describing
structure, not predicting). Label held out of clustering; used only for the reveal.

### 2.3 Clustering — **Satisfactory: YES · payoff: PARTIAL (the key Gate B finding)**

- k-means converges with restarts; restart spread is small (smooth, low-structure
  space — local minima are not a real threat here).
- **K selection defensible:** silhouette **peaks at K = 2** (~0.23); the elbow is
  gentle. K = 2 gives the cleanest two-subgroup split.
- **⚠⚠ THE KEY CHECK — the cluster-vs-label reveal lands only PARTIALLY.** At K = 2:

  | cluster | n | HTN fraction |
  |---|---|---|
  | metabolic (high BMI/waist/HbA1c, low HDL, older) | 2,154 | **0.484** |
  | lean (mirror) | 2,948 | **0.319** |

  - Between-cluster **relative risk ≈ 1.52** — a *real, clinically sensible*
    enrichment.
  - **But cluster purity = 0.612, identical to the 0.612 majority baseline** — knowing
    a patient's cluster gives essentially **zero** improvement on a hypertension guess.
  - **Diagnosis (not a bug):** the label is BP-defined, BP is *excluded* for
    leakage-safety, and the six non-BP features correlate weakly with it (age 0.24,
    waist 0.19, …). k-means finds the dominant *geometric* axis (body composition +
    age), which overlaps with but does not equal the *clinical* BP stratum.
  - **Recommended framing (prose, not a code change): "partial recovery through the
    adiposity + age channel."** Make the geometric-vs-clinical gap the teaching point.
    The student notebook §4 already invites this honest reading, so **no notebook
    change is needed** — but instructors must not expect or reward a clean recovery.
    Optional richer variant: cluster on all eight columns (BP included) as an
    explicitly-labelled *leaky contrast* that motivates why P2 excludes BP.
- **Student effort:** moderate–high. ~80–120 lines (k-means + restarts, from-scratch
  silhouette — note O(n²), subsample for ~5k rows — cross-tab).

### 2.4 Dimensionality reduction — **Satisfactory: YES · payoff: YES**

- PCA via SVD; **variance explained interpretable**: PC1 37.3%, PC2 21.8% (59.1% in
  2-D); **four** PCs reach 88.9% — the cohort is only mildly low-dimensional.
- **Biplot shows real structure**: **PC1 = adiposity/metabolic** (bmi 0.59, waist 0.62,
  hba1c +, hdl −), **PC2 = age/lipid** (age 0.60, chol 0.57). Echoes the P1 metabolic
  block.
- **Clusters-in-PC-space coherent**: the K = 2 split lies cleanly along PC1 (k-means
  and PC1 read the same adiposity axis); the hypertension overlay is broadly mixed with
  a slight high-PC1/PC2 shift — the geometric restatement of 2.3's partial reveal.
- **Student effort:** moderate. ~60–90 lines (center, SVD/eig, scree, biplot arrows,
  cluster overlay).

### 2.5 Synthesis — **Satisfactory: YES · payoff: YES**

- The three-panel arc (P1 trend → Gate A P(HTN)-by-class → Gate B PCA-by-label) renders
  coherently and the narrative holds on the **real** numbers — including, crucially, an
  **honest negative result**: the supervised signal is weak and the unsupervised
  structure is metabolic, not hypertensive. The capstone reads as a clean
  supervised-vs-unsupervised argument that *earns* its conclusion rather than asserting
  a tidy recovery.
- **Student effort:** low–moderate code (~50 lines, integration) + the ~600-word
  argument.

---

## Summary table

Every milestone is **Satisfactory-achievable** against the real NHANES data; all
nine notebooks render exit 0. "Payoff" = does the intended pedagogical effect land.

| Notebook | Satisfactory | Payoff | Student effort | Flag |
|---|---|---|---|---|
| 1.1 Least squares | ✅ | **Partial** — overfit arm clean, underfit muted | moderate, ~60–80 ln | F3 |
| 1.2 Max likelihood | ✅ | **Partial** — band flat by design; prose flag | low–mod, ~30 ln | **F2** |
| 1.3 Bayesian | ✅ | ✅ band widens; collapses to MLE | mod–high, ~50–70 ln | — |
| 1.4 Synthesis | ✅ | ✅ panel coherent | low, ~30 ln + prose | — |
| 2.1 Approx inference | ✅ | ✅ Newton fast, sampler mixes, 3-way agrees | **high, ~120–160 ln** | — |
| 2.2 Classification | ✅ | ✅ meaningful prob-vs-margin diff | mod–high, ~80–110 ln | F1 |
| 2.3 Clustering | ✅ | **Partial** — RR 1.5 real, purity ≈ baseline | mod–high, ~80–120 ln | **F1** |
| 2.4 Dim. reduction | ✅ | ✅ PCA interpretable, clusters coherent | moderate, ~60–90 ln | — |
| 2.5 Synthesis | ✅ | ✅ honest negative result | low–mod, ~50 ln + prose | — |

Plumbing verified: `checks.LINEAR_FIXTURE` (OLS/ridge/σ²) PASS; `checks.expect_gradient`
PASS on the real 2.1 loss. No plumbing function is broken. No student notebook,
Satisfactory criterion, `info521/`, or `data.py` was modified by this task.

---

## Design flags for Greg (decisions, not blockers)

Nothing here blocks deployment; each is a framing/efficacy call. **None was
silently fixed** in the student notebooks.

**F1 — The supervised + unsupervised "win" on the BP-defined label is modest, by
design.** Logistic AUC ≈ 0.645 (≈ majority-baseline accuracy); k-means cluster purity
(0.612) equals the base rate (0.612), though a real RR ≈ 1.5 enrichment exists. Cause:
BP is excluded from the features to keep the supervised task leakage-free, and the six
non-BP features correlate only weakly with a BP-defined label. This is *honest and
pedagogically fine* — the notebooks frame it well (accuracy-is-misleading note,
threshold sweep, "geometric ≠ clinical structure"). **Decision:** (a) accept the modest
signal as-is (recommended — the honesty is the lesson), or (b) add the optional
**8-column "leaky contrast"** in 2.3 (cluster with BP included) to show a strong reveal
and explicitly motivate why P2 excludes BP. No criterion needs changing either way.

**F2 — 1.2 play-artifact prose mismatches the deployed data.** It says "the age
distribution is sparse at the upper end." True for the *young-skewed synthetic
fallback*; **false for real NHANES** (density peaks at 50–70; 70–79 ≈ 18–30). The 1.3
edge-widening is a *leverage-at-both-extremes* effect, not a sparse-upper-tail one.
**Decision:** reword the 1.2 student prose from "sparse upper tail" → "high-leverage
extremes (both ends)." A one-line student-notebook edit — flagged here, not made.

**F3 — 1.1 bias–variance lesson is half-muted on the near-linear real signal.** CV
picks order 3 but within ~0.01 RMSE of order 1; the underfit arm barely shows (R² ≈
0.14). The overfit arm (order 9 worst val) is clean and the prose frames the sweep as
*discovery*, so it does not overclaim. **Decision:** accept as-is (recommended), or add
a sentence to the 1.1 play artifact noting that a near-linear signal compresses the
underfit arm.

**Synthetic-fallback note (per task):** the deployed-vs-fallback behavior differs in
exactly one pedagogically relevant place — **age distribution shape** (F2/F3). The
synthetic is young-skewed (sparse upper tail → the 1.2/1.3 sparsity story is visually
cleaner) and deliberately near-linear age→sbp (so 1.1's underfit arm is muted on
*both*). All contract-level behavior (columns, target, `hypertension()`) is identical,
so every Satisfactory criterion is achievable on either dataset.

---

## Distribution exclusion (reminder)

`solutions/`, `instructor/`, and the **answer-key portions of `checkpoints/*.md`**
must be excluded from any student-facing template. See `solutions/README.md`.
