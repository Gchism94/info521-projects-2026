# Validation Report — Reference Solutions vs. Real NHANES

Instructor-only. Confirms each milestone's Satisfactory criteria are **achievable**
and that the intended pedagogical **payoffs land** when the notebooks are solved
end-to-end against the deployed dataset (`data/nhanes_2021_2022.csv`, N = 5,102
adults 18–79). Reference implementations: `solutions/_reference.py`. Where the
**synthetic fallback** would behave differently, it is noted.

> The top-level summary table and cross-cutting design flags are at the end
> (added in Stage 4).

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
