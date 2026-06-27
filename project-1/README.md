# Project 1 — One Problem, Three Lenses (Modules 1–3)

One clinical dataset, one prediction problem (age → cardiometabolic risk),
examined through three accreting lenses and then synthesized.

## Milestones

1. **1.1 Least Squares** — design matrix, normal equations, K-fold CV, L2.
   Play artifact: a 3×3 bias–variance sweep over polynomial orders {1, 3, 9} and
   λ ∈ {1e-3, 1e-1, 1e1}. Gated by Checkpoint 1.
2. **1.2 Maximum Likelihood** — Gaussian-noise generative model, MLE for **w**
   (shown numerically equal to the 1.1 OLS fit), $\hat\sigma^2$, predictive error
   bars. Play artifact: where the model is least sure, and why. Gated by
   Checkpoint 2.
3. **1.3 Bayesian** — Gaussian prior, conjugate posterior, posterior-predictive
   credible intervals. Play artifact: prior-strength and data-amount sweeps
   showing collapse toward the MLE. Gated by Checkpoint 3.
4. **1.4 Synthesis** — one panel, three lenses; argued compare-and-contrast;
   reproducible end-to-end. Structured peer review.

## Through-line

Project 1 stays **single-predictor polynomial** the whole way (model order is the
only complexity knob, mirroring the Olympic example). Expanding to the full
multivariate feature set is one of the things **Project 2 adds**.

## Satisfactory

Each notebook states its own criteria at the top. The project is complete when
1.1–1.4 are all Satisfactory and Checkpoints 1–3 are passed. See `../GRADING.md`.

## Bridge to Project 2

1.3 leans entirely on conjugacy (closed-form Gaussian posterior). Project 2 opens
by binarizing the risk outcome — conjugacy breaks, the posterior goes
intractable, and approximate inference becomes necessary. The same binary label
then drives classification; dropping it entirely motivates clustering and PCA.
