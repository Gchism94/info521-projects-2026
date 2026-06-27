# Project 2 — From Inference to Discovery (Modules 4–7)

> **Scaffold only.** This README fixes the intended shape so Project 1 has a
> concrete target to hand off to. The milestone notebooks are written in the next
> build pass, once Project 1 is reviewed.

Project 2 continues the *same* clinical dataset from Project 1 and extends it from
prediction into discovery. Two things expand at once: the **method** (regression
→ classification → unsupervised) and the **feature space** (single predictor →
the full multivariate set).

## The additive bridge

Project 1 ended on a closed-form conjugate Gaussian posterior. Project 2 breaks
it deliberately:

1. **Approximate Bayesian inference (M4).** Binarize the risk outcome (high vs.
   low). A Bayesian **logistic regression** model has **no conjugate posterior** —
   so you implement, from scratch: Newton–Raphson (MAP), the Laplace
   approximation, and a Metropolis sampler. Compare what each buys you.
2. **Classification (M5).** The same binary label is now an SVM target. Linear
   hard-margin → soft-margin → kernelized. Evaluate with precision / recall / F1.
   Contrast the discriminative SVM with the probabilistic classifier from step 1.
3. **Clustering (M6).** Drop the labels. Implement k-means from scratch on the
   full feature set; address local minima (restarts) and model selection
   (choosing K). Do the discovered clusters relate to the risk label?
4. **Dimensionality reduction (M7).** PCA from scratch on the features; visualize
   the cohort in 2-D; feed components into the clustering and discuss.

## Suggested bundle (specs)

- **Gate A — Supervised:** approximate inference (M4) + classification (M5)
  Satisfactory, plus their checkpoints (e.g. Newton–Raphson update; precision/
  recall/F1 definitions).
- **Gate B — Unsupervised:** clustering (M6) + dimensionality reduction (M7)
  Satisfactory, plus checkpoint (e.g. k-means objective / choosing K).
- **Final synthesis:** the whole-semester arc in one narrative — regression →
  classification → structure — on one cohort. Structured peer review.

## Play, extended

- A held-out **prediction round** on the binary outcome (compare SVM vs. Bayesian
  logistic regression on the same split).
- A **cluster-vs-label reveal**: do unsupervised subgroups recover clinical risk
  strata you never showed the algorithm?
- PCA **biplot exploration**: which features drive the principal axes?

## Carry-over outcomes

The same plumbing applies: `info521.data.binarize(...)` provides the M4/M5 label;
`load_clinical()` already returns the full feature matrix for M6/M7. No estimators
will be added to the library — k-means, PCA, SVM, and the samplers are all
student-implemented.
