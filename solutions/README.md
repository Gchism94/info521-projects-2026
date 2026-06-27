# Solutions — INSTRUCTOR ONLY

> ⛔ **Do not distribute. Never include this directory in a student-facing
> repository or GitHub Classroom template.**

This directory holds the **reference solutions** to every Project 1 and Project 2
milestone, solved end-to-end against the deployed real NHANES dataset, plus the
validation report that confirms each milestone is achievable and that its
pedagogical payoff actually lands.

## Contents

- `_reference.py` — from-scratch NumPy/SciPy implementations of every estimator
  (design matrix, OLS, ridge, K-fold CV, MLE, Gaussian posterior, logistic
  log-post/grad/Hessian, Newton–Raphson, Laplace, Metropolis, Pegasos SVM, k-means,
  silhouette, PCA). The deliberate instructor-only counterpart to `info521` staying
  plumbing-only for students. **Do not import this from `info521`.**
- `project-1/1.1–1.4.qmd` — solved Project 1 notebooks (student structure, every
  TODO filled). Render with `quarto render`.
- `project-2/2.1–2.5.qmd` — solved Project 2 notebooks.
- `VALIDATION.md` — the validation report: per-milestone Satisfactory check, payoff
  landings, student-effort estimates, the real-NHANES numbers, and the design flags
  that need an instructor decision.

Rendered `*.html` is git-ignored; regenerate locally.

## ⚠ Distribution exclusion — required

When you publish a student-facing template (GitHub Classroom or otherwise), the
following instructor-only paths **must be excluded** (they contain solutions or
answer keys). The *mechanism* — a separate template repo, a sync filter, a
`.gitattributes export-ignore` rule, a release script — is your call; this README
only flags the requirement:

| Path | Why exclude |
|---|---|
| `solutions/` | full reference solutions + `_reference.py` + this report |
| `instructor/` | dataset-curation, peer-review, and build notes |
| `checkpoints/*.md` | **each file bundles the prompt with its answer key** — ship the prompts only, or strip the `## Answer key` sections, or withhold the files |

(The student-facing notebooks under `project-1/` and `project-2/`, `common/info521/`,
`data/`, `GRADING.md`, and the top-level `README.md` are intended for students.)
