# INFO 521 — Projects (Spring 2026)

Two additive projects that replace the former midterms and final. The exams'
applied, integrative, and synthesis outcomes move into the projects; the
unaided-derivation and recall outcomes move into short, retakeable in-class
**checkpoints**. Homework remains the practice layer.

## The arc

**One clinical dataset carries the whole semester.** A single continuous
prediction problem is examined through progressively richer lenses, then extended
from prediction into discovery.

### Project 1 — One Problem, Three Lenses (Modules 1–3)

Same dataset, same single predictor (age → risk), three accreting lenses:

| Milestone | Lens | Gated by |
|---|---|---|
| 1.1 | Least squares (design matrix, CV, L2) | Checkpoint 1 — normal equations |
| 1.2 | Maximum likelihood (Gaussian noise, predictive variance) | Checkpoint 2 — MLE for **w** |
| 1.3 | Bayesian (conjugate posterior, credible intervals) | Checkpoint 3 — posterior of a conjugate pair |
| 1.4 | Synthesis — all three lenses, side by side | (peer review) |

### Project 2 — From Inference to Discovery (Modules 4–7) *(scaffold to come)*

Binarize the outcome → conjugacy breaks → approximate inference (Newton–Raphson,
Laplace, MCMC) → classification (SVM) → clustering (k-means) → dimensionality
reduction (PCA), now across the **full** feature set. See `project-2/README.md`.

## Assessment model

Specs grading. Each milestone is **Satisfactory / Not-Yet** with one revise
cycle; each checkpoint is pass / retake. A milestone reaches Satisfactory only
once its checkpoint is passed. The two projects are **co-equal** summative
pillars — Project 2 is not weighted more despite being cumulative, because specs
mastery already requires Project 1's foundations. See `GRADING.md`.

## Ground rules for students

- Implement every estimator **from scratch** in NumPy/SciPy. No scikit-learn or
  other off-the-shelf solvers.
- The `info521` library is **plumbing only** — data, plotting, self-checks. It
  contains no fitting routines by design.

## Repo layout

```
common/info521/    plumbing library (data, plotting, checks) — no estimators
data/              reference dataset + generator (synthetic; swap for real)
project-1/         four .qmd milestones
project-2/         scaffold (README only for now)
checkpoints/       in-class derivation slips + answer keys
instructor/        curation, peer-review, and specs notes (your decisions)
```

## Quick start

```bash
# Python 3.11 environment with numpy 2.1.x pinned
conda env create -f environment.yml      # or use your own venv
conda activate info521-projects
pip install -e .                         # installs the info521 plumbing package
python data/generate_reference_data.py   # (re)creates the reference dataset
quarto render project-1/1.1-least-squares.qmd
```

## Instructor decisions baked in

Dataset curation, peer-review cadence, checkpoint design, and play-artifact
parameters are recorded in `instructor/`. Adjust there.
