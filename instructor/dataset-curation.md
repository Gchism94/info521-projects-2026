# Dataset Curation

## Recommendation

Deploy with a curated **real** dataset. Offer students a constrained choice (3–4
vetted options), not free-for-all (breaks the additive spine) and not a single
fixed set (kills ownership). An approved bring-your-own is a fine escape hatch.

### Vetting bar

A dataset qualifies only if it supports the **entire** arc:

1. A **continuous outcome** with a clinically sensible **single predictor** for
   the Project 1 polynomial story (ideally with mild nonlinearity so order
   matters).
2. The outcome **binarizes naturally** at a threshold, so the Project 2
   conjugacy-break (→ Bayesian logistic regression) is meaningful.
3. Enough **additional features** (≥ ~6) with some correlation structure, so PCA
   and k-means have something real to find.
4. Redistributable, no PII.

### Recommended options

- **Diabetes progression** (442 patients, 10 features, continuous one-year
  progression target). *Default.* Canonical, clean; binarizes at the median into
  high/low progression; 10 features feed Project 2. Available via
  `sklearn.datasets.load_diabetes` (use it **only** to obtain the raw arrays —
  loading data is not an estimator and does not violate the no-sklearn rule for
  student implementations) or vendor a CSV.
- **Age → systolic BP / hypertension.** Ties directly to the Unit 1
  age→systolic-BP interactive already in your materials; binarizes to
  hypertension yes/no. Strong clinical narrative.
- **NHANES subset** (BMI / BP / glucose continuous, many features). Most
  realistic and flexible; most wrangling.

## What ships in this starter

`data/clinical_reference.csv` is a **synthetic** dataset (generator:
`data/generate_reference_data.py`, seed 521) so the pipeline runs out of the box
and is fully reproducible. **Do not present it to students as real data.** It is
tuned so:

- age → risk is gently nonlinear (order 1 underfits, ~3 fits, 9 overfits);
- the upper age tail is sparse (predictive bands widen there in 1.2);
- the target splits ~50/50 at its median (clean Project 2 binarization);
- features form correlated blocks (metabolic, lipid, BP) for PCA/clustering.

### Swapping in a real dataset

Point `info521.data.load_clinical(path=...)` at a CSV that honors the contract:
header row, a continuous target column named `risk` (rename yours, or adjust the
loader), and an `age` column (or set a different `primary` predictor in the
loader). Nothing else in the notebooks needs to change.
