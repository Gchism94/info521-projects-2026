# Dataset Curation

## Deployed default: NHANES 2021–2022

The course ships wired to a curated **real** dataset: **NHANES 2021–2022** (CDC,
cycle suffix `_L`), the most recent regular cycle. `data/build_nhanes.py`
downloads the six source XPT files (DEMO, BPXO, BMX, TCHOL, HDL, GHB), merges them
on `SEQN`, derives oscillometric blood pressure, and writes
`data/nhanes_2021_2022.csv` (5,102 adults aged 18–79, complete cases). This is
what `info521.data.load_clinical()` loads by default; the **synthetic** generator
below stays as a reproducible **offline fallback**.

- **Target** `sbp` (systolic BP, mmHg) — the continuous Project 1 outcome.
- **Reserved** `dbp` (diastolic BP) — used only to build the Project 2
  hypertension label (ACC/AHA: SBP ≥ 130 or DBP ≥ 80) via
  `info521.data.hypertension(ds)`; excluded from the feature matrix so the label
  can't leak into its own predictors.
- **Features** `age, bmi, waist, chol, hdl, hba1c`.
- Blood pressure is **oscillometric** (`BPXOSY1..3` / `BPXODI1..3`); auscultatory
  BP was discontinued after 2017–2018. Survey weights are deliberately ignored
  (algorithms course, not population inference). Measured-BP-only hypertension
  prevalence is ≈0.39 for ages 18–79 (the ACC/AHA "on medication" arm, `BPQ_L`,
  is omitted by default — add it if you want the ≈0.46 population figure).
- Rebuild with `python data/build_nhanes.py` (downloads cached in the gitignored
  `data/raw/`).

## Curation rationale (why a curated real dataset)

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
  realistic and flexible; most wrangling. **← the deployed default** (NHANES
  2021–2022; see the top of this document).

## The synthetic offline fallback

`data/clinical_reference.csv` is a **synthetic** dataset (generator:
`data/generate_reference_data.py`, seed 521) that stays as the **offline fallback**
so the pipeline runs out of the box with no network access and is fully
reproducible. Load it explicitly with
`info521.data.load_clinical(path="data/clinical_reference.csv")`.
**Do not present it to students as real data.** Its synthetic `risk` score is
tuned so:

- age → (synthetic) risk is gently nonlinear (order 1 underfits, ~3 fits, 9 overfits);
- the upper age tail is sparse (predictive bands widen there in 1.2);
- the target splits ~50/50 at its median (clean median binarization);
- features form correlated blocks (metabolic, lipid, BP) for PCA/clustering.

> **Fallback caveat.** This synthetic CSV predates the SBP reframe: its columns are
> `age, bmi, sbp, glucose, hba1c, chol, hdl, risk` (a synthetic `risk` target and
> an `sbp` *feature*, with no `dbp`). Under the SBP-targeted loader it still loads
> and Project 1 runs (the loader uses its `sbp` column as the target), but
> `hypertension(ds)` raises — the synthetic has no `dbp`. For a clean offline
> drop-in that mirrors the NHANES contract (`sbp` target + reserved `dbp`),
> regenerate the synthetic with those columns. For full parity, deploy NHANES.

### Swapping in another dataset

Point `info521.data.load_clinical(path=...)` at a CSV that honors the contract:
header row, a continuous target column named `sbp`; an optional `dbp` column
(reserved — excluded from features, used to build the hypertension label); and an
`age` column (or set a different `primary` predictor in the loader). Nothing else
in the notebooks needs to change.
