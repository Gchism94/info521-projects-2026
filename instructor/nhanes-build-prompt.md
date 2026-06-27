# Claude Code task prompt — wire NHANES as the deployed default

Disposable per-task prompt. Paste into Claude Code running locally on the Mac.
Targets NHANES **2021–2022** (suffix `_L`), the most recent regular cycle.

---

```text
TASK: Wire NHANES 2021–2022 in as the deployed default dataset for
info521-projects-2026, replacing the synthetic reference dataset (which stays as
an offline fallback).

MODEL: Sonnet, medium effort. This is a documented ETL pipeline + loader edits,
not net-new algorithms. The spec below is precise; where it says VERIFY, confirm
against the CDC codebook rather than trusting variable names blind. Bump to Opus
only if codebook verification gets ambiguous.

REPO: /Users/gregchism/dev/InfoSciCourses/INFO521/info521-projects-2026
(adjust if you unzipped elsewhere). Activate the info521-projects env first.

COMMITS: commit after every stage, and more often within stages. Do NOT touch
git remote, push, or any GitHub action — local commits only.

------------------------------------------------------------------
STAGE 1 — build script + dataset (then STOP for review)
------------------------------------------------------------------
1. Add `pandas` to environment.yml as a BUILD-time dependency (the student-facing
   loader stays numpy-only; pandas is only for building the CSV).

2. Write data/build_nhanes.py for the 2021–2022 cycle (suffix _L):
   - Download these XPT files and cache them in data/raw/ (gitignore data/raw/):
       DEMO_L   (demographics)
       BPXO_L   (blood pressure, OSCILLOMETRIC — auscultatory BPX ended after
                 2017-2018, so 2021-2022 has oscillometric only)
       BMX_L    (body measures)
       TCHOL_L  (total cholesterol)
       HDL_L    (HDL)
       GHB_L    (glycohemoglobin / HbA1c)
     URL pattern to try first (VERIFY each resolves; adjust year folder if not):
       https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/<FILE>.xpt
     Read with pandas.read_sas(path, format="xport").
     # NOTE in a comment: 2021-2022 was the first post-pandemic field cycle with
     # some sampling adjustments. We ignore survey weights here because this is an
     # algorithms course about estimators, not population inference.

   - Merge all files on SEQN (inner join on the needed columns).

   - Variables to pull (VERIFY names/labels against each file's CDC codebook page,
     e.g. .../DataFiles/BPXO_L.htm — the OSCILLOMETRIC BP names in particular,
     they differ from the auscultatory file):
       age   = RIDAGEYR              (DEMO_L)
       sbp   = mean of available BPXOSY1..BPXOSY3   (BPXO_L; ignore 0 / missing)
       dbp   = mean of available BPXODI1..BPXODI3   (BPXO_L; ignore 0 / missing)
       bmi   = BMXBMI               (BMX_L)
       waist = BMXWAIST             (BMX_L)
       chol  = LBXTC                (TCHOL_L)
       hdl   = LBDHDD               (HDL_L)
       hba1c = LBXGH                (GHB_L)

   - Inclusion: adults 18 <= RIDAGEYR <= 79 (avoid the age-80 topcode); complete
     cases on [age, sbp, dbp, bmi, waist, chol, hdl, hba1c]; drop rows where any
     BP reading used is 0.

   - Write data/nhanes_2021_2022.csv with a header row and these columns IN ORDER:
       age, bmi, waist, chol, hdl, hba1c, sbp, dbp
     (features first; sbp = continuous target; dbp = reserved for the label only).

3. Print a QA summary (do not skip): N retained; age range; mean/SD of sbp; the
   hypertension prevalence under SBP>=130 or DBP>=80; per-column missingness
   (should be 0 post-filter); and the 8x8 correlation matrix. Sanity target:
   ~4,000–6,000 adults and hypertension prevalence roughly 0.45–0.50.

4. Commit ("build: NHANES 2021-2022 loader + cached CSV").

>>> STOP. Show me the QA summary before changing the loader default. <<<

------------------------------------------------------------------
STAGE 2 — switch the loader (after I approve the QA)
------------------------------------------------------------------
Edit common/info521/data.py, keeping it numpy-only (np.genfromtxt):
  - Default path -> data/nhanes_2021_2022.csv. If absent, raise a clear error
    pointing to build_nhanes.py, and mention the synthetic CSV as an offline
    fallback via load_clinical(path=...).
  - Contract: target column is 'sbp'; 'dbp' is RESERVED (not a feature, not the
    target). Return dict: X = all columns except sbp AND dbp; y = sbp;
    dbp = the dbp column; features = those feature names; primary = 'age';
    target = 'sbp'.
  - Replace the binarize helper set:
      * keep binarize(y, threshold=None) for a generic median split, AND
      * add hypertension(ds) -> (labels, rule_str) using the ACC/AHA rule
        sbp>=130 OR dbp>=80. This is the leakage-safe Project 2 label.
  - Do NOT modify checks.py: its fixtures are a fixed 5-point numeric oracle for
    verifying a from-scratch solver and are dataset-independent.
Commit.

------------------------------------------------------------------
STAGE 3 — reframe prose to systolic BP
------------------------------------------------------------------
Update wording only (no logic) in project-1/1.1–1.4 .qmd, project-1/README.md,
project-2/README.md, data/README.md, instructor/dataset-curation.md:
  - "cardiometabolic risk score" / "risk" target -> "systolic blood pressure (SBP)".
  - Axis labels and the scenario paragraph: age -> SBP.
  - Project 2 bridge: binarizing SBP -> hypertension (ACC/AHA), predicted from the
    non-BP features; note explicitly that BP-derived columns are excluded from the
    P2 feature set to avoid label leakage.
  - dataset-curation.md: mark NHANES 2021-2022 as the DEPLOYED default; keep the
    synthetic generator documented as the offline fallback.
Commit.

------------------------------------------------------------------
STAGE 4 — smoke test
------------------------------------------------------------------
- python data/build_nhanes.py (confirm CSV regenerates) -- skip the download if
  data/raw/ is already cached.
- python -c "from info521 import data; d=data.load_clinical(); print(d['X'].shape, d['features']); import numpy as np; print('HTN prev', round(data.hypertension(d)[0].mean(),3))"
- quarto render project-1/1.1-least-squares.qmd
Commit ("test: NHANES default renders end-to-end"). Print a final summary of what
changed and the dataset stats.
```

---

## Notes for the instructor (not for Claude Code)

- The oscillometric BP variable names (`BPXOSY*` / `BPXODI*`) are the one thing
  worth eyeballing in the QA output — a wrong name silently breaks the merge.
- Survey weights are deliberately ignored (algorithms course, not population
  inference). The build script leaves a comment saying so.
- The age-80 topcode is handled by restricting to 18–79; consider surfacing it as
  a teachable aside in the 1.1 prose if you want — left out by default so the
  pedagogical call stays yours.
- To add the third arm of the ACC/AHA label ("currently on BP medication"), pull
  `BPQ_L` (`BPQ050A`) from the Questionnaire category and OR it into
  `hypertension(ds)`. Omitted by default to keep the file list to six.
