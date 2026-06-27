"""Generate the synthetic clinical *reference* dataset shipped with the starter.

This dataset exists so the whole Project 1 -> Project 2 pipeline runs out of the
box and is fully reproducible. It is SYNTHETIC and must not be presented to
students as real clinical data. For deployment, swap in one of the curated real
datasets documented in ../instructor/dataset-curation.md.

Design goals (so every learning outcome has something to bite on):
  * A single primary predictor (age) with a gentle *nonlinear* trend to the
    target, so polynomial order matters: order 1 underfits, order ~3 is right,
    order 9 overfits. This powers the 1.1 play artifact.
  * Heteroscedastic-ish sparsity at the edges (few old patients) so predictive
    error bars visibly widen there in 1.2.
  * A continuous target that binarizes cleanly at its median -> high/low risk,
    for the Project 2 conjugacy-break bridge.
  * Correlated feature blocks so PCA finds real structure and k-means finds
    ~2-3 subgroups in Project 2.

Run:  python generate_reference_data.py
"""

import numpy as np

SEED = 521
N = 400
OUT = "clinical_reference.csv"


def generate():
    rng = np.random.default_rng(SEED)

    # --- primary predictor: age, sampled so the upper tail is sparse ---------
    # Beta(2, 3) skews young; rescale to [30, 85].
    age = 30 + 55 * rng.beta(2.0, 3.0, size=N)

    # --- correlated feature blocks (for PCA / clustering in Project 2) -------
    # Block A: metabolic (bmi, glucose, hba1c) share a latent factor.
    z_metabolic = rng.normal(size=N)
    bmi = 27 + 4.5 * z_metabolic + rng.normal(0, 2.0, size=N)
    glucose = 95 + 18 * z_metabolic + 0.25 * (age - 55) + rng.normal(0, 8, size=N)
    hba1c = 5.4 + 0.9 * z_metabolic + rng.normal(0, 0.3, size=N)

    # Block B: lipids (chol, hdl) share a latent factor (hdl anti-correlated).
    z_lipid = rng.normal(size=N)
    chol = 190 + 30 * z_lipid + rng.normal(0, 12, size=N)
    hdl = 55 - 12 * z_lipid + rng.normal(0, 6, size=N)

    # Block C: blood pressure, drifts with age + own noise.
    sbp = 110 + 0.45 * (age - 30) + 6 * z_metabolic + rng.normal(0, 7, size=N)

    # --- continuous target: cardiometabolic risk score ----------------------
    # Nonlinear (accelerating) in age -> justifies polynomial regression on age
    # alone in Project 1. Other features contribute the multivariate signal that
    # Project 2 unlocks.
    age_c = (age - 55) / 10.0
    f_age = 12.0 + 3.0 * age_c + 1.4 * age_c**2 + 0.25 * age_c**3
    multivariate = (
        0.35 * (bmi - 27)
        + 0.06 * (glucose - 95)
        + 0.04 * (sbp - 120)
        - 0.05 * (hdl - 55)
        + 3.0 * (hba1c - 5.6)
    )
    noise = rng.normal(0, 3.0, size=N)
    risk = f_age + multivariate + noise

    cols = {
        "age": age,
        "bmi": bmi,
        "sbp": sbp,
        "glucose": glucose,
        "hba1c": hba1c,
        "chol": chol,
        "hdl": hdl,
        "risk": risk,
    }
    names = list(cols)
    M = np.column_stack([cols[k] for k in names])

    header = ",".join(names)
    np.savetxt(OUT, M, delimiter=",", header=header, comments="", fmt="%.4f")
    print(f"Wrote {OUT}: {M.shape[0]} rows x {M.shape[1]} cols")
    print(f"  primary predictor : age  [{age.min():.1f}, {age.max():.1f}]")
    print(f"  target            : risk [{risk.min():.1f}, {risk.max():.1f}]")
    print(f"  median(risk)      : {np.median(risk):.3f}  (Project 2 binarization threshold)")
    frac_high = float(np.mean(risk > np.median(risk)))
    print(f"  high-risk fraction at median split: {frac_high:.2f}")


if __name__ == "__main__":
    generate()
