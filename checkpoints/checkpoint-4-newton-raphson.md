# Checkpoint 4 — Newton–Raphson Update for Bayesian Logistic Regression

**Gates:** Project 2.1 (Approximate Inference) — and with it, Gate A.
**Format:** in-class, closed-book, ~15–20 min, one derivation. Graded
Satisfactory / Not-Yet. **Retakeable** (an alternate version is provided below).

This checkpoint tests the study-guide outcome *"derive a second-order optimizer for
a non-conjugate posterior."* It is the supervised-gate counterpart to Checkpoints
1–2: where those derived the closed-form least-squares/MLE solution, this one
derives the **iterative** update you need once conjugacy is gone.

---

## Prompt (Version A)

Bayesian logistic regression places a Gaussian prior
$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{S}_0)$ on the weights, with a
Bernoulli likelihood $y_n \sim \text{Bernoulli}(\mu_n)$,
$\mu_n = \sigma(\mathbf{w}^\top\mathbf{x}_n)$ and $\sigma$ the logistic sigmoid.
Writing $E(\mathbf{w}) = -\log p(\mathbf{w}\mid\mathcal{D})$ (up to a constant),
derive (i) the **gradient** $\nabla E$, (ii) the **Hessian** $\nabla^2 E$, and
(iii) the **Newton–Raphson update** for the MAP estimate. State why the update is
called *iteratively reweighted least squares*.

## Prompt (Version B — retake)

Explain the **Laplace approximation** of the posterior in this same model: what it
is, *why* the **negative Hessian of the log-posterior at the mode** is the
posterior **precision**, and give **one** situation in which the approximation is
poor.

---

## Satisfactory criteria

- (A) Correct $\nabla E$ and $\nabla^2 E$, including the prior term
  $\mathbf{S}_0^{-1}\mathbf{w}$ (gradient) and $\mathbf{S}_0^{-1}$ (Hessian).
- (A) States the update $\mathbf{w} \leftarrow \mathbf{w} - \mathbf{H}^{-1}\mathbf{g}$
  and identifies the reweighting matrix $\mathbf{R} = \text{diag}(\mu_n(1-\mu_n))$
  as the "weights" that make it reweighted least squares.
- (B) Names the Gaussian centred at the mode; argues the precision = curvature of
  the log-posterior at the mode; gives one valid failure case (skewed / multimodal
  / near-separable posterior).

## Answer key

Let $\boldsymbol\mu = (\mu_1,\dots,\mu_N)^\top$,
$\mathbf{R} = \text{diag}\big(\mu_n(1-\mu_n)\big)$, and $\mathbf{X}$ the design
matrix with rows $\mathbf{x}_n^\top$.

**A:**

$$
\nabla E = \mathbf{X}^\top(\boldsymbol\mu - \mathbf{y}) + \mathbf{S}_0^{-1}\mathbf{w},
\qquad
\nabla^2 E = \mathbf{X}^\top \mathbf{R}\, \mathbf{X} + \mathbf{S}_0^{-1}.
$$

$$
\mathbf{w} \leftarrow \mathbf{w} -
\big(\mathbf{X}^\top \mathbf{R}\,\mathbf{X} + \mathbf{S}_0^{-1}\big)^{-1}
\big(\mathbf{X}^\top(\boldsymbol\mu - \mathbf{y}) + \mathbf{S}_0^{-1}\mathbf{w}\big).
$$

It is *reweighted least squares* because each Newton step solves a weighted
least-squares problem with weights $\mathbf{R}$ that are **recomputed** from
$\boldsymbol\mu$ every iteration.

**B:** Taylor-expand $E$ to second order at the mode $\mathbf{w}_{\text{MAP}}$. The
first-order term vanishes ($\nabla E = \mathbf{0}$ at the mode), leaving
$E(\mathbf{w}) \approx E(\mathbf{w}_{\text{MAP}}) +
\tfrac{1}{2}(\mathbf{w}-\mathbf{w}_{\text{MAP}})^\top \mathbf{H}\,
(\mathbf{w}-\mathbf{w}_{\text{MAP}})$ with
$\mathbf{H} = \nabla^2 E(\mathbf{w}_{\text{MAP}})$. Exponentiating gives a Gaussian
$q(\mathbf{w}) = \mathcal{N}(\mathbf{w}_{\text{MAP}}, \mathbf{H}^{-1})$: since a
Gaussian's log-density has curvature equal to its **precision**, matching the
curvature of the log-posterior at the mode sets the precision to
$\mathbf{H} = \mathbf{X}^\top\mathbf{R}\,\mathbf{X} + \mathbf{S}_0^{-1}$ (the
negative Hessian of the *log*-posterior). It is poor when the true posterior is far
from Gaussian — e.g. skewed, multimodal, or near-separable data (few positives, or
perfectly separable classes), where a single mode-centred Gaussian misstates the
spread.
