# Checkpoint 2 — Maximum Likelihood Estimator for the Weights

**Gates:** Project 1.2 (Maximum Likelihood)
**Format:** in-class, closed-book, ~15–20 min, one derivation. Graded
Satisfactory / Not-Yet. **Retakeable.**

---

## Prompt (Version A)

Assume $y_n = \mathbf{w}^\top\boldsymbol{\phi}(x_n) + \epsilon_n$ with
$\epsilon_n \sim \mathcal{N}(0, \sigma^2)$ i.i.d. Write the log-likelihood of the
dataset and derive the maximum likelihood estimator $\hat{\mathbf{w}}$. State why
it coincides with the least-squares solution.

## Prompt (Version B — retake)

For the same model, derive the MLE for the noise variance $\hat{\sigma}^2$,
treating $\hat{\mathbf{w}}$ as known.

---

## Satisfactory criteria

- Correct dataset likelihood as a product over i.i.d. points; correct log.
- Maximizes (equivalently, drops constants and minimizes the negative log).
- Reaches the stated end result; (A) notes the $\sigma^2$ terms are constant in
  $\mathbf{w}$, so maximizing likelihood = minimizing squared error.

## Answer key

**A:** $\log p(\mathbf{y}\mid\mathbf{X},\mathbf{w},\sigma^2) =
-\tfrac{N}{2}\log(2\pi\sigma^2) - \tfrac{1}{2\sigma^2}(\mathbf{y}-\mathbf{Xw})^\top(\mathbf{y}-\mathbf{Xw})$.
The only $\mathbf{w}$-dependent term is the squared error, so
$\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ — identical to OLS.

**B:** $\hat{\sigma}^2 = \tfrac{1}{N}(\mathbf{y}-\mathbf{X}\hat{\mathbf{w}})^\top(\mathbf{y}-\mathbf{X}\hat{\mathbf{w}})$.
