# Checkpoint 1 — Normal Equations

**Gates:** Project 1.1 (Least Squares)
**Format:** in-class, closed-book, ~15 min, one derivation. Graded
Satisfactory / Not-Yet. **Retakeable** (an alternate version is provided below).

---

## Prompt (Version A)

For the single-predictor linear model $f(x; w_0, w_1) = w_0 + w_1 x$ with squared-error loss

$$
\mathcal{L} = \frac{1}{N}\sum_{n=1}^{N}\bigl(y_n - (w_0 + w_1 x_n)\bigr)^2,
$$

derive the least-squares estimator $\hat{w}_0$ (you may leave it in terms of
$\hat{w}_1$). Your final answer must contain no summation symbols; use
$\overline{a} = \tfrac{1}{N}\sum_n a_n$.

## Prompt (Version B — retake)

Write the squared-error loss for a general linear model in matrix/vector form and
derive the normal equations
$\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$
by setting the gradient to zero. State the one assumption needed to invert.

---

## Satisfactory criteria

- Sets $\partial\mathcal{L}/\partial w_0 = 0$ (A) or $\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{0}$ (B).
- Algebra is correct and complete to the stated end result.
- (B) States that $\mathbf{X}^\top\mathbf{X}$ must be invertible (full column rank).

## Answer key

**A:** $\hat{w}_0 = \overline{y} - \hat{w}_1 \overline{x}$.

**B:** From $\mathcal{L} = \tfrac{1}{N}(\mathbf{y}-\mathbf{Xw})^\top(\mathbf{y}-\mathbf{Xw})$,
$\nabla_{\mathbf{w}}\mathcal{L} = -\tfrac{2}{N}\mathbf{X}^\top(\mathbf{y}-\mathbf{Xw}) = \mathbf{0}
\Rightarrow \mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^\top\mathbf{y}
\Rightarrow \hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$,
valid when $\mathbf{X}^\top\mathbf{X}$ is invertible.
