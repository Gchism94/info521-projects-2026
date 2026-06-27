"""Instructor-only reference implementations for the INFO 521 project notebooks.

This module is the deliberate counterpart to `info521` staying plumbing-only for
students: it holds the *from-scratch* estimators a correct solution would build, so
the reference solution notebooks under solutions/ can import and apply them instead
of re-deriving each one inline.

DO NOT ship this to students and DO NOT import it from `info521`. It lives only in
the instructor-only solutions/ tree. Every routine is NumPy/SciPy from scratch (no
scikit-learn), mirroring the student ground rules.

Notation follows PML where relevant.
"""

from __future__ import annotations

import numpy as np

# ======================================================================== #
# Project 1.1 — least squares, CV, ridge
# ======================================================================== #
def design_matrix(x, order):
    """Order-``order`` polynomial design: columns [1, x, x^2, ..., x^order]."""
    x = np.asarray(x, dtype=float).ravel()
    return np.vander(x, N=order + 1, increasing=True)


def ols_fit(X, y):
    """OLS via the normal equations (solve, not an explicit inverse)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    return np.linalg.solve(X.T @ X, X.T @ y)


def ridge_fit(X, y, lam):
    """Ridge: (X^T X + N*lam*I)^-1 X^T y. Matches checks.LINEAR_FIXTURE."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    return np.linalg.solve(X.T @ X + n * lam * np.eye(d), X.T @ y)


def predict(X, w):
    return np.asarray(X, float) @ np.asarray(w, float)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def kfold_indices(n, k, seed=521):
    """List of (train_idx, val_idx) for K folds over n points, seeded."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    out = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, val))
    return out


def cv_polynomial(x, y, order, k=5, lam=0.0, seed=521):
    """Mean train/val RMSE for a polynomial of a given order (optionally ridge).

    Standardizes the polynomial columns on the TRAIN fold only (so high orders are
    numerically sane and there is no fold-to-fold leakage). Returns (train, val).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float)
    n = len(x)
    tr_errs, va_errs = [], []
    for train, val in kfold_indices(n, k, seed):
        Xtr_raw = design_matrix(x[train], order)
        Xva_raw = design_matrix(x[val], order)
        # standardize non-intercept columns on train only
        mu = Xtr_raw[:, 1:].mean(axis=0)
        sd = Xtr_raw[:, 1:].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        Xtr = np.column_stack([np.ones(len(train)), (Xtr_raw[:, 1:] - mu) / sd])
        Xva = np.column_stack([np.ones(len(val)), (Xva_raw[:, 1:] - mu) / sd])
        w = ridge_fit(Xtr, y[train], lam) if lam > 0 else ols_fit(Xtr, y[train])
        tr_errs.append(rmse(y[train], Xtr @ w))
        va_errs.append(rmse(y[val], Xva @ w))
    return float(np.mean(tr_errs)), float(np.mean(va_errs))


# ======================================================================== #
# Project 1.2 — MLE
# ======================================================================== #
def mle_weights(X, y):
    """MLE for w under Gaussian noise == the OLS solution."""
    return ols_fit(X, y)


def mle_sigma2(X, y, w):
    """MLE noise variance = (1/N) sum of squared residuals."""
    r = np.asarray(y, float) - predict(X, w)
    return float(np.mean(r ** 2))


# ======================================================================== #
# Project 1.3 — conjugate Gaussian posterior
# ======================================================================== #
def gaussian_posterior(X, y, sigma2, tau2):
    """Posterior over w for prior N(0, tau2 I), likelihood N(Xw, sigma2 I).

    Returns (mean m_N, cov S_N) with precision A = X^T X / sigma2 + I / tau2.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    d = X.shape[1]
    A = (X.T @ X) / sigma2 + np.eye(d) / tau2
    S_N = np.linalg.inv(A)
    m_N = (S_N @ X.T @ y) / sigma2
    return m_N, S_N


def posterior_predictive(Phi, m_N, S_N, sigma2):
    """Predictive mean and variance at design rows Phi (includes noise sigma2)."""
    Phi = np.asarray(Phi, float)
    mean = Phi @ m_N
    var = sigma2 + np.einsum("ij,jk,ik->i", Phi, S_N, Phi)
    return mean, var


# ======================================================================== #
# Shared supervised plumbing (2.1 / 2.2)
# ======================================================================== #
def train_test_split(X, y, test_frac=0.25, seed=521):
    X = np.asarray(X, float)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test, train = perm[:n_test], perm[n_test:]
    return X[train], X[test], y[train], y[test]


def standardize_fit(X_train):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def standardize_apply(X, mu, sd):
    return (np.asarray(X, float) - mu) / sd


def add_intercept(X):
    X = np.asarray(X, float)
    return np.column_stack([np.ones(len(X)), X])


# ======================================================================== #
# Project 2.1 — Bayesian logistic regression
# ======================================================================== #
def sigmoid(a):
    """Numerically stable logistic sigmoid."""
    a = np.asarray(a, float)
    out = np.empty_like(a)
    pos = a >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-a[pos]))
    ea = np.exp(a[~pos])
    out[~pos] = ea / (1.0 + ea)
    return out


def _prior_precision(d, tau2, fit_intercept=True):
    """S0^{-1} = diag(1/tau2), with the intercept (column 0) left UNpenalized."""
    p = np.full(d, 1.0 / tau2)
    if fit_intercept:
        p[0] = 0.0
    return p


def neg_log_posterior(w, X, y, tau2, fit_intercept=True):
    """Negative log posterior (energy E) of Bayesian logistic regression."""
    w = np.asarray(w, float)
    eta = X @ w
    # stable log-likelihood: sum[ y*eta - softplus(eta) ]
    softplus = np.logaddexp(0.0, eta)
    ll = np.sum(y * eta - softplus)
    p = _prior_precision(X.shape[1], tau2, fit_intercept)
    nlp = -ll + 0.5 * np.sum(p * w * w)
    return float(nlp)


def grad_neg_log_posterior(w, X, y, tau2, fit_intercept=True):
    w = np.asarray(w, float)
    mu = sigmoid(X @ w)
    p = _prior_precision(X.shape[1], tau2, fit_intercept)
    return X.T @ (mu - y) + p * w


def hess_neg_log_posterior(w, X, y, tau2, fit_intercept=True):
    w = np.asarray(w, float)
    mu = sigmoid(X @ w)
    R = mu * (1.0 - mu)
    p = _prior_precision(X.shape[1], tau2, fit_intercept)
    return (X * R[:, None]).T @ X + np.diag(p)


def fit_map_newton(X, y, tau2, w0=None, n_iter=100, tol=1e-8, fit_intercept=True):
    """Newton-Raphson (IRLS) to the MAP. Returns (w, history of grad-norms)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1]) if w0 is None else np.asarray(w0, float).copy()
    history = []
    for _ in range(n_iter):
        g = grad_neg_log_posterior(w, X, y, tau2, fit_intercept)
        H = hess_neg_log_posterior(w, X, y, tau2, fit_intercept)
        history.append(float(np.linalg.norm(g)))
        step = np.linalg.solve(H, g)
        w = w - step
        if np.linalg.norm(step) < tol:
            g = grad_neg_log_posterior(w, X, y, tau2, fit_intercept)
            history.append(float(np.linalg.norm(g)))
            break
    return w, history


def laplace_posterior(X, y, tau2, w_map, fit_intercept=True):
    """Laplace approx: N(w_map, H^{-1}) with H the Hessian of E at the mode."""
    H = hess_neg_log_posterior(w_map, X, y, tau2, fit_intercept)
    return w_map, np.linalg.inv(H)


def metropolis(X, y, tau2, w_init, n_samples=20000, proposal_sd=0.1,
               burn_in=2000, seed=521, fit_intercept=True):
    """Random-walk Metropolis on the posterior. Returns (chain, accept_rate)."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w_init, float).copy()
    d = len(w)
    cur = neg_log_posterior(w, X, y, tau2, fit_intercept)
    chain = np.empty((n_samples, d))
    n_acc = 0
    for t in range(n_samples):
        prop = w + proposal_sd * rng.standard_normal(d)
        cand = neg_log_posterior(prop, X, y, tau2, fit_intercept)
        # accept with prob min(1, exp(-cand + cur)); work in log space
        if np.log(rng.random()) < (cur - cand):
            w, cur = prop, cand
            n_acc += 1
        chain[t] = w
    return chain[burn_in:], n_acc / n_samples


def predict_proba(X, w):
    return sigmoid(np.asarray(X, float) @ np.asarray(w, float))


# ======================================================================== #
# Project 2.2 — soft-margin SVM (Pegasos) + metrics
# ======================================================================== #
def svm_fit_pegasos(X, t, lam=1e-3, n_epochs=50, seed=521):
    """Linear soft-margin SVM by Pegasos (step 1/(lam*step)); bias unregularized.

    ``t`` must be in {-1, +1}. Pegasos minimizes (lam/2)||w||^2 + (1/N) sum hinge;
    the primal-C form J = (1/2)||w||^2 + C sum hinge corresponds to C = 1/(N*lam).
    Returns (w, b).
    """
    X = np.asarray(X, float)
    t = np.asarray(t, float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    rng = np.random.default_rng(seed)
    step = 0
    for _ in range(n_epochs):
        for i in rng.permutation(n):
            step += 1
            eta = 1.0 / (lam * step)
            if t[i] * (X[i] @ w + b) < 1.0:
                w = (1.0 - eta * lam) * w + eta * t[i] * X[i]
                b = b + eta * t[i]
            else:
                w = (1.0 - eta * lam) * w
    return w, b


def svm_decision(X, w, b):
    return np.asarray(X, float) @ np.asarray(w, float) + b


def svm_predict(X, w, b):
    """Hard labels in {-1, +1}."""
    s = svm_decision(X, w, b)
    return np.where(s >= 0, 1, -1)


def margin_width(w):
    return 2.0 / np.linalg.norm(w)


def n_support_vectors(X, t, w, b):
    """Points on/inside the margin: t*(w.x+b) <= 1."""
    return int(np.sum(np.asarray(t, float) * svm_decision(X, w, b) <= 1.0))


def precision_recall_f1(y_true01, y_pred01):
    """Precision/recall/F1 for the positive class from {0,1} arrays."""
    yt = np.asarray(y_true01).astype(int)
    yp = np.asarray(y_pred01).astype(int)
    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# ======================================================================== #
# Project 2.3 — k-means
# ======================================================================== #
def kmeans(X, K, n_iter=100, seed=521):
    """Lloyd's k-means from random distinct rows. Returns (labels, centroids, wcss)."""
    X = np.asarray(X, float)
    n = len(X)
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(n, K, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for k in range(K):
            mask = labels == k
            if mask.any():
                centroids[k] = X[mask].mean(axis=0)
            else:  # re-seed an empty cluster on the farthest point
                centroids[k] = X[d2.min(axis=1).argmax()]
    return labels, centroids, wcss(X, labels, centroids)


def wcss(X, labels, centroids):
    X = np.asarray(X, float)
    total = 0.0
    for k in range(len(centroids)):
        mask = labels == k
        if mask.any():
            total += float(((X[mask] - centroids[k]) ** 2).sum())
    return total


def kmeans_best_of(X, K, n_restarts=10, seed=521):
    """Best (lowest-WCSS) of several k-means restarts. Returns (labels, centroids,
    wcss, all_wcss)."""
    best = None
    all_w = []
    for r in range(n_restarts):
        lab, cen, w = kmeans(X, K, seed=seed + r)
        all_w.append(w)
        if best is None or w < best[2]:
            best = (lab, cen, w)
    return best[0], best[1], best[2], all_w


def silhouette(X, labels):
    """Mean silhouette score from scratch (Euclidean)."""
    X = np.asarray(X, float)
    n = len(X)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            other = labels == c
            if other.any():
                b = min(b, D[i, other].mean())
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(s.mean())


def crosstab_fraction(cluster_labels, binary_label):
    """K x 2 contingency counts + per-cluster positive fraction."""
    cl = np.asarray(cluster_labels)
    bl = np.asarray(binary_label).astype(int)
    Ks = np.unique(cl)
    table = np.zeros((len(Ks), 2), dtype=int)
    frac = np.zeros(len(Ks))
    for i, k in enumerate(Ks):
        m = cl == k
        table[i, 0] = int(np.sum(bl[m] == 0))
        table[i, 1] = int(np.sum(bl[m] == 1))
        frac[i] = bl[m].mean() if m.any() else 0.0
    return Ks, table, frac


def cluster_label_accuracy(cluster_labels, binary_label):
    """Best-case agreement if each cluster votes its majority label (purity-style)."""
    cl = np.asarray(cluster_labels)
    bl = np.asarray(binary_label).astype(int)
    correct = 0
    for k in np.unique(cl):
        m = cl == k
        correct += max(int(np.sum(bl[m] == 0)), int(np.sum(bl[m] == 1)))
    return correct / len(bl)


# ======================================================================== #
# Project 2.4 — PCA
# ======================================================================== #
def pca_fit(Xc):
    """PCA via SVD of the centered matrix. Returns (components V (d,k), evr, lambdas).

    Components are columns; sign convention: largest-magnitude loading positive.
    """
    Xc = np.asarray(Xc, float)
    n = len(Xc)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lambdas = (S ** 2) / (n - 1)
    V = Vt.T
    # deterministic sign
    for j in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1
    evr = lambdas / lambdas.sum()
    return V, evr, lambdas


def pca_project(Xc, V, k):
    return np.asarray(Xc, float) @ V[:, :k]
