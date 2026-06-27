"""Value-based self-check fixtures.

Each fixture is a fixed (X, y) input paired with the *expected numeric output* of
a correct implementation. Students call the check against their own from-scratch
function. We ship the answers as numbers, never as code, so the fixtures verify
correctness without revealing the algorithm.

Usage in a notebook:

    from info521 import checks
    X, y = checks.LINEAR_FIXTURE.X, checks.LINEAR_FIXTURE.y
    w = my_ols(X, y)                      # student's own implementation
    checks.expect_close(w, checks.LINEAR_FIXTURE.w_ols, "OLS weights")
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def expect_close(got, expected, name="value", atol=1e-4, rtol=1e-4):
    """Assert-style check that prints a clear pass/fail line."""
    got = np.asarray(got, dtype=float)
    expected = np.asarray(expected, dtype=float)
    ok = got.shape == expected.shape and np.allclose(got, expected, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        print(f"       expected: {np.round(expected, 6).tolist()}")
        print(f"       got:      {np.round(got, 6).tolist()}")
    return ok


@dataclass(frozen=True)
class LinearFixture:
    # Fixed order-2 polynomial design: columns [1, x, x^2] over x = 0..4.
    x: np.ndarray
    X: np.ndarray
    y: np.ndarray
    w_ols: np.ndarray        # ordinary least squares solution
    yhat_ols: np.ndarray     # OLS predictions on the fixture inputs
    sigma2_mle: float        # MLE noise variance = (1/N) sum residual^2
    lam_ridge: float
    w_ridge: np.ndarray      # ridge solution: (X^T X + N*lam I)^-1 X^T y


_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
_X = np.column_stack([np.ones_like(_x), _x, _x ** 2])
_y = np.array([1.0, 0.9, 3.2, 6.8, 13.1])

LINEAR_FIXTURE = LinearFixture(
    x=_x,
    X=_X,
    y=_y,
    w_ols=np.array([0.994286, -1.018571, 1.007143]),
    yhat_ols=np.array([0.994286, 0.982857, 2.985714, 7.002857, 13.034286]),
    sigma2_mle=0.019657,
    lam_ridge=0.5,
    w_ridge=np.array([0.216132, -0.01038, 0.78276]),
)
