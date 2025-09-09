from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearModel:
    feature_names: list[str]
    w: np.ndarray  # (d,)
    b: float
    x_mean: np.ndarray  # (d,)
    x_std: np.ndarray  # (d,)
    sigma: float  # residual std as simple UQ

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.x_mean) / (self.x_std + 1e-8)
        return Z @ self.w + self.b


def ridge_train(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, float]:
    # X is assumed standardized (zero-mean, unit-var per feature)
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    w = np.linalg.solve(A, X.T @ y)
    b = float(y.mean())
    return w, b


def fit_linear_regressor(
    feature_names: list[str], X: np.ndarray, y: np.ndarray, lam: float = 1e-3
) -> LinearModel:
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    Z = (X - x_mean) / (x_std + 1e-8)
    w, b = ridge_train(Z, y, lam=lam)
    y_hat = Z @ w + b
    resid = y_hat - y
    sigma = float(np.sqrt(np.maximum(1e-12, (resid**2).mean())))
    return LinearModel(
        feature_names=feature_names, w=w, b=b, x_mean=x_mean, x_std=x_std, sigma=sigma
    )
