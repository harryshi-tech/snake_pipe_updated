# snake_control/src/snake_control/gaitlib/shape_models.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SerpenoidLinearShapeModel:
    """
    Minimal shape model for SBC readiness.

    q_i = kappa + A1*sin(eta*s_i) + A2*cos(eta*s_i)
    sigma = [kappa, A1, A2]

    - jacobian(): (N,3)
    - synthesize(sigma): sigma (...,3) -> q (...,N)
    - project(q): q (...,N) -> sigma (...,3)  (least squares)

    s: arc-length coordinate for each joint; if None uses s = ds*[0..N-1].
    """
    n_joints: int
    eta: float = 1.0
    s: np.ndarray | None = None
    ds: float = 1.0

    def __post_init__(self):
        if self.s is None:
            self.s = self.ds * np.arange(self.n_joints, dtype=float)
        self.s = np.asarray(self.s, dtype=float).reshape(-1)
        if self.s.size != self.n_joints:
            raise ValueError(f"s must be length n_joints={self.n_joints}, got {self.s.size}")

        self._ones = np.ones(self.n_joints, dtype=float)
        self._sin = np.sin(self.eta * self.s)
        self._cos = np.cos(self.eta * self.s)

    def jacobian(self) -> np.ndarray:
        return np.vstack([self._ones, self._sin, self._cos]).T  # (N,3)

    def synthesize(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape[-1] != 3:
            raise ValueError(f"sigma last-dim must be 3, got {sigma.shape}")

        kappa = sigma[..., 0][..., None]
        A1 = sigma[..., 1][..., None]
        A2 = sigma[..., 2][..., None]
        return kappa * self._ones + A1 * self._sin + A2 * self._cos

    def project(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape[-1] != self.n_joints:
            raise ValueError(f"q last-dim must be n_joints={self.n_joints}, got {q.shape}")

        J = self.jacobian()  # (N,3)

        q2 = q.reshape(-1, self.n_joints)  # (M,N)
        sigmas = []
        for row in q2:
            sigma, *_ = np.linalg.lstsq(J, row, rcond=None)
            sigmas.append(sigma)
        sigmas = np.asarray(sigmas, dtype=float)

        return sigmas.reshape(q.shape[:-1] + (3,))
