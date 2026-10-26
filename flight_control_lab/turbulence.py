"""Seeded stochastic turbulence for reproducible teaching scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .aircraft6dof import RigidBodyState, quat_to_matrix


@dataclass(frozen=True)
class StochasticTurbulence:
    """Small zero-mean gust field generated from a fixed seed.

    The model is a compact sum of random-phase sinusoids. It is deterministic
    for a given seed, which keeps tests and demos repeatable. It is not a
    certified Dryden model.
    """

    enabled: bool = True
    seed: int = 7
    intensity_mps: float = 1.1
    vertical_intensity_ratio: float = 0.65
    spatial_scale_m: float = 260.0
    time_scale_s: float = 8.0
    components: int = 6

    def validate(self) -> None:
        if self.components <= 0:
            raise ValueError("components must be positive")
        if self.intensity_mps < 0 or self.vertical_intensity_ratio < 0:
            raise ValueError("turbulence intensities must be non-negative")
        if self.spatial_scale_m <= 0 or self.time_scale_s <= 0:
            raise ValueError("turbulence scales must be positive")

    def wind_nav(self, position: np.ndarray, t: float) -> np.ndarray:
        self.validate()
        if not self.enabled or self.intensity_mps == 0:
            return np.zeros(3)
        rng = np.random.default_rng(self.seed)
        phases = rng.uniform(0.0, 2.0 * math.pi, (3, self.components))
        frequency_ratio = rng.uniform(0.72, 1.28, (3, self.components))
        direction_ratio = rng.uniform(0.72, 1.28, (3, self.components))
        x, y, _ = np.asarray(position, dtype=float)
        spatial = (direction_ratio * x + (2.0 - direction_ratio) * y) / self.spatial_scale_m
        temporal = 2.0 * math.pi * frequency_ratio * t / self.time_scale_s
        waves = np.sin(spatial + temporal + phases) / math.sqrt(self.components)
        raw = np.sum(waves, axis=1)
        return np.array(
            [
                0.75 * self.intensity_mps * raw[0],
                self.intensity_mps * raw[1],
                self.intensity_mps * self.vertical_intensity_ratio * raw[2],
            ],
            dtype=float,
        )

    def wind_body(self, state: RigidBodyState, t: float) -> np.ndarray:
        return quat_to_matrix(state.quaternion).T @ self.wind_nav(state.position, t)

    def metrics(self) -> dict[str, float | int | bool]:
        return {
            "turbulence_enabled": self.enabled,
            "turbulence_seed": self.seed,
            "turbulence_intensity_mps": self.intensity_mps,
        }
