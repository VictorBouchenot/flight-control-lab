"""Deterministic carrier airwake disturbance model.

This is a compact teaching model: it represents the disturbed airflow behind a
moving carrier as a localized headwind deficit, downwash ("burble") and light
crosswind unsteadiness. It is not a CFD model and does not represent any real
ship.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .aircraft6dof import RigidBodyState, quat_to_matrix
from .carrier6dof import MovingCarrier
from .dynamics import clamp


@dataclass(frozen=True)
class CarrierAirwake:
    """Localized deterministic airwake behind a moving carrier.

    Disturbance velocities are returned in NED navigation axes. Positive z is
    downward, so ``downwash`` is a positive z component.
    """

    enabled: bool = True
    max_downwash_mps: float = 2.4
    max_headwind_mps: float = 1.8
    max_crosswind_mps: float = 0.7
    peak_distance_m: float = 115.0
    distance_scale_m: float = 85.0
    height_scale_m: float = 34.0
    lateral_scale_m: float = 22.0
    unsteadiness: float = 0.22

    def validate(self) -> None:
        if min(
            self.max_downwash_mps,
            self.max_headwind_mps,
            self.max_crosswind_mps,
            self.peak_distance_m,
            self.distance_scale_m,
            self.height_scale_m,
            self.lateral_scale_m,
        ) < 0:
            raise ValueError("airwake amplitudes and scales must be non-negative")
        if min(self.distance_scale_m, self.height_scale_m, self.lateral_scale_m) == 0:
            raise ValueError("airwake spatial scales must be positive")
        if not 0.0 <= self.unsteadiness <= 0.5:
            raise ValueError("unsteadiness must stay within the teaching range")

    def _gains(self, position: np.ndarray, carrier: MovingCarrier, t: float) -> dict[str, float]:
        deck = carrier.deck_state(t)
        rel = np.asarray(position, dtype=float) - np.array(
            [deck["x_start"], deck["y"], deck["z_down"]], dtype=float
        )
        c = math.cos(deck["heading"])
        s = math.sin(deck["heading"])
        longitudinal = c * float(rel[0]) + s * float(rel[1])
        lateral = -s * float(rel[0]) + c * float(rel[1])
        aft = -longitudinal
        # NED z is down; above the deck is negative relative z.
        height_above_deck = max(0.0, -float(rel[2]))
        if aft < 0.0 or aft > self.peak_distance_m + 4.0 * self.distance_scale_m:
            return {
                "aft": aft,
                "lateral": lateral,
                "height_above_deck": height_above_deck,
                "distance_gain": 0.0,
                "lateral_gain": 0.0,
                "height_gain": 0.0,
            }
        return {
            "aft": aft,
            "lateral": lateral,
            "height_above_deck": height_above_deck,
            "distance_gain": math.exp(-((aft - self.peak_distance_m) / self.distance_scale_m) ** 2),
            "lateral_gain": math.exp(-((lateral) / self.lateral_scale_m) ** 2),
            "height_gain": math.exp(-((height_above_deck) / self.height_scale_m) ** 2),
        }

    def wind_nav(self, position: np.ndarray, carrier: MovingCarrier, t: float) -> np.ndarray:
        """Return deterministic airwake velocity in NED axes."""

        self.validate()
        carrier.validate()
        if not self.enabled:
            return np.zeros(3)
        gains = self._gains(position, carrier, t)
        gain = gains["distance_gain"] * gains["lateral_gain"] * gains["height_gain"]
        if gain < 1e-6:
            return np.zeros(3)
        omega = 2.0 * math.pi / carrier.heave_period
        phase = omega * t + 0.025 * gains["aft"]
        downwash = self.max_downwash_mps * gain * (1.0 + self.unsteadiness * math.sin(phase))
        headwind = self.max_headwind_mps * gain * (1.0 + 0.5 * self.unsteadiness * math.cos(phase))
        crosswind = self.max_crosswind_mps * gain * math.sin(0.8 * phase)
        deck_frame = np.array([-headwind, crosswind, downwash], dtype=float)
        c = math.cos(carrier.heading)
        s = math.sin(carrier.heading)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return rotation @ deck_frame

    def wind_body(self, state: RigidBodyState, carrier: MovingCarrier, t: float) -> np.ndarray:
        """Return airwake velocity in aircraft body axes for the dynamics model."""

        wind_nav = self.wind_nav(state.position, carrier, t)
        return quat_to_matrix(state.quaternion).T @ wind_nav

    def sample_metrics(
        self,
        positions: np.ndarray,
        carrier: MovingCarrier,
        times: np.ndarray,
    ) -> dict[str, float | bool]:
        samples = [self.wind_nav(position, carrier, float(t)) for position, t in zip(positions, times)]
        if not samples:
            return {
                "airwake_enabled": self.enabled,
                "max_airwake_downwash_mps": 0.0,
                "max_abs_airwake_mps": 0.0,
            }
        arr = np.asarray(samples, dtype=float)
        return {
            "airwake_enabled": self.enabled,
            "max_airwake_downwash_mps": float(np.max(arr[:, 2])),
            "max_abs_airwake_mps": float(np.max(np.linalg.norm(arr, axis=1))),
        }
