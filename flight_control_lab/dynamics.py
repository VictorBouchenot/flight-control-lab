"""Simplified 3-D point-mass aircraft dynamics.

State convention:
    x, y: horizontal position in metres
    z: altitude above sea level in metres (positive upward)
    V: true airspeed in m/s
    gamma: flight-path angle in radians (positive climb)
    chi: heading in radians, with 0 pointing along +x

This is a teaching model, not an aerodynamic database or flight-certified model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

G = 9.80665
RHO_SEA_LEVEL = 1.225


@dataclass(frozen=True)
class AircraftParams:
    """Generic fighter-like teaching parameters."""

    mass: float = 12_000.0
    wing_area: float = 28.0
    max_thrust: float = 150_000.0
    cd0: float = 0.025
    induced_drag_k: float = 0.075
    min_speed: float = 65.0
    max_speed: float = 340.0
    min_load_factor: float = -2.5
    max_load_factor: float = 7.0
    max_bank: float = math.radians(75.0)
    max_throttle_rate: float = 1.8  # full throttle change per second

    def validate(self) -> None:
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if self.wing_area <= 0:
            raise ValueError("wing_area must be positive")
        if self.max_thrust <= 0:
            raise ValueError("max_thrust must be positive")
        if self.cd0 < 0 or self.induced_drag_k < 0:
            raise ValueError("drag coefficients must be non-negative")
        if not 1 < self.min_speed < self.max_speed:
            raise ValueError("speed limits are inconsistent")
        if self.min_load_factor >= self.max_load_factor:
            raise ValueError("load-factor limits are inconsistent")
        if not 0 < self.max_bank < math.pi / 2:
            raise ValueError("max_bank must be between 0 and 90 degrees")
        if self.max_throttle_rate <= 0:
            raise ValueError("max_throttle_rate must be positive")


@dataclass(frozen=True)
class PointMassState:
    x: float
    y: float
    z: float
    speed: float
    gamma: float
    heading: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.z, self.speed, self.gamma, self.heading],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "PointMassState":
        arr = np.asarray(values, dtype=float)
        if arr.shape != (6,):
            raise ValueError("state array must have exactly six values")
        return cls(*arr.tolist())

    @property
    def velocity(self) -> tuple[float, float, float]:
        horizontal = self.speed * math.cos(self.gamma)
        return (
            horizontal * math.cos(self.heading),
            horizontal * math.sin(self.heading),
            self.speed * math.sin(self.gamma),
        )


@dataclass(frozen=True)
class RawControl:
    """Direct point-mass control input."""

    throttle: float
    load_factor: float
    bank: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def air_density(altitude: float) -> float:
    """ISA exponential approximation, bounded for the teaching model."""

    return RHO_SEA_LEVEL * math.exp(-max(0.0, altitude) / 9_500.0)


def dynamic_pressure(altitude: float, speed: float) -> float:
    return 0.5 * air_density(altitude) * speed * speed


def drag_force(state: PointMassState, params: AircraftParams, load_factor: float) -> float:
    q = dynamic_pressure(state.z, state.speed)
    if q <= 0:
        return 0.0
    lift = abs(load_factor) * params.mass * G
    cl = lift / (q * params.wing_area)
    cd = params.cd0 + params.induced_drag_k * cl * cl
    return q * params.wing_area * cd


def derivatives(
    state: PointMassState,
    control: RawControl,
    params: AircraftParams,
) -> np.ndarray:
    """Return state derivatives for the point-mass model."""

    params.validate()
    if state.speed <= 0:
        raise ValueError("speed must be positive in flight")

    throttle = clamp(control.throttle, 0.0, 1.0)
    nz = clamp(control.load_factor, params.min_load_factor, params.max_load_factor)
    bank = clamp(control.bank, -params.max_bank, params.max_bank)

    # Keep the flight-path angle in a numerically sensible range. The controller
    # stays well inside it; this only protects direct API callers.
    gamma = clamp(state.gamma, math.radians(-85.0), math.radians(85.0))
    drag = drag_force(state, params, nz)
    thrust = throttle * params.max_thrust

    dv = (thrust - drag) / params.mass - G * math.sin(gamma)
    dgamma = (G / state.speed) * (nz * math.cos(bank) - math.cos(gamma))
    dheading = (G * nz * math.sin(bank)) / (state.speed * math.cos(gamma))
    horizontal = state.speed * math.cos(gamma)

    return np.array(
        [
            horizontal * math.cos(state.heading),
            horizontal * math.sin(state.heading),
            state.speed * math.sin(gamma),
            dv,
            dgamma,
            dheading,
        ],
        dtype=float,
    )


def rk4_step(
    state: PointMassState,
    control: RawControl,
    params: AircraftParams,
    dt: float,
) -> PointMassState:
    """Advance one deterministic RK4 integration step."""

    if dt <= 0:
        raise ValueError("dt must be positive")
    y0 = state.as_array()
    k1 = derivatives(state, control, params)
    k2 = derivatives(PointMassState.from_array(y0 + 0.5 * dt * k1), control, params)
    k3 = derivatives(PointMassState.from_array(y0 + 0.5 * dt * k2), control, params)
    k4 = derivatives(PointMassState.from_array(y0 + dt * k3), control, params)
    return PointMassState.from_array(y0 + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (angle + math.pi) % (2.0 * math.pi) - math.pi
