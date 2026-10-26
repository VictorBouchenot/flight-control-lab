"""Full-authority educational flight-control system for the 6-DOF model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .aircraft6dof import (
    Aircraft6DOFParams,
    RigidBodyState,
    SurfaceCommand,
    quat_to_matrix,
)
from .carrier6dof import MovingCarrier
from .dynamics import G, clamp, wrap_angle


@dataclass(frozen=True)
class GuidanceTargets:
    target_speed: float
    target_altitude: float
    target_heading: float
    max_bank: float
    max_vertical_speed: float = 12.0


@dataclass(frozen=True)
class FCS6DOFCommand:
    surfaces: SurfaceCommand
    target_bank: float
    target_pitch: float
    target_vertical_speed: float
    load_factor_estimate: float
    alpha_limited: bool
    beta_limited: bool


class FlightControlSystem6DOF:
    """Layered control augmentation: guidance, inner damping and protection."""

    def __init__(self, params: Aircraft6DOFParams | None = None) -> None:
        self.params = params or Aircraft6DOFParams()
        self.params.validate()
        self._last_throttle = 0.0

    def _navigation_velocity(self, state: RigidBodyState) -> np.ndarray:
        return quat_to_matrix(state.quaternion) @ state.velocity_body

    def _protected(
        self,
        state: RigidBodyState,
        surfaces: SurfaceCommand,
        load_factor_estimate: float,
    ) -> tuple[SurfaceCommand, bool, bool]:
        p = self.params
        alpha_limited = False
        beta_limited = False
        throttle = clamp(surfaces.throttle, 0.0, 1.0)
        throttle = clamp(
            throttle,
            self._last_throttle - p.throttle_rate * 0.02,
            self._last_throttle + p.throttle_rate * 0.02,
        )
        self._last_throttle = throttle
        elevator = surfaces.elevator
        aileron = surfaces.aileron
        rudder = surfaces.rudder

        if state.alpha > p.alpha_max:
            alpha_limited = True
            elevator -= 1.8 * (state.alpha - p.alpha_max)
        elif state.alpha < -p.alpha_max * 0.7:
            alpha_limited = True
            elevator -= 1.8 * (state.alpha + p.alpha_max * 0.7)
        if abs(state.beta) > p.beta_max:
            beta_limited = True
            rudder -= 0.8 * (abs(state.beta) - p.beta_max) * math.copysign(1.0, state.beta)
        if load_factor_estimate > p.max_load_factor:
            elevator -= 0.15 * (load_factor_estimate - p.max_load_factor)
        elif load_factor_estimate < p.min_load_factor:
            elevator -= 0.15 * (load_factor_estimate - p.min_load_factor)
        roll, _, _ = state.euler
        if abs(roll) > p.max_bank:
            aileron -= 1.5 * (abs(roll) - p.max_bank) * math.copysign(1.0, roll)
        return SurfaceCommand(throttle, elevator, aileron, rudder).clamped(p), alpha_limited, beta_limited

    def command(
        self,
        state: RigidBodyState,
        targets: GuidanceTargets,
        dt: float = 0.02,
    ) -> FCS6DOFCommand:
        p = self.params
        speed = max(state.airspeed, 1.0)
        roll, pitch, yaw = state.euler
        p_rate, q_rate, r_rate = (float(v) for v in state.rates_body)
        velocity_nav = self._navigation_velocity(state)
        climb_rate = -float(velocity_nav[2])
        horizontal_speed = max(1.0, float(np.linalg.norm(velocity_nav[:2])))
        gamma = math.atan2(climb_rate, horizontal_speed)

        altitude_error = targets.target_altitude - state.altitude
        target_vertical_speed = clamp(0.45 * altitude_error, -targets.max_vertical_speed, targets.max_vertical_speed)
        target_gamma = math.asin(clamp(target_vertical_speed / speed, -0.28, 0.28))
        qbar = max(1.0, 0.5 * 1.225 * speed * speed)
        cl_required = p.mass * G / (qbar * p.wing_area)
        alpha_reference = clamp(
            (cl_required - p.cl0) / p.cl_alpha,
            math.radians(-2.0),
            p.alpha_max * 0.8,
        )
        turn_lift_compensation = 0.40 * (1.0 / max(0.35, math.cos(roll)) - 1.0)
        target_pitch = clamp(target_gamma + alpha_reference + turn_lift_compensation, math.radians(-18), math.radians(16))
        heading_error = wrap_angle(targets.target_heading - yaw)
        target_bank = clamp(1.0 * heading_error, -targets.max_bank, targets.max_bank)

        p_command = clamp(1.35 * (target_bank - roll), -0.45, 0.45)
        q_command = clamp(1.75 * (target_pitch - pitch), -0.35, 0.35)
        beta_damper = clamp(-0.35 * state.beta, -0.35, 0.35)
        r_command = clamp(0.10 * heading_error - 0.45 * state.beta, -0.35, 0.35)

        aileron = 0.65 * (p_command - p_rate) - 0.28 * p_rate
        elevator = -1.15 * (q_command - q_rate) + 0.28 * q_rate
        rudder = -0.70 * (r_command - r_rate)

        # Feedforward trim elevator keeps the FCS from relying only on integrators.
        elevator += -(p.cm0 + p.cm_alpha * state.alpha) / p.cm_de
        throttle = 0.03 + 0.010 * (targets.target_speed - speed)
        load_estimate = math.cos(roll) * max(0.0, math.cos(pitch)) + (
            speed / G
        ) * q_rate * 0.0
        surfaces, alpha_limited, beta_limited = self._protected(
            state,
            SurfaceCommand(throttle, elevator, aileron, rudder),
            load_estimate,
        )
        return FCS6DOFCommand(
            surfaces,
            target_bank,
            target_pitch,
            target_vertical_speed,
            load_estimate,
            alpha_limited,
            beta_limited,
        )

    def takeoff_command(self, state: RigidBodyState, target_heading: float = 0.0) -> FCS6DOFCommand:
        return self.command(
            state,
            GuidanceTargets(
                target_speed=max(105.0, state.airspeed + 5.0),
                target_altitude=180.0,
                target_heading=target_heading,
                max_bank=math.radians(8.0),
            ),
        )

    def carrier_command(
        self,
        state: RigidBodyState,
        carrier: MovingCarrier,
        t: float,
        approach_speed: float,
    ) -> FCS6DOFCommand:
        deck = carrier.deck_state(t)
        distance_to_deck = max(0.0, deck["touchdown_x"] - state.position[0])
        flare_bias = 1.8 * max(0.0, 1.0 - distance_to_deck / 160.0)
        desired_altitude = carrier.desired_altitude(state.position[0], t) + flare_bias
        lateral_error = state.position[1] - deck["y"]
        heading_correction = clamp(-0.010 * lateral_error, math.radians(-12), math.radians(12))
        return self.command(
            state,
            GuidanceTargets(
                target_speed=approach_speed,
                target_altitude=desired_altitude,
                target_heading=deck["heading"] + heading_correction,
                max_bank=math.radians(14.0),
                max_vertical_speed=5.2,
            ),
        )
