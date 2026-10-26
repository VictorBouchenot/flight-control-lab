"""Deterministic teaching pilot model for the assistance comparison.

The pilot tracks the same intent the FCS receives (glideslope altitude,
centerline, approach speed) but with human-like limitations: a reaction
delay, modest PD gains, no aircraft-rate damping, a small deterministic
tremor, and - crucially - no envelope awareness. It commands whatever its
error chase asks for, even past alpha/bank limits.

This is a teaching model of an imperfect human stick, not a human-factors
model. Its only purpose is to make the FCS contribution visible: same pilot,
same intent, same disturbances, with and without the assistance layer.
"""

from __future__ import annotations

import math

import numpy as np

from .aircraft6dof import RigidBodyState, SurfaceCommand, quat_to_matrix
from .carrier6dof import MovingCarrier
from .dynamics import clamp


class PilotModel:
    """Imperfect deterministic pilot: delay, PD chase, tremor, no protection."""

    def __init__(
        self,
        delay_s: float = 0.20,
        tremor: float = 0.012,
        elevator_gain_alt: float = 0.0022,
        elevator_gain_gamma: float = 1.1,
        aileron_gain_bank: float = 1.5,
        bank_gain_lateral: float = 0.016,
        throttle_gain: float = 0.012,
    ) -> None:
        if delay_s < 0:
            raise ValueError("delay must be non-negative")
        self.delay_s = delay_s
        self.tremor = tremor
        self.k_e_alt = elevator_gain_alt
        self.k_e_gamma = elevator_gain_gamma
        self.k_a_bank = aileron_gain_bank
        self.k_bank_lat = bank_gain_lateral
        self.k_throttle = throttle_gain
        self._buffer: list[tuple[float, SurfaceCommand]] = []
        self._pending = SurfaceCommand(0.3, 0.0, 0.0, 0.0)
        # Slow trim integrator: real pilots trim the aircraft; without it a
        # pure proportional stick holds a large steady-state error.
        self._trim_elevator = 0.0
        # Per-second gain (integrated with the caller's dt, not per call).
        self._trim_integrator_gain = 8.0e-4
        self._trim_limit = 0.20
        self._trim_throttle = 0.5
        self._last_t: float | None = None
        self._commands: list[SurfaceCommand] = []

    def warm_start(self, trim: SurfaceCommand) -> None:
        """Begin on the trim stick position, like a pilot already established."""

        self._pending = trim
        self._trim_elevator = trim.elevator
        self._trim_throttle = trim.throttle

    def _integrate_trim(self, alt_error: float, t: float) -> None:
        dt_step = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        self._trim_elevator = clamp(
            self._trim_elevator - self._trim_integrator_gain * alt_error * dt_step,
            -self._trim_limit,
            self._trim_limit,
        )

    def _delayed(self, desired: SurfaceCommand, t: float) -> SurfaceCommand:
        self._buffer.append((t + self.delay_s, desired))
        while self._buffer and self._buffer[0][0] <= t:
            _, self._pending = self._buffer.pop(0)
        return self._pending

    def carrier_command(
        self,
        state: RigidBodyState,
        carrier: MovingCarrier,
        t: float,
        approach_speed: float,
    ) -> SurfaceCommand:
        """Raw stick for the carrier approach: chase glideslope and centerline."""

        deck = carrier.deck_state(t)
        velocity_nav = quat_to_matrix(state.quaternion) @ state.velocity_body
        climb_rate = -float(velocity_nav[2])
        horizontal_speed = max(1.0, float(np.linalg.norm(velocity_nav[:2])))
        gamma = math.atan2(climb_rate, horizontal_speed)

        desired_altitude = carrier.desired_altitude(state.position[0], t)
        alt_error = desired_altitude - state.altitude
        # The pilot knows the glideslope and aims to fly down it, correcting
        # for slope error on top of the nominal descent rate.
        glide_vs = -horizontal_speed * math.tan(math.radians(carrier.glide_slope_degrees))
        target_vs = clamp(glide_vs + 0.5 * alt_error, -7.0, 7.0)
        target_gamma = math.atan2(target_vs, horizontal_speed)
        gamma_error = target_gamma - gamma

        # The pilot pulls whatever the error chase asks: no alpha/load awareness.
        self._integrate_trim(alt_error, t)
        elevator = (
            self._trim_elevator
            - self.k_e_alt * alt_error
            - self.k_e_gamma * gamma_error
            - 0.30 * float(state.rates_body[1])
        )
        lateral_error = state.position[1] - deck["y"]
        bank_target = clamp(-self.k_bank_lat * lateral_error, math.radians(-35.0), math.radians(35.0))
        roll, _, _ = state.euler
        # Weak rate sense: humans feel rotation imperfectly, so only a light
        # damping term - not enough to prevent pilot-induced oscillation.
        aileron = self.k_a_bank * (bank_target - roll) - 0.35 * float(state.rates_body[0])
        rudder = -0.5 * state.beta
        throttle = 0.03 + self.k_throttle * (approach_speed - state.airspeed)

        tremor_e = self.tremor * math.sin(2.0 * math.pi * 1.3 * t)
        tremor_a = self.tremor * math.sin(2.0 * math.pi * 1.7 * t + 1.1)
        desired = SurfaceCommand(
            clamp(throttle, 0.0, 1.0),
            clamp(elevator + tremor_e, math.radians(-25.0), math.radians(25.0)),
            clamp(aileron + tremor_a, math.radians(-25.0), math.radians(25.0)),
            clamp(rudder, math.radians(-20.0), math.radians(20.0)),
        )
        issued = self._delayed(desired, t)
        self._commands.append(issued)
        return issued

    def maneuver_command(
        self,
        state: RigidBodyState,
        target_heading: float,
        t: float,
        target_speed: float = 138.0,
        target_altitude: float = 2_000.0,
    ) -> SurfaceCommand:
        """Raw stick for a heading capture: aggressive bank chase, no protection."""

        roll, _, yaw = state.euler
        heading_error = (target_heading - yaw + math.pi) % (2.0 * math.pi) - math.pi
        # A student banks hard toward the target heading - more than the FCS's 45 deg cap.
        bank_target = clamp(0.9 * heading_error, math.radians(-55.0), math.radians(55.0))
        aileron = self.k_a_bank * (bank_target - roll) - 0.35 * float(state.rates_body[0])

        velocity_nav = quat_to_matrix(state.quaternion) @ state.velocity_body
        climb_rate = -float(velocity_nav[2])
        horizontal_speed = max(1.0, float(np.linalg.norm(velocity_nav[:2])))
        gamma = math.atan2(climb_rate, horizontal_speed)
        alt_error = target_altitude - state.altitude
        target_vs = clamp(0.6 * alt_error, -9.0, 9.0)
        target_gamma = math.atan2(target_vs, horizontal_speed)
        gamma_error = target_gamma - gamma
        # No bank-to-load awareness: the pilot does not pull more in the turn,
        # so the nose drops as the bank builds.
        self._integrate_trim(alt_error, t)
        elevator = (
            self._trim_elevator
            - self.k_e_alt * alt_error
            - self.k_e_gamma * gamma_error
            - 0.30 * float(state.rates_body[1])
        )
        rudder = -0.5 * state.beta
        throttle = self._trim_throttle + self.k_throttle * (target_speed - state.airspeed)

        tremor_e = self.tremor * math.sin(2.0 * math.pi * 1.3 * t)
        tremor_a = self.tremor * math.sin(2.0 * math.pi * 1.7 * t + 1.1)
        desired = SurfaceCommand(
            clamp(throttle, 0.0, 1.0),
            clamp(elevator + tremor_e, math.radians(-25.0), math.radians(25.0)),
            clamp(aileron + tremor_a, math.radians(-25.0), math.radians(25.0)),
            clamp(rudder, math.radians(-20.0), math.radians(20.0)),
        )
        issued = self._delayed(desired, t)
        self._commands.append(issued)
        return issued

    def stick_history(self) -> dict[str, list[float]]:
        """Issued stick commands, one per call, for stick-vs-surface plots."""

        return {
            "throttle": [c.throttle for c in self._commands],
            "elevator": [c.elevator for c in self._commands],
            "aileron": [c.aileron for c in self._commands],
            "rudder": [c.rudder for c in self._commands],
        }
