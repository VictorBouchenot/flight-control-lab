"""Moving carrier model with periodic deck motion."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .aircraft6dof import RigidBodyState
from .dynamics import clamp


@dataclass(frozen=True)
class MovingCarrier:
    """Carrier moving along +x with sinusoidal heave, pitch and roll."""

    x_start0: float = 4_500.0
    y: float = 0.0
    deck_height: float = 18.0
    length: float = 190.0
    width: float = 32.0
    speed: float = 12.0
    heading: float = 0.0
    glide_slope_degrees: float = 3.5
    touchdown_fraction: float = 0.45
    heave_amplitude: float = 1.1
    heave_period: float = 9.0
    pitch_amplitude_deg: float = 0.7
    pitch_period: float = 11.0
    roll_amplitude_deg: float = 1.2
    roll_period: float = 7.5

    def validate(self) -> None:
        if min(self.deck_height, self.length, self.width) <= 0:
            raise ValueError("deck geometry must be positive")
        if not 0 <= self.speed <= 25:
            raise ValueError("carrier speed outside teaching range")
        if not 0.5 <= self.glide_slope_degrees <= 8:
            raise ValueError("glide slope outside teaching range")
        if not 0 < self.touchdown_fraction < 1:
            raise ValueError("touchdown_fraction must be between 0 and 1")
        if min(self.heave_period, self.pitch_period, self.roll_period) <= 0:
            raise ValueError("motion periods must be positive")

    def x_start(self, t: float) -> float:
        return self.x_start0 + self.speed * t

    def touchdown_x(self, t: float) -> float:
        return self.x_start(t) + self.touchdown_fraction * self.length

    def deck_state(self, t: float) -> dict[str, float]:
        self.validate()
        omega_h = 2 * math.pi / self.heave_period
        omega_p = 2 * math.pi / self.pitch_period
        omega_r = 2 * math.pi / self.roll_period
        heave = self.heave_amplitude * math.sin(omega_h * t)
        heave_rate = self.heave_amplitude * omega_h * math.cos(omega_h * t)
        pitch = math.radians(self.pitch_amplitude_deg) * math.sin(omega_p * t)
        roll = math.radians(self.roll_amplitude_deg) * math.sin(omega_r * t)
        return {
            "x_start": self.x_start(t),
            "touchdown_x": self.touchdown_x(t),
            "y": self.y,
            "z_down": -self.deck_height - heave,
            "heave": heave,
            "heave_rate_up": heave_rate,
            "pitch": pitch,
            "roll": roll,
            "heading": self.heading,
            "speed": self.speed,
        }

    def desired_altitude(self, x: float, t: float) -> float:
        deck = self.deck_state(t)
        distance = max(0.0, deck["touchdown_x"] - x)
        return -deck["z_down"] + math.tan(math.radians(self.glide_slope_degrees)) * distance

    def deck_plane_z_down(self, x: float, y: float, t: float) -> float:
        deck = self.deck_state(t)
        # Small-angle plane around the touchdown point. NED z is down.
        return (
            deck["z_down"]
            + deck["pitch"] * (x - deck["touchdown_x"])
            - deck["roll"] * (y - self.y)
        )


@dataclass(frozen=True)
class CarrierTouchdown:
    touched: bool
    success: bool
    reason: str
    x: float
    y: float
    deck_x: float
    relative_speed: float
    relative_sink_rate: float
    lineup_error: float
    deck_roll_deg: float
    deck_pitch_deg: float


def assess_moving_touchdown(
    state: RigidBodyState,
    carrier: MovingCarrier,
    t: float,
    approach_speed: float,
    max_sink_rate: float = 4.5,
) -> CarrierTouchdown:
    carrier.validate()
    deck = carrier.deck_state(t)
    x_rel = state.position[0] - deck["x_start"]
    y_rel = state.position[1] - deck["y"]
    from .aircraft6dof import quat_to_matrix

    velocity_nav = quat_to_matrix(state.quaternion) @ state.velocity_body
    deck_vx = deck["speed"] * math.cos(deck["heading"])
    relative_speed = float(np.linalg.norm(velocity_nav[:2] - np.array([deck_vx, 0.0])))
    relative_sink = float(velocity_nav[2] + deck["heave_rate_up"])
    in_length = 0.0 <= x_rel <= carrier.length
    in_width = abs(y_rel) <= carrier.width / 2.0
    speed_ok = approach_speed - carrier.speed - 10 <= relative_speed <= approach_speed - carrier.speed + 12
    sink_ok = relative_sink <= max_sink_rate
    deck_angle_ok = abs(deck["roll"]) <= math.radians(2.5) and abs(deck["pitch"]) <= math.radians(2.0)
    success = in_length and in_width and speed_ok and sink_ok and deck_angle_ok

    if success:
        reason = "within moving-deck touchdown, lineup, speed, sink and deck-motion limits"
    elif not in_length:
        reason = "outside moving deck length"
    elif not in_width:
        reason = "outside moving deck width"
    elif not speed_ok:
        reason = "relative speed outside approach window"
    elif not sink_ok:
        reason = "relative sink rate above teaching limit"
    else:
        reason = "deck motion outside teaching touchdown limits"

    return CarrierTouchdown(
        True,
        success,
        reason,
        float(state.position[0]),
        float(state.position[1]),
        float(deck["touchdown_x"]),
        relative_speed,
        relative_sink,
        float(y_rel),
        math.degrees(deck["roll"]),
        math.degrees(deck["pitch"]),
    )
