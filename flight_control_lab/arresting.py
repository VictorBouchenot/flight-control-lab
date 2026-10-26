"""Arresting-wire and post-touchdown rollout scoring for the carrier scenario.

The model is a teaching approximation: a captured wire applies a bounded
constant deceleration, with cable payout, remaining deck and deck-roll limits
checked explicitly. It is not a model of any real arresting gear.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .aircraft6dof import RigidBodyState, quat_to_matrix
from .carrier6dof import CarrierTouchdown, MovingCarrier
from .dynamics import G


@dataclass(frozen=True)
class ArrestingGear:
    """Teaching parameters for a four-wire carrier arresting layout."""

    wire_fractions: tuple[float, float, float, float] = (0.16, 0.28, 0.44, 0.58)
    target_wire: int = 3
    engagement_window_m: float = 16.0
    lateral_capture_limit_m: float = 4.0
    max_deceleration_g: float = 2.7
    rolling_deceleration_mps2: float = 1.5
    max_cable_payout_m: float = 90.0
    max_deck_roll_deg: float = 2.2

    def validate(self) -> None:
        if len(self.wire_fractions) != 4:
            raise ValueError("the teaching deck uses four arresting wires")
        if any(f <= 0 or f >= 1 for f in self.wire_fractions):
            raise ValueError("wire fractions must stay inside the deck")
        if tuple(sorted(self.wire_fractions)) != self.wire_fractions:
            raise ValueError("wire fractions must be ordered from stern to bow")
        if not 1 <= self.target_wire <= len(self.wire_fractions):
            raise ValueError("target_wire must refer to an existing wire")
        if min(self.engagement_window_m, self.lateral_capture_limit_m) <= 0:
            raise ValueError("capture limits must be positive")
        if self.max_deceleration_g <= 0 or self.rolling_deceleration_mps2 < 0:
            raise ValueError("arresting deceleration limits are invalid")
        if self.max_cable_payout_m <= 0 or self.max_deck_roll_deg <= 0:
            raise ValueError("cable payout and deck-roll limits must be positive")


@dataclass(frozen=True)
class ArrestmentResult:
    engaged: bool
    success: bool
    reason: str
    wire_number: int
    wire_offset_m: float
    relative_longitudinal_speed_mps: float
    stop_distance_m: float
    stop_time_s: float
    remaining_deck_m: float
    stop_margin_m: float
    cable_payout_m: float
    max_deck_roll_during_rollout_deg: float
    score: float
    stop_deck_x_m: float
    stop_deck_y_m: float

    def metrics(self) -> dict[str, float | int | bool | str]:
        return {
            "arrestment_engaged": self.engaged,
            "arrestment_success": self.success,
            "arrestment_reason": self.reason,
            "arrestment_wire_number": self.wire_number,
            "arrestment_wire_offset_m": self.wire_offset_m,
            "arrestment_relative_speed_mps": self.relative_longitudinal_speed_mps,
            "arrestment_stop_distance_m": self.stop_distance_m,
            "arrestment_rollout_time_s": self.stop_time_s,
            "arrestment_remaining_deck_m": self.remaining_deck_m,
            "arrestment_stop_margin_m": self.stop_margin_m,
            "arrestment_cable_payout_m": self.cable_payout_m,
            "arrestment_max_deck_roll_deg": self.max_deck_roll_during_rollout_deg,
            "arrestment_score": self.score,
            "arrestment_stop_deck_x_m": self.stop_deck_x_m,
            "arrestment_stop_deck_y_m": self.stop_deck_y_m,
        }


def evaluate_arrestment(
    state: RigidBodyState,
    touchdown: CarrierTouchdown,
    carrier: MovingCarrier,
    t: float,
    gear: ArrestingGear | None = None,
) -> ArrestmentResult:
    """Score wire capture and the subsequent rollout on the moving deck."""

    gear = gear or ArrestingGear()
    gear.validate()
    carrier.validate()
    deck = carrier.deck_state(t)
    heading = deck["heading"]
    x_rel = touchdown.x - deck["x_start"]
    wire_positions = np.asarray(gear.wire_fractions) * carrier.length
    wire_number = int(np.argmin(np.abs(wire_positions - x_rel))) + 1
    wire_offset = float(x_rel - wire_positions[wire_number - 1])

    velocity_nav = quat_to_matrix(state.quaternion) @ state.velocity_body
    deck_direction = np.array([math.cos(heading), math.sin(heading)], dtype=float)
    relative_longitudinal_speed = float(
        np.dot(velocity_nav[:2], deck_direction) - deck["speed"]
    )
    relative_longitudinal_speed = max(0.0, relative_longitudinal_speed)

    engaged = bool(
        touchdown.success
        and abs(touchdown.lineup_error) <= gear.lateral_capture_limit_m
        and -10.0 <= wire_offset <= gear.engagement_window_m
    )
    deceleration = gear.max_deceleration_g * G + gear.rolling_deceleration_mps2
    stop_distance = relative_longitudinal_speed**2 / (2.0 * deceleration)
    stop_time = relative_longitudinal_speed / deceleration
    remaining_deck = carrier.length - x_rel
    stop_margin = remaining_deck - stop_distance
    cable_payout = stop_distance

    sample_count = max(2, int(math.ceil(stop_time / 0.05)) + 1)
    sample_times = np.linspace(t, t + stop_time, sample_count)
    max_roll = max(abs(carrier.deck_state(float(sample))["roll"]) for sample in sample_times)
    max_roll_deg = math.degrees(max_roll)

    stop_deck_x = deck["x_start"] + x_rel + stop_distance * math.cos(heading)
    stop_deck_y = deck["y"] + stop_distance * math.sin(heading)

    wire_score = max(0.0, 1.0 - abs(wire_offset - 7.0) / 18.0)
    lineup_score = max(0.0, 1.0 - abs(touchdown.lineup_error) / gear.lateral_capture_limit_m)
    deck_score = max(0.0, 1.0 - max_roll_deg / gear.max_deck_roll_deg)
    margin_score = max(0.0, min(1.0, stop_margin / max(1.0, remaining_deck)))
    score = 100.0 * (0.35 * wire_score + 0.25 * lineup_score + 0.25 * margin_score + 0.15 * deck_score)

    if not touchdown.success:
        reason = "no arrested rollout: touchdown outside teaching limits"
    elif not engaged:
        reason = "bolter: wire capture window missed"
    elif stop_distance > gear.max_cable_payout_m:
        reason = "arresting cable payout exceeded"
    elif stop_distance > remaining_deck:
        reason = "deck overrun after wire capture"
    elif max_roll_deg > gear.max_deck_roll_deg:
        reason = "deck roll outside rollout limit"
    else:
        reason = f"successful {wire_number}-wire arrestment and rollout"

    success = bool(
        touchdown.success
        and engaged
        and stop_distance <= gear.max_cable_payout_m
        and stop_distance <= remaining_deck
        and max_roll_deg <= gear.max_deck_roll_deg
    )
    return ArrestmentResult(
        engaged,
        success,
        reason,
        wire_number,
        wire_offset,
        relative_longitudinal_speed,
        stop_distance,
        stop_time,
        remaining_deck,
        stop_margin,
        cable_payout,
        max_roll_deg,
        score,
        stop_deck_x,
        stop_deck_y,
    )
