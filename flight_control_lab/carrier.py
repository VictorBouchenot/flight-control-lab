"""Fixed carrier-deck geometry and landing assessment."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .dynamics import PointMassState


@dataclass(frozen=True)
class CarrierDeck:
    """A fixed, simplified carrier landing area."""

    x_start: float = 4_000.0
    length: float = 180.0
    width: float = 30.0
    altitude: float = 18.0
    centerline_y: float = 0.0
    glide_slope_degrees: float = 3.5
    touchdown_x_fraction: float = 0.45

    def validate(self) -> None:
        if self.length <= 0 or self.width <= 0:
            raise ValueError("deck dimensions must be positive")
        if self.altitude < 0:
            raise ValueError("deck altitude cannot be negative")
        if not 0.5 <= self.glide_slope_degrees <= 8.0:
            raise ValueError("glide slope must be between 0.5 and 8 degrees")
        if not 0.0 < self.touchdown_x_fraction < 1.0:
            raise ValueError("touchdown fraction must be between 0 and 1")

    @property
    def touchdown_x(self) -> float:
        return self.x_start + self.touchdown_x_fraction * self.length

    def desired_altitude(self, x: float) -> float:
        self.validate()
        distance = max(0.0, self.touchdown_x - x)
        return self.altitude + math.tan(math.radians(self.glide_slope_degrees)) * distance

    def lateral_error(self, y: float) -> float:
        return y - self.centerline_y


@dataclass(frozen=True)
class TouchdownResult:
    touched: bool
    success: bool
    reason: str
    x: float
    y: float
    speed: float
    sink_rate: float
    glidepath_error_m: float
    lineup_error_m: float


def assess_touchdown(
    state: PointMassState,
    deck: CarrierDeck,
    approach_speed: float,
    max_sink_rate: float = 4.5,
) -> TouchdownResult:
    """Classify a state that has reached or crossed deck height."""

    deck.validate()
    _, _, vz = state.velocity
    sink_rate = max(0.0, -vz)
    glidepath_error = state.z - deck.desired_altitude(state.x)
    lineup_error = deck.lateral_error(state.y)
    in_length = deck.x_start <= state.x <= deck.x_start + deck.length
    in_width = abs(lineup_error) <= deck.width / 2.0
    speed_ok = approach_speed - 8.0 <= state.speed <= approach_speed + 10.0
    sink_ok = sink_rate <= max_sink_rate
    success = in_length and in_width and speed_ok and sink_ok

    if success:
        reason = "on deck, on centerline, within speed and sink limits"
    elif not in_length:
        reason = "touchdown point outside deck length"
    elif not in_width:
        reason = "touchdown point outside deck width"
    elif not speed_ok:
        reason = "touchdown speed outside approach window"
    else:
        reason = "sink rate above teaching limit"

    return TouchdownResult(
        touched=True,
        success=success,
        reason=reason,
        x=state.x,
        y=state.y,
        speed=state.speed,
        sink_rate=sink_rate,
        glidepath_error_m=glidepath_error,
        lineup_error_m=lineup_error,
    )
