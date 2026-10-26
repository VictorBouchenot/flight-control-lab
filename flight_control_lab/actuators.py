"""Rate- and position-limited actuator model."""

from __future__ import annotations

from dataclasses import dataclass

from .aircraft6dof import Aircraft6DOFParams, SurfaceCommand
from .dynamics import clamp


@dataclass(frozen=True)
class ActuatorState:
    throttle: float = 0.0
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0

    def surfaces(self) -> SurfaceCommand:
        return SurfaceCommand(self.throttle, self.elevator, self.aileron, self.rudder)


def step_actuators(
    actual: ActuatorState,
    desired: SurfaceCommand,
    params: Aircraft6DOFParams,
    dt: float,
) -> ActuatorState:
    if dt <= 0:
        raise ValueError("dt must be positive")
    d = desired.clamped(params)
    throttle_step = params.throttle_rate * dt
    surface_step = params.surface_rate * dt
    return ActuatorState(
        actual.throttle + clamp(d.throttle - actual.throttle, -throttle_step, throttle_step),
        actual.elevator + clamp(d.elevator - actual.elevator, -surface_step, surface_step),
        actual.aileron + clamp(d.aileron - actual.aileron, -surface_step, surface_step),
        actual.rudder + clamp(d.rudder - actual.rudder, -surface_step, surface_step),
    )
