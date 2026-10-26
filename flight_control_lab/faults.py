"""Simple actuator degradation models for robustness scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from .aircraft6dof import SurfaceCommand
from .dynamics import clamp


@dataclass(frozen=True)
class ActuatorFault:
    """Scale one control channel after a chosen time."""

    surface: str = "rudder"
    start_time: float = 35.0
    effectiveness: float = 0.45
    enabled: bool = True

    def validate(self) -> None:
        if self.surface not in {"throttle", "elevator", "aileron", "rudder"}:
            raise ValueError("unknown actuator surface")
        if self.start_time < 0:
            raise ValueError("fault start time must be non-negative")
        if not 0.0 < self.effectiveness <= 1.0:
            raise ValueError("effectiveness must be in (0, 1]")

    def apply(self, command: SurfaceCommand, t: float) -> SurfaceCommand:
        self.validate()
        if not self.enabled or t < self.start_time or self.effectiveness == 1.0:
            return command
        values = {
            "throttle": command.throttle,
            "elevator": command.elevator,
            "aileron": command.aileron,
            "rudder": command.rudder,
        }
        values[self.surface] *= self.effectiveness
        return SurfaceCommand(
            clamp(values["throttle"], 0.0, 1.0),
            values["elevator"],
            values["aileron"],
            values["rudder"],
        )

    def metrics(self) -> dict[str, float | str | bool]:
        return {
            "actuator_fault_enabled": self.enabled,
            "actuator_fault_surface": self.surface,
            "actuator_fault_start_time_s": self.start_time,
            "actuator_fault_effectiveness": self.effectiveness,
        }
