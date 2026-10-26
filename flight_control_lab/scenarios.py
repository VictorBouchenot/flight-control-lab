"""Runnable educational flight-control scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .carrier import CarrierDeck, TouchdownResult, assess_touchdown
from .control import ControlCommand, FlightControlSystem
from .dynamics import (
    AircraftParams,
    PointMassState,
    RawControl,
    clamp,
    drag_force,
    rk4_step,
    G,
)


@dataclass(frozen=True)
class SimulationResult:
    name: str
    times: np.ndarray
    states: np.ndarray
    commands: np.ndarray
    metrics: dict[str, float | bool | str]

    def final_state(self) -> PointMassState:
        return PointMassState.from_array(self.states[-1])


def _record_result(
    name: str,
    times: list[float],
    states: list[PointMassState],
    commands: list[ControlCommand | RawControl],
    metrics: dict[str, float | bool | str],
) -> SimulationResult:
    command_rows = []
    for command in commands:
        if isinstance(command, ControlCommand):
            command_rows.append(
                [command.throttle, command.load_factor, command.bank]
            )
        else:
            command_rows.append(
                [command.throttle, command.load_factor, command.bank]
            )
    return SimulationResult(
        name=name,
        times=np.asarray(times, dtype=float),
        states=np.asarray([state.as_array() for state in states], dtype=float),
        commands=np.asarray(command_rows, dtype=float),
        metrics=metrics,
    )


def simulate_takeoff(
    params: AircraftParams | None = None,
    dt: float = 0.02,
    timeout: float = 45.0,
) -> SimulationResult:
    """Ground roll, rotation at teaching speed, then capture 150 m altitude."""

    p = params or AircraftParams()
    p.validate()
    if dt <= 0 or timeout <= 0:
        raise ValueError("dt and timeout must be positive")

    fcs = FlightControlSystem(p)
    x = 0.0
    speed = 0.0
    time = 0.0
    rotation_time: float | None = None
    rotation_distance: float | None = None
    rolling_friction = 0.025 * p.mass * G
    rotation_speed = 72.0

    times = [0.0]
    states = [PointMassState(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)]
    commands: list[ControlCommand | RawControl] = [RawControl(1.0, 1.0, 0.0)]

    # Ground roll: one-dimensional, with aerodynamic drag and rolling friction.
    while speed < rotation_speed and time < timeout:
        q_drag = 0.5 * 1.225 * speed * speed * p.wing_area * p.cd0
        acceleration = max(0.0, (p.max_thrust - q_drag - rolling_friction) / p.mass)
        speed += acceleration * dt
        x += speed * dt
        time += dt
        times.append(time)
        states.append(PointMassState(x, 0.0, 0.0, max(speed, 1.0), 0.0, 0.0))
        commands.append(RawControl(1.0, 1.0, 0.0))

    if speed < rotation_speed:
        return _record_result(
            "takeoff", times, states, commands, {"rotated": False, "failure": "timeout during ground roll"}
        )

    rotation_time = time
    rotation_distance = x
    state = PointMassState(x, 0.0, 0.5, speed, math.radians(2.0), 0.0)

    while time < timeout and state.z < 150.0:
        command = fcs.takeoff_rotation_command(state)
        state = rk4_step(state, command.raw(), p, dt)
        if state.z < 0.0:
            state = PointMassState(state.x, state.y, 0.0, state.speed, max(0.0, state.gamma), state.heading)
        time += dt
        times.append(time)
        states.append(state)
        commands.append(command)

    # Altitude capture demonstrates that the aid hands over to a stable climb/hold.
    while time < timeout and abs(state.z - 150.0) > 2.0:
        command = fcs.altitude_hold_command(state, 150.0, 105.0, 0.0)
        state = rk4_step(state, command.raw(), p, dt)
        time += dt
        times.append(time)
        states.append(state)
        commands.append(command)

    final = state
    metrics: dict[str, float | bool | str] = {
        "rotated": rotation_time is not None,
        "rotation_time_s": float(rotation_time),
        "rotation_distance_m": float(rotation_distance),
        "target_altitude_reached": state.z >= 148.0,
        "final_altitude_m": final.z,
        "final_speed_mps": final.speed,
        "duration_s": time,
    }
    return _record_result("takeoff", times, states, commands, metrics)


def simulate_maneuver_assist(
    params: AircraftParams | None = None,
    dt: float = 0.02,
    timeout: float = 35.0,
) -> SimulationResult:
    """Capture a 120-degree heading change with protected bank/load limits."""

    p = params or AircraftParams()
    fcs = FlightControlSystem(p)
    state = PointMassState(0.0, 0.0, 2_000.0, 160.0, 0.0, 0.0)
    target_heading = math.radians(120.0)
    times = [0.0]
    states = [state]
    commands: list[ControlCommand] = []
    time = 0.0

    while time < timeout:
        command = fcs.altitude_hold_command(state, 2_000.0, 165.0, target_heading)
        state = rk4_step(state, command.raw(), p, dt)
        time += dt
        times.append(time)
        states.append(state)
        commands.append(command)
        heading_error = abs(((target_heading - state.heading + math.pi) % (2 * math.pi)) - math.pi)
        if heading_error < math.radians(1.0) and time > 5.0:
            break

    heading_error = abs(((target_heading - state.heading + math.pi) % (2 * math.pi)) - math.pi)
    command_arr = np.asarray([[c.throttle, c.load_factor, c.bank] for c in commands])
    metrics: dict[str, float | bool | str] = {
        "heading_captured": heading_error < math.radians(2.0),
        "heading_error_deg": math.degrees(heading_error),
        "max_abs_bank_deg": math.degrees(float(np.max(np.abs(command_arr[:, 2])))),
        "max_load_factor": float(np.max(command_arr[:, 1])),
        "min_load_factor": float(np.min(command_arr[:, 1])),
        "final_altitude_m": state.z,
        "duration_s": time,
    }
    return _record_result("maneuver", times, states, commands, metrics)


def simulate_carrier_approach(
    params: AircraftParams | None = None,
    deck: CarrierDeck | None = None,
    dt: float = 0.02,
    timeout: float = 75.0,
    initial_lateral_offset: float = 20.0,
    initial_altitude_error: float = 10.0,
) -> tuple[SimulationResult, TouchdownResult]:
    """Fly a fixed-carrier approach with line-up and glidepath corrections."""

    p = params or AircraftParams()
    d = deck or CarrierDeck()
    p.validate()
    d.validate()
    if dt <= 0 or timeout <= 0:
        raise ValueError("dt and timeout must be positive")

    fcs = FlightControlSystem(p)
    approach_speed = 74.0
    start_x = 0.0
    start_altitude = d.desired_altitude(start_x) + initial_altitude_error
    state = PointMassState(
        start_x,
        initial_lateral_offset,
        start_altitude,
        approach_speed + 4.0,
        math.radians(-2.0),
        0.0,
    )
    times = [0.0]
    states = [state]
    commands: list[ControlCommand] = []
    time = 0.0
    touchdown: TouchdownResult | None = None

    while time < timeout:
        command = fcs.carrier_approach_command(
            state,
            desired_altitude=d.desired_altitude(state.x),
            lateral_error=d.lateral_error(state.y),
            approach_speed=approach_speed,
        )
        next_state = rk4_step(state, command.raw(), p, dt)
        time += dt
        times.append(time)
        states.append(next_state)
        commands.append(command)
        if next_state.z <= d.altitude:
            touchdown = assess_touchdown(next_state, d, approach_speed)
            state = next_state
            break
        state = next_state

    if touchdown is None:
        final_altitude = state.z
        touchdown = TouchdownResult(
            touched=False,
            success=False,
            reason="timeout before deck intercept",
            x=state.x,
            y=state.y,
            speed=state.speed,
            sink_rate=max(0.0, -state.velocity[2]),
            glidepath_error_m=final_altitude - d.desired_altitude(state.x),
            lineup_error_m=d.lateral_error(state.y),
        )

    alt_errors = np.asarray(
        [s.z - d.desired_altitude(s.x) for s in states if s.x <= d.touchdown_x],
        dtype=float,
    )
    line_errors = np.asarray([d.lateral_error(s.y) for s in states], dtype=float)
    metrics: dict[str, float | bool | str] = {
        "touchdown": touchdown.touched,
        "success": touchdown.success,
        "reason": touchdown.reason,
        "touchdown_x_m": touchdown.x,
        "touchdown_y_m": touchdown.y,
        "touchdown_speed_mps": touchdown.speed,
        "sink_rate_mps": touchdown.sink_rate,
        "max_abs_glidepath_error_m": float(np.max(np.abs(alt_errors))) if alt_errors.size else 0.0,
        "final_lineup_error_m": float(line_errors[-1]),
        "duration_s": time,
    }
    return _record_result("carrier", times, states, commands, metrics), touchdown
