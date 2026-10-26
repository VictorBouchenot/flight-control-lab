"""Scripted Pugachev-style cobra demonstration on the 6-DOF model.

The cobra is a scripted post-stall demonstration, not a maneuver the FCS is
asked to fly: a full-aft-stick pull drives the aircraft past 90 degrees of
angle of attack on the teaching post-stall aerodynamic extension, the speed
bleeds off, then the standard FCS (with its alpha protection) recovers the
aircraft to level flight. This mirrors the project's theme: spectacular
maneuver, automatic corrections afterwards.

Honesty note: the post-stall model is a flat-plate teaching approximation,
not a validated high-alpha aerodynamic database.
"""

from __future__ import annotations

import math

import numpy as np

from .actuators import ActuatorState, step_actuators
from .aircraft6dof import (
    Aircraft6DOFParams,
    RigidBodyState,
    SurfaceCommand,
    rk4_step_6dof,
    trim_level_flight,
)
from .dynamics import clamp
from .fcs6dof import FlightControlSystem6DOF, GuidanceTargets
from .scenarios6dof import SixDOFResult


def simulate_cobra_6dof(
    params: Aircraft6DOFParams | None = None,
    entry_speed: float = 135.0,
    entry_altitude: float = 3_000.0,
    pull_time: float = 1.0,
    release_alpha: float = math.radians(75.0),
    dt: float = 0.01,
    timeout: float = 70.0,
) -> SixDOFResult:
    """Fly a scripted cobra: pull, float past 90 deg alpha, push, FCS recovery.

    Phases: level entry, full-aft stick until ``release_alpha``, neutral float
    while inertia carries the nose past 90 degrees, scripted nose-down push,
    then the standard FCS recaptures level flight. Tuned so the generic model
    peaks around 100 degrees of alpha and recovers within seconds.
    """

    p = params or Aircraft6DOFParams(post_stall=True, post_stall_cm_factor=0.45)
    if not p.post_stall:
        raise ValueError("the cobra scenario requires post_stall=True parameters")
    p.validate()
    fcs = FlightControlSystem6DOF(p)
    state, trim = trim_level_flight(p, speed=entry_speed, altitude=entry_altitude, heading=0.0)
    actuators = ActuatorState(trim.throttle, trim.elevator, trim.aileron, trim.rudder)

    times = [0.0]
    states = [state]
    surfaces = [trim]
    time = 0.0
    phase = "hold"
    pull_start = pull_time
    recovery: GuidanceTargets | None = None
    peak_alpha = 0.0
    min_speed = state.airspeed
    min_altitude = state.altitude
    hold_altitude = entry_altitude
    settle_since: float | None = None

    while time < timeout:
        if phase == "hold" and time >= pull_start:
            phase = "pull"
        if phase == "pull":
            desired = SurfaceCommand(0.05, -p.max_surface, 0.0, 0.0)
            if state.alpha >= release_alpha or time - pull_start > 8.0:
                phase = "float"
        elif phase == "float":
            # stick neutral: pitch inertia carries the nose through the cobra peak
            desired = SurfaceCommand(0.05, 0.0, 0.0, 0.0)
            if state.alpha < math.radians(60.0) or time - pull_start > 19.0:
                phase = "push"
        elif phase == "push":
            # scripted nose-down: unload the post-stall airframe and let speed rebuild
            desired = SurfaceCommand(0.9, 0.7 * p.max_surface, 0.0, 0.0)
            if state.alpha < math.radians(10.0):
                phase = "recover"
                hold_altitude = state.altitude
                recovery = GuidanceTargets(
                    entry_speed, hold_altitude, 0.0, math.radians(30.0), max_vertical_speed=12.0
                )
        else:
            if phase == "hold":
                desired = trim
            else:  # recover: the standard FCS, with its protections, holds the jet level
                assert recovery is not None
                desired = fcs.command(state, recovery, dt).surfaces
        actuators = step_actuators(actuators, desired, p, dt)
        state = rk4_step_6dof(state, actuators.surfaces(), p, dt)
        time += dt
        times.append(time)
        states.append(state)
        surfaces.append(actuators.surfaces())
        peak_alpha = max(peak_alpha, state.alpha)
        if phase != "hold":
            min_speed = min(min_speed, state.airspeed)
            min_altitude = min(min_altitude, state.altitude)
        if phase != "hold" and state.altitude < 300.0:
            break  # safety floor for the demonstration
        if phase == "recover":
            # recovery is declared only after the FCS has HELD level flight
            # for a sustained window, not on a transient swing through it
            settled = (
                abs(state.alpha) < math.radians(6.0)
                and abs(state.airspeed - entry_speed) < 12.0
                and abs(state.altitude - hold_altitude) < 60.0
                and abs(state.euler[0]) < math.radians(8.0)
            )
            if settled and settle_since is None:
                settle_since = time
            elif not settled:
                settle_since = None
            if settle_since is not None and time - settle_since >= 4.0:
                break

    euler = state.euler
    recovered = (
        phase == "recover"
        and settle_since is not None
        and time - settle_since >= 4.0
    )
    metrics: dict[str, float | bool | str] = {
        "peak_alpha_deg": math.degrees(peak_alpha),
        "cobra_reached": peak_alpha >= math.radians(90.0),
        "entry_speed_mps": entry_speed,
        "min_speed_mps": min_speed,
        "final_speed_mps": state.airspeed,
        "entry_altitude_m": entry_altitude,
        "min_altitude_m": min_altitude,
        "altitude_lost_m": entry_altitude - min_altitude,
        "final_altitude_m": state.altitude,
        "recovered": recovered,
        "recovery_note": "sustained level flight: alpha<6 deg, speed within 12 m/s of entry, altitude within 60 m, held 4 s",
        "duration_s": time,
        "release_alpha_deg": math.degrees(release_alpha),
        "model_note": "flat-plate post-stall teaching approximation",
    }
    return SixDOFResult.from_states("cobra_6dof", times, states, surfaces, metrics)
