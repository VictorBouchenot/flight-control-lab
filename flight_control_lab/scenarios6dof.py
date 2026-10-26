"""Full 6-DOF scenarios: takeoff, manoeuvre assist and moving-carrier approach."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .actuators import ActuatorState, step_actuators
from .airwake import CarrierAirwake
from .arresting import ArrestingGear, evaluate_arrestment
from .aircraft6dof import (
    Aircraft6DOFParams,
    RigidBodyState,
    SurfaceCommand,
    euler_to_quat,
    rk4_step_6dof,
    trim_level_flight,
)
from .carrier6dof import CarrierTouchdown, MovingCarrier, assess_moving_touchdown
from .dynamics import G, clamp
from .faults import ActuatorFault
from .fcs6dof import FlightControlSystem6DOF
from .landing_score import landing_quality_metrics
from .turbulence import StochasticTurbulence


@dataclass(frozen=True)
class SixDOFResult:
    name: str
    times: np.ndarray
    positions: np.ndarray
    velocities_body: np.ndarray
    quaternions: np.ndarray
    rates_body: np.ndarray
    surface_history: np.ndarray
    metrics: dict[str, float | bool | str]

    @classmethod
    def from_states(
        cls,
        name: str,
        times: list[float],
        states: list[RigidBodyState],
        surfaces: list[SurfaceCommand],
        metrics: dict[str, float | bool | str],
    ) -> "SixDOFResult":
        return cls(
            name,
            np.asarray(times, dtype=float),
            np.asarray([s.position for s in states], dtype=float),
            np.asarray([s.velocity_body for s in states], dtype=float),
            np.asarray([s.quaternion for s in states], dtype=float),
            np.asarray([s.rates_body for s in states], dtype=float),
            np.asarray(
                [[s.throttle, s.elevator, s.aileron, s.rudder] for s in surfaces],
                dtype=float,
            ),
            metrics,
        )


def simulate_takeoff_6dof(
    params: Aircraft6DOFParams | None = None,
    dt: float = 0.02,
    timeout: float = 50.0,
) -> SixDOFResult:
    """Runway acceleration, rotation and full-6DOF climb capture."""

    p = params or Aircraft6DOFParams()
    p.validate()
    fcs = FlightControlSystem6DOF(p)
    time = 0.0
    x = 0.0
    speed = 0.0
    rotation_speed = 76.0
    rolling_friction = 0.025 * p.mass * G
    states: list[RigidBodyState] = []
    surfaces_history: list[SurfaceCommand] = []
    times: list[float] = []

    while speed < rotation_speed and time < timeout:
        drag = 0.5 * 1.225 * speed * speed * p.wing_area * p.cd0
        acceleration = max(0.0, (p.max_thrust - drag - rolling_friction) / p.mass)
        speed += acceleration * dt
        x += speed * dt
        time += dt
        state = RigidBodyState(
            np.array([x, 0.0, 0.0]),
            np.array([max(speed, 1.0), 0.0, 0.0]),
            euler_to_quat(0.0, 0.0, 0.0),
            np.zeros(3),
        )
        states.append(state)
        surfaces_history.append(SurfaceCommand(1.0, 0.0, 0.0, 0.0))
        times.append(time)

    rotation_time = time
    rotation_distance = x
    alpha = math.radians(4.0)
    state = RigidBodyState(
        np.array([x, 0.0, -0.8]),
        np.array([speed * math.cos(alpha), 0.0, speed * math.sin(alpha)]),
        euler_to_quat(0.0, alpha, 0.0),
        np.zeros(3),
    )
    actuators = ActuatorState(1.0, 0.0, 0.0, 0.0)

    while time < timeout and (state.altitude < 175.0 or abs(state.altitude - 180.0) > 2.0):
        command = fcs.takeoff_command(state)
        actuators = step_actuators(actuators, command.surfaces, p, dt)
        state = rk4_step_6dof(state, actuators.surfaces(), p, dt)
        time += dt
        states.append(state)
        surfaces_history.append(actuators.surfaces())
        times.append(time)

    metrics: dict[str, float | bool | str] = {
        "rotated": rotation_time < timeout,
        "rotation_time_s": rotation_time,
        "rotation_distance_m": rotation_distance,
        "target_altitude_reached": state.altitude >= 175.0,
        "final_altitude_m": state.altitude,
        "final_speed_mps": state.airspeed,
        "duration_s": time,
    }
    return SixDOFResult.from_states("takeoff_6dof", times, states, surfaces_history, metrics)


def simulate_maneuver_6dof(
    params: Aircraft6DOFParams | None = None,
    dt: float = 0.02,
    timeout: float = 65.0,
    controller: object | None = None,
) -> SixDOFResult:
    """Heading capture at altitude with envelope-protected roll and pitch.

    ``controller`` optionally replaces the FCS: a callable
    ``(state, target_heading, t) -> SurfaceCommand`` used to fly the same
    capture with a different command source (e.g. the raw pilot model in the
    assistance comparison).
    """

    p = params or Aircraft6DOFParams()
    state, trim = trim_level_flight(p, speed=135.0, altitude=2_000.0)
    fcs = FlightControlSystem6DOF(p)
    actuators = ActuatorState(trim.throttle, trim.elevator, trim.aileron, trim.rudder)
    if controller is not None and hasattr(controller, "warm_start"):
        controller.warm_start(trim)
    times = [0.0]
    states = [state]
    surfaces = [trim]
    time = 0.0
    target_heading = math.radians(120.0)
    heading_hold = 0.0

    from .fcs6dof import GuidanceTargets

    while time < timeout:
        if controller is None:
            desired_surfaces = fcs.command(
                state,
                GuidanceTargets(138.0, 2_000.0, target_heading, math.radians(45.0)),
                dt,
            ).surfaces
        else:
            desired_surfaces = controller(state, target_heading, time)
        actuators = step_actuators(actuators, desired_surfaces, p, dt)
        state = rk4_step_6dof(state, actuators.surfaces(), p, dt)
        time += dt
        times.append(time)
        states.append(state)
        surfaces.append(actuators.surfaces())
        yaw = state.euler[2]
        # A capture must persist: crossing the target heading mid-oscillation
        # (as a departing raw pilot does) does not count.
        if abs((target_heading - yaw + math.pi) % (2 * math.pi) - math.pi) < math.radians(1.5):
            heading_hold += dt
        else:
            heading_hold = 0.0
        if heading_hold >= 2.0 and time > 8:
            break

    yaw = state.euler[2]
    heading_error = abs((target_heading - yaw + math.pi) % (2 * math.pi) - math.pi)
    surf = np.asarray([[s.throttle, s.elevator, s.aileron, s.rudder] for s in surfaces])
    metrics: dict[str, float | bool | str] = {
        "heading_captured": heading_error < math.radians(3.0),
        "heading_error_deg": math.degrees(heading_error),
        "final_altitude_m": state.altitude,
        "max_aileron_deg": math.degrees(float(np.max(np.abs(surf[:, 2])))),
        "max_elevator_deg": math.degrees(float(np.max(np.abs(surf[:, 1])))),
        "max_alpha_deg": math.degrees(max(s.alpha for s in states)),
        "max_abs_beta_deg": math.degrees(max(abs(s.beta) for s in states)),
        "duration_s": time,
    }
    return SixDOFResult.from_states("maneuver_6dof", times, states, surfaces, metrics)


def simulate_moving_carrier_6dof(
    params: Aircraft6DOFParams | None = None,
    carrier: MovingCarrier | None = None,
    airwake: CarrierAirwake | None = None,
    arresting_gear: ArrestingGear | None = None,
    turbulence: StochasticTurbulence | None = None,
    actuator_fault: ActuatorFault | None = None,
    start_lateral_offset: float = 10.0,
    start_altitude_bias: float = 12.0,
    dt: float = 0.02,
    timeout: float = 130.0,
    controller: object | None = None,
) -> tuple[SixDOFResult, CarrierTouchdown]:
    """Intercept and land on a moving carrier with sinusoidal deck motion.

    ``controller`` optionally replaces the FCS: a callable
    ``(state, ship, t, approach_speed) -> SurfaceCommand`` used to fly the same
    approach with a different command source (e.g. the raw pilot model in the
    assistance comparison). The actuator fault applies after it either way.
    """

    p = params or Aircraft6DOFParams()
    ship = carrier or MovingCarrier()
    wake = airwake or CarrierAirwake()
    gear = arresting_gear or ArrestingGear()
    gusts = turbulence or StochasticTurbulence(enabled=False)
    fault = actuator_fault or ActuatorFault(enabled=False)
    p.validate()
    ship.validate()
    wake.validate()
    gear.validate()
    gusts.validate()
    fault.validate()
    fcs = FlightControlSystem6DOF(p)
    approach_speed = ship.speed + 58.0
    start_x = -450.0
    start_altitude = ship.desired_altitude(start_x, 0.0) + start_altitude_bias
    state, trim = trim_level_flight(p, speed=approach_speed + 4.0, altitude=start_altitude, heading=0.0)
    # Start slightly high and right of the centerline so the aid visibly corrects.
    state = RigidBodyState(
        np.array([start_x, start_lateral_offset, -start_altitude]),
        state.velocity_body,
        state.quaternion,
        state.rates_body,
    )
    actuators = ActuatorState(trim.throttle, trim.elevator, trim.aileron, trim.rudder)
    if controller is not None and hasattr(controller, "warm_start"):
        controller.warm_start(trim)
    times = [0.0]
    states = [state]
    surfaces = [trim]
    time = 0.0
    touchdown: CarrierTouchdown | None = None

    while time < timeout:
        if controller is None:
            desired_surfaces = fcs.carrier_command(state, ship, time, approach_speed).surfaces
        else:
            desired_surfaces = controller(state, ship, time, approach_speed)
        desired_surfaces = fault.apply(desired_surfaces, time)
        actuators = step_actuators(actuators, desired_surfaces, p, dt)
        wind_body = wake.wind_body(state, ship, time) + gusts.wind_body(state, time)
        next_state = rk4_step_6dof(state, actuators.surfaces(), p, dt, wind_body=wind_body)
        time += dt
        times.append(time)
        states.append(next_state)
        surfaces.append(actuators.surfaces())
        deck_z = ship.deck_plane_z_down(next_state.position[0], next_state.position[1], time)
        if next_state.position[2] >= deck_z:
            touchdown = assess_moving_touchdown(next_state, ship, time, approach_speed)
            state = next_state
            break
        state = next_state

    if touchdown is None:
        deck = ship.deck_state(time)
        velocity_nav = state.position  # placeholder values make timeout explicit
        touchdown = CarrierTouchdown(
            False,
            False,
            "timeout before moving-deck intercept",
            float(state.position[0]),
            float(state.position[1]),
            float(deck["touchdown_x"]),
            0.0,
            0.0,
            float(state.position[1] - deck["y"]),
            math.degrees(deck["roll"]),
            math.degrees(deck["pitch"]),
        )

    glide_errors = np.asarray(
        [s.altitude - ship.desired_altitude(s.position[0], t) for s, t in zip(states, times)]
    )
    line_errors = np.asarray([s.position[1] - ship.deck_state(t)["y"] for s, t in zip(states, times)])
    arrestment = evaluate_arrestment(state, touchdown, ship, time, gear) if touchdown.touched else None
    metrics: dict[str, float | bool | str] = {
        "touchdown": touchdown.touched,
        "success": touchdown.success,
        "reason": touchdown.reason,
        "touchdown_x_m": touchdown.x,
        "touchdown_y_m": touchdown.y,
        "relative_speed_mps": touchdown.relative_speed,
        "relative_sink_rate_mps": touchdown.relative_sink_rate,
        "lineup_error_m": touchdown.lineup_error,
        "max_abs_glidepath_error_m": float(np.max(np.abs(glide_errors))),
        "final_lineup_error_m": float(line_errors[-1]),
        "deck_roll_deg": touchdown.deck_roll_deg,
        "deck_pitch_deg": touchdown.deck_pitch_deg,
        "duration_s": time,
        **wake.sample_metrics(np.asarray([s.position for s in states]), ship, np.asarray(times)),
        **(arrestment.metrics() if arrestment else {}),
        **gusts.metrics(),
        **fault.metrics(),
    }
    metrics.update(landing_quality_metrics(metrics))
    return SixDOFResult.from_states("moving_carrier_6dof", times, states, surfaces, metrics), touchdown


def simulate_carrier_challenge_6dof(
    challenge: str,
    dt: float = 0.02,
    timeout: float = 130.0,
) -> tuple[SixDOFResult, CarrierTouchdown]:
    """Run a harder carrier scenario used to stress the landing aids."""

    if challenge == "rough_seas":
        result, touchdown = simulate_moving_carrier_6dof(
            carrier=MovingCarrier(
                heave_amplitude=1.5,
                heave_period=8.0,
                pitch_amplitude_deg=1.0,
                pitch_period=9.0,
                roll_amplitude_deg=1.8,
                roll_period=6.5,
            ),
            airwake=CarrierAirwake(
                max_downwash_mps=2.8,
                max_headwind_mps=2.1,
                max_crosswind_mps=1.0,
            ),
            dt=dt,
            timeout=timeout,
        )
    elif challenge == "offset_recovery":
        result, touchdown = simulate_moving_carrier_6dof(
            start_lateral_offset=25.0,
            start_altitude_bias=18.0,
            dt=dt,
            timeout=timeout,
        )
    elif challenge == "strong_airwake":
        result, touchdown = simulate_moving_carrier_6dof(
            airwake=CarrierAirwake(
                max_downwash_mps=3.2,
                max_headwind_mps=2.4,
                max_crosswind_mps=1.2,
                unsteadiness=0.30,
            ),
            dt=dt,
            timeout=timeout,
        )
    elif challenge == "turbulent_recovery":
        result, touchdown = simulate_moving_carrier_6dof(
            turbulence=StochasticTurbulence(seed=23, intensity_mps=1.35),
            start_lateral_offset=14.0,
            dt=dt,
            timeout=timeout,
        )
    elif challenge == "aileron_fault":
        result, touchdown = simulate_moving_carrier_6dof(
            actuator_fault=ActuatorFault("aileron", start_time=38.0, effectiveness=0.58),
            dt=dt,
            timeout=timeout,
        )
    else:
        raise ValueError(f"unknown carrier challenge: {challenge}")
    result.metrics["challenge"] = challenge
    return result, touchdown
