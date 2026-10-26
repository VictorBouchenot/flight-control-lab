"""Controller comparisons: automatic FCS guidance versus an imperfect pilot model.

Both pursue the same scenario targets. The FCS does not consume the pilot
model's commands. Shadow commands are diagnostics only, not control inputs.
"""

from __future__ import annotations

import math

import numpy as np

from .aircraft6dof import Aircraft6DOFParams, SurfaceCommand
from .airwake import CarrierAirwake
from .arresting import ArrestingGear
from .carrier6dof import MovingCarrier
from .dynamics import clamp
from .fcs6dof import FlightControlSystem6DOF, GuidanceTargets
from .pilot import PilotModel
from .scenarios6dof import SixDOFResult, simulate_maneuver_6dof, simulate_moving_carrier_6dof


def envelope_metrics(result: SixDOFResult, params: Aircraft6DOFParams | None = None) -> dict[str, float]:
    """Peak beyond each protected envelope limit, plus a control-loss flag."""

    p = params or Aircraft6DOFParams()
    alphas = np.asarray([s.alpha for s in _states(result)])
    betas = np.asarray([s.beta for s in _states(result)])
    rolls = np.asarray([abs(s.euler[0]) for s in _states(result)])
    duration = float(result.times[-1]) if len(result.times) else 0.0
    return {
        "max_alpha_deg": math.degrees(float(np.max(alphas))),
        "max_abs_beta_deg": math.degrees(float(np.max(np.abs(betas)))),
        "max_bank_deg": math.degrees(float(np.max(rolls))),
        "flight_time_s": duration,
        # departure from controlled flight: past any teaching limit by a wide margin
        "lost_control": bool(np.max(alphas) > math.radians(25.0) or np.max(rolls) > math.radians(60.0)),
    }


def _states(result: SixDOFResult):
    from .aircraft6dof import RigidBodyState

    return [
        RigidBodyState(result.positions[i], result.velocities_body[i], result.quaternions[i], result.rates_body[i])
        for i in range(len(result.times))
    ]


class _ShadowPilotManeuver:
    """Same idea for the heading capture: FCS flies, shadow pilot records."""

    def __init__(self, fcs: FlightControlSystem6DOF, pilot: PilotModel, dt: float) -> None:
        self._fcs = fcs
        self._pilot = pilot
        self._dt = dt

    def warm_start(self, trim: SurfaceCommand) -> None:
        self._pilot.warm_start(trim)

    def __call__(self, state, target_heading, t) -> SurfaceCommand:
        self._pilot.maneuver_command(state, target_heading, t)  # recorded, not applied
        targets = GuidanceTargets(138.0, 2_000.0, target_heading, math.radians(45.0))
        return self._fcs.command(state, targets, self._dt).surfaces


def simulate_carrier_assist_comparison(
    dt: float = 0.02,
    timeout: float = 130.0,
) -> dict[str, object]:
    """Same carrier approach twice: FCS-assisted vs raw pilot stick.

    Both runs share the pilot's intent (glideslope, centerline, approach
    speed), the moving deck and the deterministic airwake. The guidance and feedback laws differ; this is not an isolated
    assistance-layer ablation.
    """

    carrier = MovingCarrier()
    airwake = CarrierAirwake()
    gear = ArrestingGear()

    assisted, assisted_td = simulate_moving_carrier_6dof(
        carrier=carrier, airwake=airwake, arresting_gear=gear, dt=dt, timeout=timeout
    )
    pilot = PilotModel()
    raw, raw_td = simulate_moving_carrier_6dof(
        carrier=carrier,
        airwake=airwake,
        arresting_gear=gear,
        dt=dt,
        timeout=timeout,
        controller=pilot.carrier_command,
    )
    assisted_env = envelope_metrics(assisted)
    raw_env = envelope_metrics(raw)
    comparison = {
        # Stick history from the raw run: it shows the pilot fighting the
        # burble unaided. (On the assisted trajectory the FCS holds a small
        # deliberate path offset, so a shadow stick there only re-trims - it
        # carries no intent and is not shown.)
        "pilot_stick": pilot.stick_history(),
        "assisted": {**assisted_env, **assisted.metrics},
        "raw": {**raw_env, **raw.metrics},
        "delta": {
            "max_alpha_deg": assisted_env["max_alpha_deg"] - raw_env["max_alpha_deg"],
            "max_bank_deg": assisted_env["max_bank_deg"] - raw_env["max_bank_deg"],
            "max_abs_glidepath_error_m": assisted.metrics["max_abs_glidepath_error_m"] - raw.metrics["max_abs_glidepath_error_m"],
            "landing_quality_score": assisted.metrics["landing_quality_score"] - raw.metrics["landing_quality_score"],
        },
        "assisted_touchdown": assisted_td,
        "raw_touchdown": raw_td,
        "assisted_result": assisted,
        "raw_result": raw,
    }
    return comparison


def _maneuver_metrics(result: SixDOFResult, target_heading: float, dt: float) -> dict[str, float]:
    """Capture quality: overshoot, settle time, altitude deviation."""

    states = _states(result)
    headings = np.asarray([s.euler[2] for s in states])
    altitudes = np.asarray([s.altitude for s in states])
    # signed error along the turn direction (the capture always turns right here)
    errors = np.asarray([(target_heading - h + math.pi) % (2.0 * math.pi) - math.pi for h in headings])
    overshoot = max(0.0, -float(np.min(errors)))
    beyond = np.abs(errors) > math.radians(3.0)
    if np.any(beyond):
        # settle = first time after the last excursion (full duration if never)
        last_outside = int(np.max(np.nonzero(beyond)))
        settle = float(result.times[min(last_outside + 1, len(result.times) - 1)])
    else:
        settle = 0.0
    return {
        "heading_overshoot_deg": math.degrees(overshoot),
        "settle_time_s": settle,
        "max_abs_altitude_error_m": float(np.max(np.abs(altitudes - 2_000.0))),
    }


def simulate_maneuver_assist_comparison(
    dt: float = 0.02,
    timeout: float = 65.0,
) -> dict[str, object]:
    """Same 120 deg heading capture twice: FCS-assisted vs raw pilot stick.

    Same aircraft, same trim, same target. The assisted run flies through the
    FCS (a shadow pilot records its stick on that same trajectory); the raw
    run applies the pilot stick directly to the surfaces.
    """

    target_heading = math.radians(120.0)
    shadow_pilot = PilotModel()
    shadow = _ShadowPilotManeuver(FlightControlSystem6DOF(), shadow_pilot, dt)
    assisted = simulate_maneuver_6dof(dt=dt, timeout=timeout, controller=shadow)
    pilot = PilotModel()
    raw = simulate_maneuver_6dof(dt=dt, timeout=timeout, controller=pilot.maneuver_command)

    assisted_env = envelope_metrics(assisted)
    raw_env = envelope_metrics(raw)
    assisted_cap = _maneuver_metrics(assisted, target_heading, dt)
    raw_cap = _maneuver_metrics(raw, target_heading, dt)
    # A capture only counts if the aircraft stayed under control: ending near
    # the target heading after rolling inverted is luck, not a capture.
    assisted_clean = bool(assisted.metrics["heading_captured"]) and not assisted_env["lost_control"]
    raw_clean = bool(raw.metrics["heading_captured"]) and not raw_env["lost_control"]
    return {
        "pilot_stick": shadow_pilot.stick_history(),
        "assisted": {**assisted_env, **assisted_cap, **assisted.metrics, "clean_capture": assisted_clean},
        "raw": {**raw_env, **raw_cap, **raw.metrics, "clean_capture": raw_clean},
        "delta": {
            "heading_overshoot_deg": assisted_cap["heading_overshoot_deg"] - raw_cap["heading_overshoot_deg"],
            "settle_time_s": assisted_cap["settle_time_s"] - raw_cap["settle_time_s"],
            "max_abs_altitude_error_m": assisted_cap["max_abs_altitude_error_m"] - raw_cap["max_abs_altitude_error_m"],
            "max_bank_deg": assisted_env["max_bank_deg"] - raw_env["max_bank_deg"],
            "max_alpha_deg": assisted_env["max_alpha_deg"] - raw_env["max_alpha_deg"],
        },
        "assisted_result": assisted,
        "raw_result": raw,
    }
