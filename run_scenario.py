#!/usr/bin/env python3
"""Run one flight-control-lab scenario and print its metrics."""

from __future__ import annotations

import argparse
import json

from flight_control_lab.airwake import CarrierAirwake
from flight_control_lab.cobra import simulate_cobra_6dof
from flight_control_lab.scenarios import (
    simulate_carrier_approach,
    simulate_maneuver_assist,
    simulate_takeoff,
)
from flight_control_lab.scenarios6dof import (
    simulate_carrier_challenge_6dof,
    simulate_maneuver_6dof,
    simulate_moving_carrier_6dof,
    simulate_takeoff_6dof,
)


def metrics_for(name: str, calm: bool = False) -> dict:
    if name == "takeoff":
        return dict(simulate_takeoff_6dof().metrics)
    if name == "maneuver":
        return dict(simulate_maneuver_6dof().metrics)
    if name == "carrier":
        airwake = CarrierAirwake(enabled=not calm)
        result, touchdown = simulate_moving_carrier_6dof(airwake=airwake)
        metrics = dict(result.metrics)
        metrics["touchdown_success"] = touchdown.success
        return metrics
    if name in {
        "carrier-rough",
        "carrier-offset",
        "carrier-strong-airwake",
        "carrier-turbulent",
        "carrier-aileron-fault",
    }:
        challenge = {
            "carrier-rough": "rough_seas",
            "carrier-offset": "offset_recovery",
            "carrier-strong-airwake": "strong_airwake",
            "carrier-turbulent": "turbulent_recovery",
            "carrier-aileron-fault": "aileron_fault",
        }[name]
        result, touchdown = simulate_carrier_challenge_6dof(challenge)
        metrics = dict(result.metrics)
        metrics["touchdown_success"] = touchdown.success
        return metrics
    if name == "cobra":
        return dict(simulate_cobra_6dof().metrics)
    if name == "assist-compare-maneuver":
        from flight_control_lab.assistance import simulate_maneuver_assist_comparison

        comparison = simulate_maneuver_assist_comparison()
        metrics = {}
        for mode in ("assisted", "raw"):
            for key in (
                "max_alpha_deg",
                "max_bank_deg",
                "max_abs_altitude_error_m",
                "heading_overshoot_deg",
                "settle_time_s",
                "lost_control",
                "clean_capture",
            ):
                metrics[f"{mode}_{key}"] = comparison[mode][key]
        return metrics
    if name == "assist-compare":
        from flight_control_lab.assistance import simulate_carrier_assist_comparison

        comparison = simulate_carrier_assist_comparison()
        metrics = {}
        for mode in ("assisted", "raw"):
            for key in (
                "max_alpha_deg",
                "max_bank_deg",
                "max_abs_glidepath_error_m",
                "landing_quality_score",
                "lost_control",
                "touchdown",
                "reason",
            ):
                metrics[f"{mode}_{key}"] = comparison[mode][key]
        return metrics
    if name == "takeoff-pointmass":
        return dict(simulate_takeoff().metrics)
    if name == "maneuver-pointmass":
        return dict(simulate_maneuver_assist().metrics)
    if name == "carrier-pointmass":
        result, touchdown = simulate_carrier_approach()
        metrics = dict(result.metrics)
        metrics["touchdown_success"] = touchdown.success
        return metrics
    raise ValueError(f"unknown scenario: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        choices=[
            "takeoff",
            "maneuver",
            "carrier",
            "takeoff-pointmass",
            "maneuver-pointmass",
            "carrier-pointmass",
            "carrier-rough",
            "carrier-offset",
            "carrier-strong-airwake",
            "carrier-turbulent",
            "carrier-aileron-fault",
            "cobra",
            "assist-compare",
            "assist-compare-maneuver",
        ],
        help="scenario to run; plain names use the full 6-DOF model",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable metrics")
    parser.add_argument(
        "--calm",
        action="store_true",
        help="disable the deterministic carrier airwake for the carrier scenario",
    )
    args = parser.parse_args()
    metrics = metrics_for(args.scenario, calm=args.calm)

    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print(f"{args.scenario} metrics")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
