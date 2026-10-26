#!/usr/bin/env python3
"""Render 3D previews (still + animation) of a carrier approach scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from flight_control_lab.airwake import CarrierAirwake
from flight_control_lab.arresting import ArrestingGear
from flight_control_lab.carrier6dof import MovingCarrier
from flight_control_lab.scenarios6dof import simulate_carrier_challenge_6dof, simulate_moving_carrier_6dof
from flight_control_lab.viz3d_render import render_animation, render_still

CHALLENGES = ("baseline", "rough_seas", "offset_recovery", "strong_airwake", "turbulent_recovery", "aileron_fault")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", choices=CHALLENGES, default="baseline")
    parser.add_argument("--still", type=Path, default=Path("artifacts/carrier-3d-preview.png"))
    parser.add_argument("--animation", type=Path, default=Path("artifacts/carrier-approach-3d.gif"))
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--no-animation", action="store_true", help="only render the still image")
    args = parser.parse_args()

    gear = ArrestingGear()
    if args.challenge == "baseline":
        carrier = MovingCarrier()
        result, touchdown = simulate_moving_carrier_6dof(carrier=carrier, airwake=CarrierAirwake(), arresting_gear=gear)
    else:
        result, touchdown = simulate_carrier_challenge_6dof(args.challenge)
        carrier = MovingCarrier()
        if args.challenge == "rough_seas":
            carrier = MovingCarrier(
                heave_amplitude=1.5, heave_period=8.0,
                pitch_amplitude_deg=1.0, pitch_period=9.0,
                roll_amplitude_deg=1.8, roll_period=6.5,
            )

    still = render_still(result, carrier, gear, touchdown, args.still)
    print(still)
    if not args.no_animation:
        animation = render_animation(result, carrier, gear, touchdown, args.animation, frames=args.frames, fps=args.fps)
        print(animation)


if __name__ == "__main__":
    main()
