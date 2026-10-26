#!/usr/bin/env python3
"""Render 3D previews (still + animation) of the scripted cobra maneuver."""

from __future__ import annotations

import argparse
from pathlib import Path

from flight_control_lab.cobra import simulate_cobra_6dof
from flight_control_lab.viz3d_render import render_cobra_animation, render_cobra_still


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--still", type=Path, default=Path("artifacts/cobra-3d-preview.png"))
    parser.add_argument("--animation", type=Path, default=Path("artifacts/cobra-3d.gif"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    result = simulate_cobra_6dof()
    print(render_cobra_still(result, args.still))
    if not args.no_animation:
        print(render_cobra_animation(result, args.animation, frames=args.frames, fps=args.fps))


if __name__ == "__main__":
    main()
