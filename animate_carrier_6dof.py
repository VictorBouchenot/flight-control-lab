#!/usr/bin/env python3
"""Render an animated top-down and profile view of the carrier approach."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from flight_control_lab.airwake import CarrierAirwake
from flight_control_lab.arresting import ArrestingGear
from flight_control_lab.carrier6dof import MovingCarrier
from flight_control_lab.scenarios6dof import simulate_moving_carrier_6dof


def render(output: Path, frames: int = 60, fps: int = 12) -> None:
    carrier = MovingCarrier()
    airwake = CarrierAirwake()
    gear = ArrestingGear()
    result, touchdown = simulate_moving_carrier_6dof(carrier=carrier, airwake=airwake)
    x, y, z_down = result.positions.T
    altitude = -z_down
    distance = touchdown.x - x
    tail = (distance >= 0.0) & (distance <= 1400.0)
    tail_indices = np.flatnonzero(tail)
    frame_indices = np.unique(np.linspace(tail_indices[0], len(result.times) - 1, frames).astype(int))
    desired = np.asarray([carrier.desired_altitude(px, t) for px, t in zip(x, result.times)])
    downwash = np.asarray(
        [airwake.wind_nav(position, carrier, t)[2] for position, t in zip(result.positions, result.times)]
    )

    fig, (ax_deck, ax_profile) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="#10141b")
    fig.suptitle("flight-control-lab — moving-carrier approach", color="white", fontsize=13, fontweight="bold")
    status = fig.text(0.01, 0.015, "", color="#d7dde8", fontsize=8, family="monospace")

    def draw_frame(step: int):
        idx = int(frame_indices[step])
        t = float(result.times[idx])
        deck = carrier.deck_state(t)
        ax_deck.clear()
        ax_profile.clear()
        ax_deck.set_facecolor("#151a23")
        ax_profile.set_facecolor("#151a23")

        rect = plt.Rectangle(
            (deck["x_start"], -carrier.width / 2),
            carrier.length,
            carrier.width,
            facecolor="#f5a742",
            edgecolor="#f5a742",
            alpha=0.30,
            lw=1.2,
        )
        ax_deck.add_patch(rect)
        for wire_index, fraction in enumerate(gear.wire_fractions, start=1):
            wire_x = deck["x_start"] + fraction * carrier.length
            ax_deck.plot([wire_x, wire_x], [-carrier.width / 2, carrier.width / 2], color="#d7dde8", lw=1.0)
            ax_deck.text(wire_x + 2, carrier.width / 2 + 2, str(wire_index), color="#d7dde8", fontsize=7)
        ax_deck.plot(x[tail], np.zeros_like(x[tail]), "--", color="#6aa9ff", lw=0.9)
        ax_deck.plot(x[tail_indices[0]:idx + 1], y[tail_indices[0]:idx + 1], color="#39d5c0", lw=2.0)
        ax_deck.scatter(x[idx], y[idx], color="#ff6b6b", s=38)
        if step == len(frame_indices) - 1:
            ax_deck.scatter(
                result.metrics["arrestment_stop_deck_x_m"],
                result.metrics["arrestment_stop_deck_y_m"],
                color="#ffd166",
                marker="X",
                s=58,
            )
        ax_deck.set_xlim(touchdown.x - 1350, touchdown.x + 260)
        ax_deck.set_ylim(-42, 42)
        ax_deck.set_xlabel("x (m)", color="white", fontsize=8)
        ax_deck.set_ylabel("lateral (m)", color="white", fontsize=8)
        ax_deck.set_title("Top-down deck capture", color="white", fontsize=10)
        ax_deck.grid(alpha=0.20)
        ax_deck.tick_params(colors="white", labelsize=7)

        ax_profile.plot(distance[tail], desired[tail], "--", color="#f5a742", lw=1.4, label="glideslope")
        ax_profile.plot(distance[tail], altitude[tail], color="#394254", lw=1.0, alpha=0.8)
        ax_profile.plot(distance[tail_indices[0]:idx + 1], altitude[tail_indices[0]:idx + 1], color="#6aa9ff", lw=2.0, label="aircraft")
        ax_profile.scatter(distance[idx], altitude[idx], color="#ff6b6b", s=34)
        ax_wake = ax_profile.twinx()
        ax_wake.plot(distance[tail], downwash[tail], ":", color="#ff78c8", lw=1.2, label="downwash")
        ax_wake.scatter(distance[idx], downwash[idx], color="#ff78c8", s=24)
        ax_wake.set_ylim(0, max(2.5, float(np.max(downwash[tail])) * 1.25))
        ax_wake.set_ylabel("downwash (m/s)", color="#ff78c8", fontsize=8)
        ax_wake.tick_params(colors="#ff78c8", labelsize=7)
        ax_profile.set_xlim(1400, 0)
        ax_profile.set_ylim(0, max(100.0, float(np.max(altitude[tail])) * 1.1))
        ax_profile.set_xlabel("distance to touchdown (m)", color="white", fontsize=8)
        ax_profile.set_ylabel("altitude (m)", color="white", fontsize=8)
        ax_profile.set_title("Profile + burble", color="white", fontsize=10)
        ax_profile.grid(alpha=0.22)
        ax_profile.tick_params(colors="white", labelsize=7)
        if step == 0:
            lines, labels = ax_profile.get_legend_handles_labels()
            wake_lines, wake_labels = ax_wake.get_legend_handles_labels()
            ax_profile.legend(lines + wake_lines, labels + wake_labels, facecolor="#1b2230", labelcolor="white", fontsize=7)

        status.set_text(
            f"t={t:5.1f}s  deck roll={deck['roll'] * 57.2958:+.2f} deg  "
            f"lineup={y[idx]:+.2f} m  downwash={downwash[idx]:.2f} m/s  "
            f"target={result.metrics['arrestment_reason']}"
        )
        return []

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=95)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/carrier-approach-6dof.gif"))
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    render(args.output, frames=args.frames, fps=args.fps)
    print(args.output)


if __name__ == "__main__":
    main()
