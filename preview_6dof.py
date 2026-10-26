#!/usr/bin/env python3
"""Render static diagnostic previews for the full 6-DOF scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flight_control_lab.airwake import CarrierAirwake
from flight_control_lab.arresting import ArrestingGear
from flight_control_lab.carrier6dof import MovingCarrier
from flight_control_lab.scenarios6dof import (
    simulate_maneuver_6dof,
    simulate_moving_carrier_6dof,
    simulate_takeoff_6dof,
)


def scenario_result(name: str):
    if name == "takeoff":
        return simulate_takeoff_6dof(), None
    if name == "maneuver":
        return simulate_maneuver_6dof(), None
    if name == "carrier":
        carrier = MovingCarrier()
        airwake = CarrierAirwake()
        result, touchdown = simulate_moving_carrier_6dof(carrier=carrier, airwake=airwake)
        return result, (carrier, touchdown, airwake)
    raise ValueError(f"unknown scenario: {name}")


def render_carrier_deck(result, carrier: MovingCarrier, touchdown, airwake: CarrierAirwake, output: Path) -> None:
    """Render a focused top-down deck, wire and rollout diagnostic."""

    gear = ArrestingGear()
    deck = carrier.deck_state(result.times[-1])
    x, y, z_down = result.positions.T
    altitude = -z_down
    distance_to_touchdown = touchdown.x - x
    tail = (distance_to_touchdown >= 0.0) & (distance_to_touchdown <= 1200.0)
    desired = np.asarray([carrier.desired_altitude(px, t) for px, t in zip(x, result.times)])
    wake_samples = np.asarray(
        [airwake.wind_nav(position, carrier, t) for position, t in zip(result.positions, result.times)]
    )

    fig = plt.figure(figsize=(13, 10), facecolor="#10141b")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.30, wspace=0.24)
    fig.suptitle(
        "flight-control-lab — carrier deck, wires and arrested rollout",
        color="white",
        fontsize=15,
        fontweight="bold",
    )

    ax_deck = fig.add_subplot(grid[0, :])
    ax_deck.set_facecolor("#151a23")
    deck_rect = plt.Rectangle(
        (deck["x_start"], -carrier.width / 2),
        carrier.length,
        carrier.width,
        facecolor="#f5a742",
        alpha=0.28,
        edgecolor="#f5a742",
        linewidth=1.5,
        label="moving deck",
    )
    ax_deck.add_patch(deck_rect)
    for index, fraction in enumerate(gear.wire_fractions, start=1):
        wire_x = deck["x_start"] + fraction * carrier.length
        color = "#8be05c" if index == result.metrics.get("arrestment_wire_number") else "#d7dde8"
        width = 2.8 if index == result.metrics.get("arrestment_wire_number") else 1.4
        ax_deck.plot([wire_x, wire_x], [-carrier.width / 2, carrier.width / 2], color=color, lw=width)
        ax_deck.text(
            wire_x + 2,
            carrier.width / 2 + (2.0 if index % 2 else 6.0),
            f"{index}-wire",
            color=color,
            fontsize=8,
        )
    ax_deck.plot(x[tail], y[tail], color="#39d5c0", lw=2.0, label="final approach")
    ax_deck.axhline(0.0, color="#6aa9ff", ls="--", lw=1.0, label="centerline")
    ax_deck.scatter(touchdown.x, touchdown.y, color="#ff6b6b", s=55, label="touchdown")
    stop_x = result.metrics.get("arrestment_stop_deck_x_m")
    stop_y = result.metrics.get("arrestment_stop_deck_y_m")
    if stop_x is not None:
        ax_deck.scatter(stop_x, stop_y, color="#ffd166", s=70, marker="X", label="rollout stop")
    ax_deck.set_xlim(deck["x_start"] - 360, deck["x_start"] + carrier.length + 90)
    ax_deck.set_ylim(-48, 48)
    ax_deck.set_xlabel("x (m)", color="white")
    ax_deck.set_ylabel("lateral offset (m)", color="white")
    ax_deck.set_title("Top-down wire capture and rollout", color="white")
    ax_deck.grid(alpha=0.22)
    ax_deck.tick_params(colors="white")
    ax_deck.legend(facecolor="#1b2230", labelcolor="white", loc="upper left")

    ax_profile = fig.add_subplot(grid[1, 0])
    ax_profile.set_facecolor("#151a23")
    ax_profile.plot(distance_to_touchdown[tail], altitude[tail], color="#6aa9ff", lw=2, label="aircraft")
    ax_profile.plot(distance_to_touchdown[tail], desired[tail], "--", color="#f5a742", lw=1.7, label="moving glideslope")
    ax_profile.scatter(0.0, altitude[np.argmin(np.abs(distance_to_touchdown))], color="#ff6b6b", s=40)
    ax_wake = ax_profile.twinx()
    ax_wake.plot(distance_to_touchdown[tail], wake_samples[tail, 2], ":", color="#ff78c8", lw=1.8, label="downwash")
    ax_wake.set_ylabel("downwash (m/s)", color="#ff78c8")
    ax_wake.tick_params(axis="y", colors="#ff78c8")
    lines, labels = ax_profile.get_legend_handles_labels()
    wake_lines, wake_labels = ax_wake.get_legend_handles_labels()
    ax_profile.legend(lines + wake_lines, labels + wake_labels, facecolor="#1b2230", labelcolor="white", loc="upper right")
    ax_profile.set_xlim(1200, 0)
    ax_profile.set_xlabel("distance to touchdown (m)", color="white")
    ax_profile.set_ylabel("altitude (m)", color="white")
    ax_profile.set_title("Final approach profile", color="white")
    ax_profile.grid(alpha=0.25)
    ax_profile.tick_params(colors="white")

    ax_errors = fig.add_subplot(grid[1, 1])
    ax_errors.set_facecolor("#151a23")
    glide_error = altitude - desired
    ax_errors.plot(distance_to_touchdown[tail], glide_error[tail], color="#8be05c", lw=1.8, label="glidepath error")
    ax_errors.plot(distance_to_touchdown[tail], y[tail], color="#ff9f6e", lw=1.8, label="lineup error")
    ax_errors.axhline(0.0, color="#d7dde8", lw=0.8, alpha=0.7)
    ax_errors.set_xlim(1200, 0)
    ax_errors.set_xlabel("distance to touchdown (m)", color="white")
    ax_errors.set_ylabel("error (m)", color="white")
    ax_errors.set_title("Approach corrections", color="white")
    ax_errors.grid(alpha=0.25)
    ax_errors.tick_params(colors="white")
    ax_errors.legend(facecolor="#1b2230", labelcolor="white")

    summary = (
        f"{result.metrics['arrestment_reason']}\n"
        f"wire: {result.metrics['arrestment_wire_number']}  "
        f"offset: {result.metrics['arrestment_wire_offset_m']:.1f} m\n"
        f"stop: {result.metrics['arrestment_stop_distance_m']:.1f} m  "
        f"margin: {result.metrics['arrestment_stop_margin_m']:.1f} m\n"
        f"max deck roll during rollout: {result.metrics['arrestment_max_deck_roll_deg']:.2f} deg\n"
        f"arrestment score: {result.metrics['arrestment_score']:.1f}/100"
    )
    ax_deck.text(
        0.012,
        0.07,
        summary,
        transform=ax_deck.transAxes,
        color="#d7dde8",
        fontsize=9,
        family="monospace",
        bbox={"facecolor": "#10141b", "edgecolor": "#394254", "alpha": 0.88},
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def render(name: str, output: Path) -> list[Path]:
    result, extra = scenario_result(name)
    x, y, z_down = result.positions.T
    altitude = -z_down
    speed = np.linalg.norm(result.velocities_body, axis=1)
    surfaces = np.degrees(result.surface_history[:, 1:])

    fig = plt.figure(figsize=(13, 9), facecolor="#10141b")
    fig.suptitle(
        f"flight-control-lab — {name} (full 6-DOF educational model)",
        color="white",
        fontsize=15,
        fontweight="bold",
    )

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.set_facecolor("#10141b")
    ax3d.plot(x, y, altitude, color="#39d5c0", lw=2.2, label="6-DOF trajectory")
    ax3d.scatter(x[0], y[0], altitude[0], color="#8be05c", s=35, label="start")
    ax3d.scatter(x[-1], y[-1], altitude[-1], color="#ff6b6b", s=35, label="end")
    if extra:
        carrier, touchdown, airwake = extra
        deck = carrier.deck_state(result.times[-1])
        deck_x = np.array([deck["x_start"], deck["x_start"] + carrier.length])
        deck_y = np.array([-carrier.width / 2, carrier.width / 2])
        xx, yy = np.meshgrid(deck_x, deck_y)
        zz = np.full_like(xx, -deck["z_down"])
        ax3d.plot_surface(xx, yy, zz, color="#f5a742", alpha=0.38)
        gear = ArrestingGear()
        for fraction in gear.wire_fractions:
            wire_x = deck["x_start"] + fraction * carrier.length
            ax3d.plot(
                [wire_x, wire_x],
                [-carrier.width / 2, carrier.width / 2],
                [-deck["z_down"], -deck["z_down"]],
                color="white",
                lw=1.0,
            )
        ax3d.scatter(touchdown.x, touchdown.y, -deck["z_down"], color="#ff6b6b", s=45)
        if "arrestment_stop_deck_x_m" in result.metrics:
            ax3d.scatter(
                result.metrics["arrestment_stop_deck_x_m"],
                result.metrics["arrestment_stop_deck_y_m"],
                -deck["z_down"],
                color="#ffd166",
                marker="X",
                s=55,
            )
    ax3d.set_xlabel("x (m)", color="white")
    ax3d.set_ylabel("y (m)", color="white")
    ax3d.set_zlabel("altitude (m)", color="white")
    ax3d.tick_params(colors="white")
    ax3d.legend(facecolor="#1b2230", labelcolor="white")

    ax_alt = fig.add_subplot(2, 2, 2)
    ax_alt.set_facecolor("#151a23")
    ax_alt.plot(result.times, altitude, color="#6aa9ff", lw=2, label="aircraft")
    if extra:
        carrier, _, airwake = extra
        desired = [carrier.desired_altitude(px, t) for px, t in zip(x, result.times)]
        ax_alt.plot(result.times, desired, "--", color="#f5a742", lw=1.7, label="moving glideslope")
        wake_samples = np.asarray(
            [airwake.wind_nav(position, carrier, t) for position, t in zip(result.positions, result.times)]
        )
        ax_wake = ax_alt.twinx()
        ax_wake.plot(result.times, wake_samples[:, 2], ":", color="#ff78c8", lw=1.8, label="airwake downwash")
        ax_wake.set_ylabel("downwash (m/s)", color="#ff78c8")
        ax_wake.tick_params(axis="y", colors="#ff78c8")
        lines, labels = ax_alt.get_legend_handles_labels()
        wake_lines, wake_labels = ax_wake.get_legend_handles_labels()
        ax_alt.legend(lines + wake_lines, labels + wake_labels, facecolor="#1b2230", labelcolor="white", loc="best")
    ax_alt.set_title("Altitude / glidepath", color="white")
    ax_alt.set_xlabel("time (s)", color="white")
    ax_alt.set_ylabel("altitude (m)", color="white")
    ax_alt.grid(alpha=0.25)
    ax_alt.tick_params(colors="white")
    if not extra:
        ax_alt.legend(facecolor="#1b2230", labelcolor="white")

    ax_speed = fig.add_subplot(2, 2, 3)
    ax_speed.set_facecolor("#151a23")
    ax_speed.plot(result.times, speed, color="#8be05c", lw=2)
    ax_speed.set_title("Airspeed", color="white")
    ax_speed.set_xlabel("time (s)", color="white")
    ax_speed.set_ylabel("m/s", color="white")
    ax_speed.grid(alpha=0.25)
    ax_speed.tick_params(colors="white")

    ax_surface = fig.add_subplot(2, 2, 4)
    ax_surface.set_facecolor("#151a23")
    ax_surface.plot(result.times, surfaces[:, 0], label="elevator", lw=1.5)
    ax_surface.plot(result.times, surfaces[:, 1], label="aileron", lw=1.5)
    ax_surface.plot(result.times, surfaces[:, 2], label="rudder", lw=1.5)
    ax_surface.set_title("Control-surface commands", color="white")
    ax_surface.set_xlabel("time (s)", color="white")
    ax_surface.set_ylabel("degrees", color="white")
    ax_surface.grid(alpha=0.25)
    ax_surface.tick_params(colors="white")
    ax_surface.legend(facecolor="#1b2230", labelcolor="white")

    preferred_metrics = [
        "success",
        "reason",
        "arrestment_reason",
        "arrestment_wire_number",
        "arrestment_score",
        "relative_sink_rate_mps",
        "lineup_error_m",
        "max_airwake_downwash_mps",
        "rotation_time_s",
        "heading_error_deg",
        "max_alpha_deg",
        "duration_s",
    ]
    shown_metrics = [(key, result.metrics[key]) for key in preferred_metrics if key in result.metrics][:8]
    metrics_text = "\n".join(
        f"{key}: {value:.3g}" if isinstance(value, float) else f"{key}: {value}"
        for key, value in shown_metrics
    )
    fig.text(0.01, 0.012, metrics_text, color="#d7dde8", fontsize=8, family="monospace")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    created = [output]
    if extra:
        carrier, touchdown, airwake = extra
        deck_output = output.with_name(output.stem.replace("carrier", "carrier-deck") + output.suffix)
        render_carrier_deck(result, carrier, touchdown, airwake, deck_output)
        created.append(deck_output)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=["takeoff", "maneuver", "carrier"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("artifacts") / f"{args.scenario}-6dof-preview.png"
    for created in render(args.scenario, output):
        print(created)


if __name__ == "__main__":
    main()
