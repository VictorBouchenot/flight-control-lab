"""Matplotlib renderers for the 3D carrier-approach views.

Split from :mod:`flight_control_lab.viz3d` so the geometry helpers stay
dependency-light. Importing this module pulls in matplotlib with the Agg
backend, matching the project's other preview scripts.
"""

from __future__ import annotations

from pathlib import Path

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .arresting import ArrestingGear
from .carrier6dof import MovingCarrier
from .scenarios6dof import SixDOFResult
from .viz3d import (
    aircraft_glyph,
    approach_window,
    deck_markings,
    deck_mesh,
    ned_to_display,
    rafale_mesh,
    ship_mesh,
    wire_lines,
)

SHIP_COLORS = {"hull": "#4d5665", "island": "#6a7382"}
AIRCRAFT_COLORS = {"wing": "#d3dae6", "canard": "#c2cad9", "fin": "#b7c0d0", "fuselage_side": "#aab4c6"}

DECK_COLOR = "#f5a742"
TRAIL_COLOR = "#39d5c0"
GLIDESLOPE_COLOR = "#6aa9ff"
GLYPH_COLOR = "#ff6b6b"
WIRE_COLOR = "#d7dde8"
BG = "#10141b"
PANEL = "#151a23"


def _style_3d(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=7)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.15)
        pane.label.set_color("white")
    ax.grid(alpha=0.20)


def _draw_deck(ax, carrier: MovingCarrier, gear: ArrestingGear, t: float, sea: bool = True) -> None:
    for polygon, part in ship_mesh(carrier, t):
        ax.add_collection3d(Poly3DCollection([polygon], facecolor=SHIP_COLORS[part], edgecolor="#39404d", lw=0.4, alpha=0.92))
    grid_x, grid_y, grid_z = deck_mesh(carrier, t)
    ax.plot_surface(grid_x, grid_y, grid_z, color=DECK_COLOR, alpha=0.55, edgecolor=DECK_COLOR, lw=0.5)
    for line in deck_markings(carrier, t):
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#f7f3e8", lw=0.8, alpha=0.85)
    for line in wire_lines(carrier, gear, t):
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#ffd166", lw=1.8)
    if sea:
        x0, x1 = grid_x.min() - 500.0, grid_x.max() + 120.0
        y0, y1 = grid_y.min() - 45.0, grid_y.max() + 45.0
        sea_x, sea_y = np.meshgrid([x0, x1], [y0, y1])
        ax.plot_surface(sea_x, sea_y, np.zeros_like(sea_x), color="#2f6f9f", alpha=0.10)


def _draw_glyph(ax, segments, color: str = GLYPH_COLOR, lw: float = 1.8) -> None:
    for segment in segments:
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=lw)


def _draw_rafale(ax, state, scale: float = 1.0) -> None:
    for polygon, part in rafale_mesh(state, scale=scale):
        ax.add_collection3d(
            Poly3DCollection(
                [polygon],
                facecolor=AIRCRAFT_COLORS.get(part, "#9aa4b5"),
                edgecolor="#ff6b6b",
                lw=0.5,
                alpha=0.97,
            )
        )


def _set_box(ax, x_center: float, x_span: float, y_span: float, z_span: float) -> None:
    ax.set_xlim(x_center - x_span / 2, x_center + x_span / 2)
    ax.set_ylim(-y_span / 2, y_span / 2)
    ax.set_zlim(0.0, z_span)
    ax.set_box_aspect((x_span, y_span, z_span * 2.2))
    # few round ticks so the labels never pile up at the pane edges
    x0, x1 = ax.get_xlim()
    ax.set_xticks(np.linspace(x0, x1, 5).round(-1))
    ax.set_yticks(np.linspace(-y_span / 2, y_span / 2, 3).round(-1))
    ax.set_zticks(np.linspace(0.0, z_span, 4).round(-1))


def render_still(
    result: SixDOFResult,
    carrier: MovingCarrier,
    gear: ArrestingGear,
    touchdown,
    output: Path,
    trail_distance: float = 1_400.0,
) -> Path:
    """Two-panel 3D still: final-700 m perspective and deck close-up."""

    from .aircraft6dof import RigidBodyState

    display = ned_to_display(result.positions)
    tail = approach_window(result.times, result.positions, touchdown.x, min(trail_distance, 700.0))
    t_end = float(result.times[-1])
    views = [
        # (x_center offset, x_span, y_span, z_span, elev, azim, title, glyph_count)
        (touchdown.x - 260.0, 820.0, 110.0, 95.0, 20.0, -122.0, "Final 700 m", 5),
        (touchdown.x + 10.0, 270.0, 58.0, 46.0, 16.0, -108.0, "Deck close-up", 3),
    ]

    fig = plt.figure(figsize=(12.5, 5.6), facecolor=BG)
    fig.suptitle(
        f"flight-control-lab - 3D carrier approach ({result.metrics.get('challenge', 'baseline')})",
        color="white",
        fontsize=13,
        fontweight="bold",
    )

    for slot, (x_center, x_span, y_span, z_span, elev, azim, title, glyphs) in enumerate(views):
        ax = fig.add_subplot(1, 2, slot + 1, projection="3d")
        _draw_deck(ax, carrier, gear, t_end)
        desired = np.asarray(
            [carrier.desired_altitude(px, t) for px, t in zip(result.positions[tail, 0], result.times[tail])]
        )
        ax.plot(
            result.positions[tail, 0], np.zeros(len(tail)), desired,
            "--", color=GLIDESLOPE_COLOR, lw=0.9, label="glideslope",
        )
        ax.plot(display[tail, 0], display[tail, 1], display[tail, 2], color=TRAIL_COLOR, lw=2.0, label="aircraft")
        glyph_idx = np.unique(np.linspace(tail[0], tail[-1], glyphs).astype(int))
        for k in glyph_idx:
            state = RigidBodyState(result.positions[k], result.velocities_body[k], result.quaternions[k], result.rates_body[k])
            _draw_rafale(ax, state, scale=2.6)
        ax.scatter(
            [touchdown.x], [touchdown.y], [-carrier.deck_plane_z_down(touchdown.x, touchdown.y, t_end)],
            color="#ffd166", marker="X", s=55, label="touchdown",
        )
        _set_box(ax, x_center, x_span, y_span, z_span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x (m)", fontsize=7, labelpad=6)
        # mpl 3D clips pane labels outside the axes bbox; keep the y label on
        # the right-hand panel only so nothing is cut at the figure edge
        if slot == 1:
            ax.set_ylabel("lateral (m)", fontsize=7, labelpad=8)
        ax.set_zlabel("altitude (m)", fontsize=7, labelpad=6)
        _style_3d(ax, title)
        if slot == 0:
            ax.legend(facecolor="#1b2230", labelcolor="white", fontsize=7, loc="upper left")

    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return output


def render_animation(
    result: SixDOFResult,
    carrier: MovingCarrier,
    gear: ArrestingGear,
    touchdown,
    output: Path,
    frames: int = 48,
    fps: int = 12,
    trail_distance: float = 1_200.0,
) -> Path:
    """Animated 3D view: deck moves, wires sweep, the aircraft glyph flies in."""

    display = ned_to_display(result.positions)
    tail = approach_window(result.times, result.positions, touchdown.x, trail_distance)
    frame_indices = np.unique(np.linspace(tail[0], len(result.times) - 1, frames).astype(int))

    fig = plt.figure(figsize=(8.6, 6.2), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("flight-control-lab - 3D moving-deck approach", color="white", fontsize=12, fontweight="bold")
    status = fig.text(0.02, 0.03, "", color="#d7dde8", fontsize=8, family="monospace")

    def draw_frame(step: int):
        idx = int(frame_indices[step])
        t = float(result.times[idx])
        deck = carrier.deck_state(t)
        ax.clear()
        _draw_deck(ax, carrier, gear, t)
        ax.plot(display[tail[0]:idx + 1, 0], display[tail[0]:idx + 1, 1], display[tail[0]:idx + 1, 2], color=TRAIL_COLOR, lw=1.8)
        from .aircraft6dof import RigidBodyState

        state = RigidBodyState(result.positions[idx], result.velocities_body[idx], result.quaternions[idx], result.rates_body[idx])
        _draw_rafale(ax, state, scale=3.0)
        deck_center = deck["x_start"] + carrier.length * 0.5
        gap = abs(deck_center - display[idx, 0])
        x_span = float(np.clip(gap + 320.0, 480.0, 1_450.0))
        x_center = (deck_center + display[idx, 0]) / 2.0
        _set_box(ax, x_center, x_span, 130.0, 110.0)
        ax.view_init(elev=18.0, azim=-128.0)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("lateral (m)", fontsize=7)
        ax.set_zlabel("altitude (m)", fontsize=7)
        _style_3d(ax, "Moving deck, wires and burble-correcting approach")
        status.set_text(
            f"t={t:5.1f}s  deck roll={np.degrees(deck['roll']):+.2f} deg  deck pitch={np.degrees(deck['pitch']):+.2f} deg  "
            f"heave={deck['heave']:+.2f} m  lineup={result.positions[idx, 1] - deck['y']:+.2f} m  "
            f"alt={display[idx, 2]:.1f} m"
        )
        return []

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return output


def render_cobra_still(result: SixDOFResult, output: Path) -> Path:
    """3D still of the cobra: trajectory, attitude sequence and alpha/speed trace."""

    from .aircraft6dof import RigidBodyState

    display = ned_to_display(result.positions)
    alphas = np.degrees([math.atan2(v[2], v[0]) for v in result.velocities_body])
    speeds = np.linalg.norm(result.velocities_body, axis=1)
    peak_idx = int(np.argmax(alphas))

    fig = plt.figure(figsize=(12.5, 5.6), facecolor=BG)
    fig.suptitle("flight-control-lab - scripted cobra (post-stall teaching model)", color="white", fontsize=13, fontweight="bold")

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    n = len(result.times)
    hot = np.flatnonzero(alphas > 20.0)
    step = result.times[1] - result.times[0]
    lo = max(0, hot[0] - int(0.3 / step))
    hi = min(n - 1, hot[-1] + int(0.8 / step))
    pick = np.unique(
        np.concatenate([
            np.linspace(lo, peak_idx, 4).astype(int),
            np.linspace(peak_idx, hi, 3).astype(int)[1:],
        ])
    )
    ax.plot(display[lo:hi + 1, 0], display[lo:hi + 1, 1], display[lo:hi + 1, 2], color=TRAIL_COLOR, lw=2.0, alpha=0.95)
    for k in pick:
        state = RigidBodyState(result.positions[k], result.velocities_body[k], result.quaternions[k], result.rates_body[k])
        _draw_rafale(ax, state, scale=3.4)
    ax.scatter([display[peak_idx, 0]], [display[peak_idx, 1]], [display[peak_idx, 2]], color="#ffd166", marker="X", s=60, label=f"peak alpha {alphas[peak_idx]:.0f} deg")
    x_mid = display[lo:hi + 1, 0].mean()
    span = max(300.0, np.ptp(display[lo:hi + 1, 0]) * 1.25)
    _set_box(ax, x_mid, span, 120.0, 150.0)
    ax.set_zlim(display[lo:hi + 1, 2].min() - 25.0, display[lo:hi + 1, 2].max() + 45.0)
    ax.view_init(elev=14.0, azim=-108.0)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("lateral (m)", fontsize=7)
    ax.set_zlabel("altitude (m)", fontsize=7)
    ax.legend(facecolor="#1b2230", labelcolor="white", fontsize=8, loc="upper left")
    _style_3d(ax, "Attitude sequence through the cobra")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor(PANEL)
    ax2.plot(result.times, alphas, color=GLYPH_COLOR, lw=2.0, label="alpha (deg)")
    ax2.axhline(90.0, color="#d7dde8", lw=0.8, ls=":", label="90 deg")
    ax2.set_xlabel("time (s)", color="white", fontsize=9)
    ax2.set_ylabel("angle of attack (deg)", color=GLYPH_COLOR, fontsize=9)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.grid(alpha=0.22)
    ax2b = ax2.twinx()
    ax2b.plot(result.times, speeds, color=GLIDESLOPE_COLOR, lw=1.6, label="speed (m/s)")
    ax2b.set_ylabel("speed (m/s)", color=GLIDESLOPE_COLOR, fontsize=9)
    ax2b.tick_params(colors="#8fbfff", labelsize=8)
    ax2.set_title("Alpha and speed through the maneuver", color="white", fontsize=10)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, facecolor="#1b2230", labelcolor="white", fontsize=8, loc="upper right")
    m = result.metrics
    fig.text(
        0.5, 0.02,
        f"peak alpha {m['peak_alpha_deg']:.0f} deg | min speed {m['min_speed_mps']:.0f} m/s | "
        f"altitude lost {m['altitude_lost_m']:.0f} m | recovered: {m['recovered']}",
        color="#d7dde8", fontsize=9, ha="center", family="monospace",
    )

    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return output


def render_cobra_animation(result: SixDOFResult, output: Path, frames: int = 60, fps: int = 15) -> Path:
    """Animated 3D cobra: the Rafale pulls past vertical and recovers."""

    from .aircraft6dof import RigidBodyState

    display = ned_to_display(result.positions)
    frame_indices = np.unique(np.linspace(0, len(result.times) - 1, frames).astype(int))

    fig = plt.figure(figsize=(8.6, 6.2), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("flight-control-lab - scripted cobra", color="white", fontsize=12, fontweight="bold")
    status = fig.text(0.02, 0.03, "", color="#d7dde8", fontsize=9, family="monospace")

    def draw_frame(step: int):
        idx = int(frame_indices[step])
        t = float(result.times[idx])
        state = RigidBodyState(result.positions[idx], result.velocities_body[idx], result.quaternions[idx], result.rates_body[idx])
        ax.clear()
        ax.plot(display[: idx + 1, 0], display[: idx + 1, 1], display[: idx + 1, 2], color=TRAIL_COLOR, lw=1.8)
        _draw_rafale(ax, state, scale=3.0)
        _set_box(ax, display[idx, 0], 420.0, 100.0, 100.0)
        ax.set_zlim(display[idx, 2] - 70.0, display[idx, 2] + 90.0)
        ax.view_init(elev=14.0, azim=-108.0)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("lateral (m)", fontsize=7)
        ax.set_zlabel("altitude (m)", fontsize=7)
        _style_3d(ax, "Pull, float past 90 deg, push, FCS recovery")
        alpha = np.degrees(state.alpha)
        status.set_text(f"t={t:4.1f}s  alpha={alpha:6.1f} deg  speed={state.airspeed:5.1f} m/s  alt={state.altitude:6.0f} m")
        return []

    animation = FuncAnimation(fig, draw_frame, frames=len(frame_indices), interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return output
