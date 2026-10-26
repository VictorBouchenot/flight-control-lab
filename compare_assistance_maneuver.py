"""Pilot-assistance comparison preview: 120 deg heading capture, FCS on vs off.

Same teaching pilot, same aircraft trim, same target heading. In "assisted"
mode the FCS shapes, damps and limits the response; in "raw" mode the stick
drives the surfaces directly. A shadow pilot records its stick on the assisted
trajectory, so the stick-vs-surface panel compares like with like.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flight_control_lab.assistance import simulate_maneuver_assist_comparison
from flight_control_lab.aircraft6dof import Aircraft6DOFParams, RigidBodyState

OUT = Path("/tmp/assistance-maneuver.png")
TARGET_HEADING_DEG = 120.0
TARGET_ALTITUDE_M = 2_000.0


def _states(result) -> list[RigidBodyState]:
    return [
        RigidBodyState(result.positions[i], result.velocities_body[i], result.quaternions[i], result.rates_body[i])
        for i in range(len(result.times))
    ]


def main() -> None:
    comparison = simulate_maneuver_assist_comparison()
    params = Aircraft6DOFParams()
    assisted, raw = comparison["assisted_result"], comparison["raw_result"]
    a_states, r_states = _states(assisted), _states(raw)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "120 deg heading capture: automatic FCS vs pilot model (different control laws)",
        fontsize=13, fontweight="bold",
    )
    gs = fig.add_gridspec(2, 3)

    # 1. Heading vs target
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(assisted.times, [np.degrees(s.euler[2]) for s in a_states], lw=1.6, color="tab:blue", label="assisted (FCS)")
    ax.plot(raw.times, [np.degrees(s.euler[2]) for s in r_states], lw=1.6, color="tab:red", label="raw stick")
    ax.axhline(TARGET_HEADING_DEG, color="k", ls=":", lw=1, label="target 120 deg")
    ax.set_title("Heading capture")
    ax.set_xlabel("time (s)"); ax.set_ylabel("heading (deg)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 2. Bank vs FCS cap
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(assisted.times, [np.degrees(s.euler[0]) for s in a_states], lw=1.4, color="tab:blue")
    ax.plot(raw.times, [np.degrees(s.euler[0]) for s in r_states], lw=1.4, color="tab:red")
    cap = 45.0
    ax.axhline(cap, color="k", ls="--", lw=1); ax.axhline(-cap, color="k", ls="--", lw=1, label=f"FCS bank cap +/-{cap:.0f} deg")
    ax.set_title("Bank angle vs FCS cap")
    ax.set_xlabel("time (s)"); ax.set_ylabel("bank (deg)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 3. Alpha vs envelope limit
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(assisted.times, [np.degrees(s.alpha) for s in a_states], lw=1.4, color="tab:blue")
    ax.plot(raw.times, [np.degrees(s.alpha) for s in r_states], lw=1.4, color="tab:red")
    ax.axhline(np.degrees(params.alpha_max), color="k", ls="--", lw=1, label=f"FCS limit {np.degrees(params.alpha_max):.0f} deg")
    ax.set_title("Angle of attack vs envelope limit")
    ax.set_xlabel("time (s)"); ax.set_ylabel("alpha (deg)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 4. Stick vs surface (same assisted trajectory)
    ax = fig.add_subplot(gs[1, 0])
    stick = comparison["pilot_stick"]["aileron"]
    n = min(len(stick), len(assisted.times) - 1)
    ax.plot(assisted.times[1 : n + 1], stick[:n], lw=1.2, color="tab:orange", label="shadow pilot command, NOT an FCS input")
    surf = assisted.surface_history[:, 2]
    ax.plot(assisted.times[: len(surf)], surf, lw=1.4, color="tab:blue", label="actual FCS aileron surface")
    ax.set_title("Shadow command (not applied) vs FCS surface")
    ax.set_xlabel("time (s)"); ax.set_ylabel("rad (both in aileron-surface units)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 5. Altitude deviation
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(assisted.times, [s.altitude - TARGET_ALTITUDE_M for s in a_states], lw=1.6, color="tab:blue")
    ax.plot(raw.times, [s.altitude - TARGET_ALTITUDE_M for s in r_states], lw=1.6, color="tab:red")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_title("Altitude deviation from target")
    ax.set_xlabel("time (s)"); ax.set_ylabel("altitude error (m)")
    ax.grid(alpha=0.3)

    # 6. Outcome summary
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    a, r = comparison["assisted"], comparison["raw"]
    def f(v, nd=1):
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)
    rows = [
        ("", "assisted", "raw"),
        ("max bank (deg)", f(a["max_bank_deg"]), f(r["max_bank_deg"])),
        ("max alpha (deg)", f(a["max_alpha_deg"]), f(r["max_alpha_deg"])),
        ("altitude err (m)", f(a["max_abs_altitude_error_m"]), f(r["max_abs_altitude_error_m"])),
        ("settle time (s)", f(a["settle_time_s"]), "never" if not r["clean_capture"] else f(r["settle_time_s"])),
        ("outcome", "clean capture", "departure (rolled inverted)"),
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.scale(1, 1.6)
    for j in range(3):
        table[0, j].set_facecolor("#dde6f2")
    ax.set_title("Outcome (identical scenario)", fontsize=11)
    ax.text(0.5, -0.08,
            "Results of this teaching model and scenario, not evidence of\nreal-aircraft or real-pilot fidelity.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, style="italic")

    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")
    for mode in ("assisted", "raw"):
        m = comparison[mode]
        print(mode, {k: round(v, 2) if isinstance(v, float) else v for k, v in m.items() if k in ("max_bank_deg", "max_alpha_deg", "max_abs_altitude_error_m", "settle_time_s", "heading_overshoot_deg", "lost_control", "clean_capture")})


if __name__ == "__main__":
    main()
