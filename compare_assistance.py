"""Pilot-assistance comparison preview: identical carrier approach, FCS on vs off.

One pilot model flies the same moving-deck approach twice. In "assisted" mode
the flight control system shapes, damps and limits the stick commands; in "raw"
mode they reach the surfaces directly. The figure makes the assistance visible:
stick vs surface, envelope excursions, glidepath tracking and the outcome.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flight_control_lab.assistance import simulate_carrier_assist_comparison
from flight_control_lab.aircraft6dof import Aircraft6DOFParams, RigidBodyState

OUT = Path("/tmp/assistance-compare.png")


def _states(result) -> list[RigidBodyState]:
    return [
        RigidBodyState(result.positions[i], result.velocities_body[i], result.quaternions[i], result.rates_body[i])
        for i in range(len(result.times))
    ]


def main() -> None:
    comparison = simulate_carrier_assist_comparison()
    params = Aircraft6DOFParams()
    assisted, raw = comparison["assisted_result"], comparison["raw_result"]
    a_states, r_states = _states(assisted), _states(raw)

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Carrier approach: automatic FCS vs pilot model (different control laws)",
        fontsize=13, fontweight="bold",
    )
    gs = fig.add_gridspec(2, 3)

    # 1. Glidepath tracking
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(assisted.times, -np.asarray(assisted.positions)[:, 2], lw=1.6, color="tab:blue", label="assisted (FCS)")
    ax.plot(raw.times, -np.asarray(raw.positions)[:, 2], lw=1.6, color="tab:red", label="raw stick")
    ax.set_title("Altitude profile")
    ax.set_xlabel("time (s)"); ax.set_ylabel("altitude (m)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 2. Angle of attack vs envelope limit
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(assisted.times, [np.degrees(s.alpha) for s in a_states], lw=1.4, color="tab:blue")
    ax.plot(raw.times, [np.degrees(s.alpha) for s in r_states], lw=1.4, color="tab:red")
    ax.axhline(np.degrees(params.alpha_max), color="k", ls="--", lw=1, label=f"FCS limit {np.degrees(params.alpha_max):.0f} deg")
    ax.set_title("Angle of attack vs envelope limit")
    ax.set_xlabel("time (s)"); ax.set_ylabel("alpha (deg)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 3. Bank angle vs envelope limit
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(assisted.times, [np.degrees(s.euler[0]) for s in a_states], lw=1.4, color="tab:blue")
    ax.plot(raw.times, [np.degrees(s.euler[0]) for s in r_states], lw=1.4, color="tab:red")
    lim = np.degrees(params.max_bank)
    ax.axhline(lim, color="k", ls="--", lw=1); ax.axhline(-lim, color="k", ls="--", lw=1, label=f"FCS limit +/-{lim:.0f} deg")
    ax.set_title("Bank angle vs envelope limit")
    ax.set_xlabel("time (s)"); ax.set_ylabel("bank (deg)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 4. What the assistance does: pilot stick vs actual elevator surface
    ax = fig.add_subplot(gs[1, 0])
    stick = comparison["pilot_stick"]["elevator"]
    n = min(len(stick), len(raw.times) - 1)
    ax.plot(raw.times[1 : n + 1], stick[:n], lw=1.2, color="tab:orange", label="pilot stick, raw run (surface-equivalent)")
    surf = assisted.surface_history[:, 1]
    ax.plot(assisted.times[: len(surf)], surf, lw=1.4, color="tab:blue", label="FCS elevator surface, assisted run")
    ax.set_title("Stick effort raw vs surface assisted, same approach")
    ax.set_xlabel("time (s)"); ax.set_ylabel("rad (both in elevator-surface units)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 5. Centerline tracking
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(assisted.times, np.asarray(assisted.positions)[:, 1], lw=1.6, color="tab:blue")
    ax.plot(raw.times, np.asarray(raw.positions)[:, 1], lw=1.6, color="tab:red")
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_title("Lateral position (0 = deck centerline)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)

    # 6. Outcome summary
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    a, r, d = comparison["assisted"], comparison["raw"], comparison["delta"]
    def f(v, nd=1):
        return f"{v:.{nd}f}" if isinstance(v, float) else str(v)
    rows = [
        ("", "assisted", "raw"),
        ("max alpha (deg)", f(a["max_alpha_deg"]), f(r["max_alpha_deg"])),
        ("max bank (deg)", f(a["max_bank_deg"]), f(r["max_bank_deg"])),
        ("glidepath err (m)", f(a["max_abs_glidepath_error_m"]), f(r["max_abs_glidepath_error_m"])),
        ("landing score", f(a["landing_quality_score"]), f(r["landing_quality_score"])),
        ("outcome", "trapped on deck", "departure, off deck"),
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.scale(1, 1.6)
    for j in range(3):
        table[0, j].set_facecolor("#dde6f2")
    ax.set_title("Outcome (identical scenario)", fontsize=11)
    ax.text(0.5, -0.08,
            "Results of this teaching model and scenario; the score is the\nproject's internal scorecard, not independent validation.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, style="italic")

    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")
    print("assisted:", {k: round(v, 2) if isinstance(v, float) else v for k, v in a.items() if k in ("max_alpha_deg", "max_bank_deg", "max_abs_glidepath_error_m", "landing_quality_score", "lost_control", "touchdown")})
    print("raw:     ", {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items() if k in ("max_alpha_deg", "max_bank_deg", "max_abs_glidepath_error_m", "landing_quality_score", "lost_control", "touchdown")})
    print("delta score:", round(d["landing_quality_score"], 1))


if __name__ == "__main__":
    main()
