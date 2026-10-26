#!/usr/bin/env python3
"""Compare the nominal and harder moving-carrier landing scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flight_control_lab.scenarios6dof import (
    simulate_carrier_challenge_6dof,
    simulate_moving_carrier_6dof,
)


SCENARIOS = {
    "baseline": None,
    "rough seas": "rough_seas",
    "offset recovery": "offset_recovery",
    "strong airwake": "strong_airwake",
    "turbulent": "turbulent_recovery",
    "aileron fault": "aileron_fault",
}


def collect() -> dict[str, dict[str, float | str | bool]]:
    rows: dict[str, dict[str, float | str | bool]] = {}
    for label, challenge in SCENARIOS.items():
        if challenge is None:
            result, touchdown = simulate_moving_carrier_6dof()
        else:
            result, touchdown = simulate_carrier_challenge_6dof(challenge)
        rows[label] = {
            "touchdown_success": touchdown.success,
            "arrestment_success": bool(result.metrics["arrestment_success"]),
            "wire": str(result.metrics["arrestment_wire_number"]),
            "sink": float(result.metrics["relative_sink_rate_mps"]),
            "lineup": abs(float(result.metrics["lineup_error_m"])),
            "score": float(result.metrics["landing_quality_score"]),
            "grade": str(result.metrics["landing_quality_grade"]),
            "stop_margin": float(result.metrics["arrestment_stop_margin_m"]),
            "downwash": float(result.metrics["max_airwake_downwash_mps"]),
            "reason": str(result.metrics["arrestment_reason"]),
        }
    return rows


def render(rows: dict[str, dict[str, float | str | bool]], output: Path) -> None:
    labels = list(rows)
    colors = ["#8be05c" if rows[label]["arrestment_success"] else "#ff6b6b" for label in labels]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#10141b")
    fig.suptitle(
        "flight-control-lab — moving-carrier challenge comparison",
        color="white",
        fontsize=15,
        fontweight="bold",
    )

    panels = [
        ("sink", "relative sink rate (m/s)", "Touchdown vertical demand"),
        ("lineup", "absolute lineup error (m)", "Centerline correction"),
        ("score", "landing quality / 100", "Overall landing score"),
        ("stop_margin", "remaining deck margin (m)", "Rollout margin"),
    ]
    for ax, (key, ylabel, title) in zip(axes.flat, panels):
        ax.set_facecolor("#151a23")
        values = [float(rows[label][key]) for label in labels]
        bars = ax.bar(labels, values, color=colors, alpha=0.9)
        ax.set_title(title, color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.tick_params(axis="x", colors="white", rotation=22, labelsize=8)
        ax.tick_params(axis="y", colors="white")
        ax.grid(axis="y", alpha=0.22)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=8,
            )

    summary = "\n".join(
        f"{label:15s}  {rows[label]['wire']}-wire  "
        f"downwash {rows[label]['downwash']:.1f} m/s  "
        f"{rows[label]['reason']}"
        for label in labels
    )
    fig.text(0.01, 0.012, summary, color="#d7dde8", fontsize=8, family="monospace")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/carrier-challenges-6dof-preview.png"))
    args = parser.parse_args()
    rows = collect()
    render(rows, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
