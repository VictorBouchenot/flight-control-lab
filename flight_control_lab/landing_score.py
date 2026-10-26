"""Landing-quality scorecard for the carrier scenarios."""

from __future__ import annotations

from .dynamics import clamp


def landing_quality_metrics(metrics: dict[str, float | bool | str]) -> dict[str, float | str]:
    """Compute a transparent 0-100 landing score from existing metrics."""

    sink_score = clamp(1.0 - float(metrics["relative_sink_rate_mps"]) / 6.0, 0.0, 1.0)
    lineup_score = clamp(1.0 - abs(float(metrics["lineup_error_m"])) / 8.0, 0.0, 1.0)
    glide_score = clamp(1.0 - float(metrics["max_abs_glidepath_error_m"]) / 45.0, 0.0, 1.0)
    speed_score = clamp(1.0 - abs(float(metrics["relative_speed_mps"]) - 58.0) / 20.0, 0.0, 1.0)
    arrestment_score = clamp(float(metrics["arrestment_score"]) / 100.0, 0.0, 1.0)
    score = 100.0 * (
        0.20 * sink_score
        + 0.20 * lineup_score
        + 0.15 * glide_score
        + 0.15 * speed_score
        + 0.30 * arrestment_score
    )
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 55:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"
    return {
        "landing_sink_score": 100.0 * sink_score,
        "landing_lineup_score": 100.0 * lineup_score,
        "landing_glidepath_score": 100.0 * glide_score,
        "landing_speed_score": 100.0 * speed_score,
        "landing_arrestment_component": 100.0 * arrestment_score,
        "landing_quality_score": score,
        "landing_quality_grade": grade,
    }
