"""3D visualisation helpers for the moving-carrier approach.

All plotting happens in a display frame where x follows the carrier heading,
y is lateral and z points up. Simulation states stay in NED; conversion lives
in :func:`ned_to_display` so there is a single place to audit the sign flip.
"""

from __future__ import annotations

import math

import numpy as np

from .aircraft6dof import RigidBodyState, quat_to_matrix
from .carrier6dof import MovingCarrier
from .arresting import ArrestingGear


def ned_to_display(positions: np.ndarray) -> np.ndarray:
    """Map NED positions (x north, y east, z down) to (x, y, z-up) for plotting."""

    positions = np.asarray(positions, dtype=float)
    display = positions.copy()
    display[..., 2] = -positions[..., 2]
    return display


def deck_mesh(carrier: MovingCarrier, t: float, resolution: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deck surface grid in the display frame at time ``t``.

    The deck tilts with the carrier's sinusoidal pitch and roll through
    :meth:`MovingCarrier.deck_plane_z_down`, so the returned mesh moves exactly
    like the deck the touchdown assessment uses.
    """

    deck = carrier.deck_state(t)
    xs = np.linspace(deck["x_start"], deck["x_start"] + carrier.length, resolution)
    ys = np.linspace(carrier.y - carrier.width / 2, carrier.y + carrier.width / 2, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_z = -np.vectorize(lambda x, y: carrier.deck_plane_z_down(x, y, t))(grid_x, grid_y)
    return grid_x, grid_y, grid_z


def wire_lines(carrier: MovingCarrier, gear: ArrestingGear, t: float) -> list[np.ndarray]:
    """Arresting wires as 3D polylines lying on the deck at time ``t``."""

    deck = carrier.deck_state(t)
    lines: list[np.ndarray] = []
    for fraction in gear.wire_fractions:
        wx = deck["x_start"] + fraction * carrier.length
        ys = np.linspace(carrier.y - carrier.width / 2, carrier.y + carrier.width / 2, 12)
        zs = -np.asarray([carrier.deck_plane_z_down(wx, y, t) for y in ys]) + 0.15
        lines.append(np.column_stack([np.full_like(ys, wx), ys, zs]))
    return lines


# Simple teaching glyph: nose, tail, wing tips, tailplane and fin in body axes.
_AIRCRAFT_FRAME = np.array(
    [
        [9.0, 0.0, 0.0],   # nose
        [-7.0, 0.0, 0.0],  # tail
        [-1.0, 8.5, 0.0],  # right wing tip
        [-1.0, -8.5, 0.0],  # left wing tip
        [-7.0, 3.0, 0.0],  # right tailplane tip
        [-7.0, -3.0, 0.0],  # left tailplane tip
        [-7.0, 0.0, -3.2],  # fin top (body z is down, so negative is up)
    ]
)
_AIRCRAFT_EDGES = [(0, 1), (2, 3), (1, 4), (1, 5), (1, 6), (0, 2), (0, 3)]


def aircraft_glyph(state: RigidBodyState, scale: float = 1.0) -> list[np.ndarray]:
    """Wireframe aircraft segments in the display frame at ``state``'s attitude."""

    rotation = quat_to_matrix(state.quaternion)
    body = _AIRCRAFT_FRAME * scale
    nav = (rotation @ body.T).T + state.position
    display = ned_to_display(nav)
    return [display[[i, j]] for i, j in _AIRCRAFT_EDGES]


def approach_window(result_times: np.ndarray, positions: np.ndarray, touchdown_x: float, max_distance: float):
    """Indices covering the final ``max_distance`` metres before touchdown."""

    distance = touchdown_x - positions[:, 0]
    mask = (distance >= 0.0) & (distance <= max_distance)
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        indices = np.arange(len(result_times))
    return indices


# ---------------------------------------------------------------------------
# Rafale-inspired teaching glyph (delta wing + canards, single fin).
# Body axes: x forward, y right, z down. Dimensions loosely Rafale-scaled
# (length ~15 m, span ~10.9 m) for a readable, recognizable silhouette.
# ---------------------------------------------------------------------------

_RAFALE_POLYGONS = {
    "wing": [
        (2.5, 0.0, 0.35), (-4.0, 5.45, 0.25), (-5.6, 5.45, 0.25),
        (-6.6, 0.0, 0.35), (-5.6, -5.45, 0.25), (-4.0, -5.45, 0.25),
    ],
    "canard": [
        (4.6, 0.0, 0.15), (2.7, 1.9, 0.15), (2.2, 1.9, 0.15),
        (1.6, 0.0, 0.15), (2.2, -1.9, 0.15), (2.7, -1.9, 0.15),
    ],
    "fin": [
        (-4.6, 0.0, -0.1), (-7.3, 0.0, -3.5), (-8.0, 0.0, -3.5), (-6.4, 0.0, -0.1),
    ],
    "fuselage_side": [
        (7.65, 0.0, 0.25), (3.0, 0.0, -0.55), (-6.9, 0.0, -0.35),
        (-7.4, 0.0, 0.35), (0.0, 0.0, 0.75),
    ],
}


def rafale_mesh(state: RigidBodyState, scale: float = 1.0) -> list[tuple[np.ndarray, str]]:
    """Rafale-like polygons in the display frame for the given state.

    Returns (polygon, part-name) pairs; the renderer owns colors.
    """

    rotation = quat_to_matrix(state.quaternion)
    out: list[tuple[np.ndarray, str]] = []
    for part, polygon in _RAFALE_POLYGONS.items():
        body = np.asarray(polygon, dtype=float) * scale
        nav = (rotation @ body.T).T + state.position
        out.append((ned_to_display(nav), part))
    return out


def _ship_transform(carrier: MovingCarrier, t: float):
    deck = carrier.deck_state(t)
    td_x = deck["touchdown_x"]

    def apply(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        z = pts[:, 2] + deck["heave"] - deck["pitch"] * (pts[:, 0] - td_x) + deck["roll"] * (pts[:, 1] - carrier.y)
        return np.column_stack([pts[:, 0], pts[:, 1], z])

    return apply


def ship_mesh(carrier: MovingCarrier, t: float) -> list[tuple[np.ndarray, str]]:
    """Hull, island and deck-edge polygons of the carrier in the display frame.

    Every part follows the same heave/pitch/roll transform as the deck plane
    used by the touchdown assessment, so the visuals match the physics.
    """

    deck = carrier.deck_state(t)
    x0 = deck["x_start"]
    x1 = x0 + carrier.length
    w = carrier.width / 2.0
    h = carrier.deck_height
    apply = _ship_transform(carrier, t)
    parts: list[tuple[np.ndarray, str]] = []

    # Hull silhouette with a tapered bow and rounded stern; the flight deck
    # overhangs the hull like a real carrier.
    hw = w * 0.78
    for side in (-1.0, 1.0):
        y = side * hw
        parts.append((
            apply([
                (x0 - 4, y, h),          # stern top
                (x1 + 2, y, h),          # deck edge runs past the hull
                (x1 + 16, y, h * 0.55),  # bow taper
                (x1 + 8, y, 0.0),        # bow at the waterline
                (x0 + 6, y, 0.0),        # waterline runs aft
                (x0 - 4, y, h * 0.15),   # rounded stern heel
            ]),
            "hull",
        ))
    parts.append((apply([(x0 - 4, -hw, h), (x0 - 4, -hw, h * 0.15), (x0 - 4, hw, h * 0.15), (x0 - 4, hw, h)]), "hull"))

    # Island on the starboard side, around 60% of the deck length.
    ix0 = x0 + 0.58 * carrier.length
    ix1 = ix0 + 14.0
    iy0 = w - 3.0
    iy1 = w + 5.0
    top = h + 11.0
    for face in [
        [(ix0, iy0, h), (ix0, iy1, h), (ix0, iy1, top), (ix0, iy0, top)],
        [(ix1, iy0, h), (ix1, iy1, h), (ix1, iy1, top), (ix1, iy0, top)],
        [(ix0, iy0, h), (ix1, iy0, h), (ix1, iy0, top), (ix0, iy0, top)],
        [(ix0, iy1, h), (ix1, iy1, h), (ix1, iy1, top), (ix0, iy1, top)],
        [(ix0, iy0, top), (ix1, iy0, top), (ix1, iy1, top), (ix0, iy1, top)],
    ]:
        parts.append((apply(face), "island"))
    # mast above the island
    mx = (ix0 + ix1) / 2.0
    my = (iy0 + iy1) / 2.0
    parts.append((
        apply([(mx - 1.2, my, top), (mx + 1.2, my, top), (mx + 0.5, my, top + 6.0), (mx - 0.5, my, top + 6.0)]),
        "island",
    ))
    return parts


def deck_markings(carrier: MovingCarrier, t: float) -> list[np.ndarray]:
    """Centerline dashes and landing-strip edges lying on the deck."""

    deck = carrier.deck_state(t)
    x0 = deck["x_start"]
    lines: list[np.ndarray] = []
    for fraction in np.linspace(0.06, 0.94, 12):
        xs = np.array([x0 + fraction * carrier.length, x0 + (fraction + 0.03) * carrier.length])
        ys = np.zeros(2)
        zs = -np.asarray([carrier.deck_plane_z_down(x, 0.0, t) for x in xs]) + 0.12
        lines.append(np.column_stack([xs, ys, zs]))
    for edge in (-carrier.width / 2 + 1.0, carrier.width / 2 - 1.0):
        xs = np.linspace(x0 + 4.0, x0 + carrier.length - 4.0, 20)
        ys = np.full_like(xs, edge)
        zs = -np.asarray([carrier.deck_plane_z_down(x, edge, t) for x in xs]) + 0.12
        lines.append(np.column_stack([xs, ys, zs]))
    return lines
