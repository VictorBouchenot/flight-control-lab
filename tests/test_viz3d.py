"""Tests for the 3D visualisation helpers and renderers."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flight_control_lab.aircraft6dof import RigidBodyState, euler_to_quat, trim_level_flight, Aircraft6DOFParams
from flight_control_lab.airwake import CarrierAirwake
from flight_control_lab.arresting import ArrestingGear
from flight_control_lab.carrier6dof import MovingCarrier
from flight_control_lab.scenarios6dof import simulate_moving_carrier_6dof
from flight_control_lab.viz3d import (
    deck_markings,
    rafale_mesh,
    ship_mesh,
    aircraft_glyph,
    approach_window,
    deck_mesh,
    ned_to_display,
    wire_lines,
)
from flight_control_lab.viz3d_render import render_animation, render_still


class TestDisplayFrame(unittest.TestCase):
    def test_ned_to_display_flips_vertical_only(self):
        positions = np.array([[10.0, 5.0, -80.0], [12.0, -3.0, -18.0]])
        display = ned_to_display(positions)
        np.testing.assert_allclose(display[:, 0], positions[:, 0])
        np.testing.assert_allclose(display[:, 1], positions[:, 1])
        np.testing.assert_allclose(display[:, 2], [80.0, 18.0])
        # input array must not be mutated
        np.testing.assert_allclose(positions[:, 2], [-80.0, -18.0])


class TestDeckMesh(unittest.TestCase):
    def setUp(self):
        self.carrier = MovingCarrier()

    def test_mesh_shape_and_height(self):
        grid_x, grid_y, grid_z = deck_mesh(self.carrier, 0.0)
        self.assertEqual(grid_x.shape, (6, 6))
        self.assertEqual(grid_y.shape, (6, 6))
        deck = self.carrier.deck_state(0.0)
        # flat reference: heave/pitch/roll are all zero at t=0
        np.testing.assert_allclose(grid_z, self.carrier.deck_height, atol=1e-9)
        self.assertAlmostEqual(grid_x.min(), deck["x_start"])
        self.assertAlmostEqual(grid_x.max(), deck["x_start"] + self.carrier.length)

    def test_mesh_follows_heave(self):
        # isolate heave so pitch/roll cannot skew the deck mean
        heave_only = MovingCarrier(pitch_amplitude_deg=0.0, roll_amplitude_deg=0.0)
        t_up = heave_only.heave_period / 4.0  # maximum positive heave (deck up)
        _, _, z_up = deck_mesh(heave_only, t_up)
        _, _, z_down = deck_mesh(heave_only, 3.0 * t_up)
        self.assertGreater(z_up.mean(), z_down.mean())
        self.assertAlmostEqual(
            z_up.mean() - z_down.mean(), 2.0 * heave_only.heave_amplitude, places=1
        )

    def test_mesh_tilts_with_roll(self):
        roll_only = MovingCarrier(heave_amplitude=0.0, pitch_amplitude_deg=0.0)
        t = roll_only.roll_period / 4.0  # max roll
        _, grid_y, grid_z = deck_mesh(roll_only, t)
        # meshgrid 'xy' indexing: axis 0 spans the lateral (y) direction
        starboard = grid_z[-1, :].mean()
        port = grid_z[0, :].mean()
        expected = math.radians(roll_only.roll_amplitude_deg) * roll_only.width
        self.assertAlmostEqual(abs(starboard - port), expected, places=1)


class TestWiresAndGlyph(unittest.TestCase):
    def test_wire_lines_match_gear(self):
        carrier = MovingCarrier()
        gear = ArrestingGear()
        lines = wire_lines(carrier, gear, 1.3)
        self.assertEqual(len(lines), len(gear.wire_fractions))
        deck = carrier.deck_state(1.3)
        for line, fraction in zip(lines, gear.wire_fractions):
            np.testing.assert_allclose(line[:, 0], deck["x_start"] + fraction * carrier.length)
            self.assertTrue(np.all(line[:, 2] > 0.0))  # wires above sea level

    def test_glyph_segments_and_level_wings(self):
        params = Aircraft6DOFParams()
        state, _ = trim_level_flight(params, speed=70.0, altitude=100.0)
        segments = aircraft_glyph(state)
        self.assertEqual(len(segments), 7)
        wing = segments[1]  # wing-tip to wing-tip segment
        self.assertAlmostEqual(wing[0, 2], wing[1, 2], places=6)  # level wings: same height
        span = abs(wing[0, 1] - wing[1, 1])
        self.assertAlmostEqual(span, 17.0, places=6)

    def test_glyph_banks_with_aircraft(self):
        params = Aircraft6DOFParams()
        state, _ = trim_level_flight(params, speed=70.0, altitude=100.0)
        banked = RigidBodyState(
            state.position,
            state.velocity_body,
            euler_to_quat(math.radians(30.0), 0.0, 0.0),
            state.rates_body,
        )
        wing = aircraft_glyph(banked)[1]
        height_diff = abs(wing[0, 2] - wing[1, 2])
        self.assertAlmostEqual(height_diff, 17.0 * math.sin(math.radians(30.0)), places=1)


class TestApproachWindow(unittest.TestCase):
    def test_window_selects_final_segment(self):
        times = np.linspace(0.0, 10.0, 11)
        positions = np.column_stack([np.linspace(0.0, 1_000.0, 11), np.zeros(11), np.zeros(11)])
        window = approach_window(times, positions, touchdown_x=1_000.0, max_distance=300.0)
        self.assertEqual(window[0], 7)
        self.assertEqual(window[-1], 10)


class TestRenderSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.carrier = MovingCarrier()
        cls.gear = ArrestingGear()
        cls.result, cls.touchdown = simulate_moving_carrier_6dof(
            carrier=cls.carrier, airwake=CarrierAirwake(), arresting_gear=cls.gear, dt=0.05
        )

    def test_still_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = render_still(self.result, self.carrier, self.gear, self.touchdown, Path(tmp) / "still.png")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 10_000)

    def test_animation_writes_gif(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = render_animation(
                self.result, self.carrier, self.gear, self.touchdown,
                Path(tmp) / "anim.gif", frames=4, fps=4,
            )
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 5_000)
            with open(out, "rb") as handle:
                self.assertEqual(handle.read(6), b"GIF89a")


if __name__ == "__main__":
    unittest.main()


class TestRafaleMesh(unittest.TestCase):
    def test_parts_present_and_sized(self):
        params = Aircraft6DOFParams()
        state, _ = trim_level_flight(params, speed=100.0, altitude=500.0)
        mesh = rafale_mesh(state)
        parts = {part for _, part in mesh}
        self.assertEqual(parts, {"wing", "canard", "fin", "fuselage_side"})
        wing = dict((part, poly) for poly, part in mesh)["wing"]
        span = wing[:, 1].max() - wing[:, 1].min()
        self.assertAlmostEqual(span, 10.9, places=5)

    def test_mesh_follows_position_and_bank(self):
        params = Aircraft6DOFParams()
        level, _ = trim_level_flight(params, speed=100.0, altitude=500.0)
        level_wing = dict((part, poly) for poly, part in rafale_mesh(level))["wing"]
        banked = RigidBodyState(
            level.position + np.array([50.0, 0.0, -20.0]),
            level.velocity_body,
            euler_to_quat(math.radians(40.0), 0.0, 0.0),
            level.rates_body,
        )
        banked_wing = dict((part, poly) for poly, part in rafale_mesh(banked))["wing"]
        # position offset carried through (same attitude family, x shifted by 50)
        self.assertAlmostEqual(float(banked_wing[:, 0].mean() - level_wing[:, 0].mean()), 50.0, delta=0.5)
        # 40 deg bank spreads the wingtips in height by roughly span * sin(40)
        z_spread = float(banked_wing[:, 2].max() - banked_wing[:, 2].min())
        self.assertGreater(z_spread, 10.9 * math.sin(math.radians(40.0)) * 0.9)


class TestShipMesh(unittest.TestCase):
    def test_parts_and_motion(self):
        carrier = MovingCarrier()
        parts = ship_mesh(carrier, 0.0)
        kinds = {part for _, part in parts}
        self.assertEqual(kinds, {"hull", "island"})
        # island top sits above the deck
        island_top = max(float(poly[:, 2].max()) for poly, part in parts if part == "island")
        self.assertGreater(island_top, carrier.deck_height + 8.0)
        # hull bottom at sea level when heave is zero
        hull_bottom = min(float(poly[:, 2].min()) for poly, part in parts if part == "hull")
        self.assertAlmostEqual(hull_bottom, 0.0, places=6)

    def test_ship_follows_heave(self):
        carrier = MovingCarrier(pitch_amplitude_deg=0.0, roll_amplitude_deg=0.0)
        t_up = carrier.heave_period / 4.0
        bottom_up = min(float(poly[:, 2].min()) for poly, part in ship_mesh(carrier, t_up) if part == "hull")
        self.assertAlmostEqual(bottom_up, carrier.heave_amplitude, places=3)

    def test_markings_on_deck(self):
        carrier = MovingCarrier()
        t = 1.7
        deck = carrier.deck_state(t)
        for line in deck_markings(carrier, t):
            self.assertTrue(np.all(line[:, 0] >= deck["x_start"]))
            self.assertTrue(np.all(line[:, 0] <= deck["x_start"] + carrier.length))
            self.assertTrue(np.all(line[:, 2] > 0.0))
