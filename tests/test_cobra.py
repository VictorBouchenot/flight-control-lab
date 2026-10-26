"""Tests for the post-stall aerodynamic extension and the cobra scenario."""

from __future__ import annotations

import math
import unittest

import numpy as np

from flight_control_lab.aircraft6dof import (
    Aircraft6DOFParams,
    RigidBodyState,
    SurfaceCommand,
    aerodynamics,
    euler_to_quat,
)
from flight_control_lab.cobra import simulate_cobra_6dof


def _state_at_alpha(alpha_deg: float, speed: float = 120.0, altitude: float = 3_000.0) -> RigidBodyState:
    alpha = math.radians(alpha_deg)
    return RigidBodyState(
        position=np.array([0.0, 0.0, -altitude]),
        velocity_body=np.array([speed * math.cos(alpha), 0.0, speed * math.sin(alpha)]),
        quaternion=euler_to_quat(0.0, alpha, 0.0),
        rates_body=np.zeros(3),
    )


class TestAlphaDefinition(unittest.TestCase):
    def test_alpha_exceeds_ninety_when_flow_reverses(self):
        state = _state_at_alpha(120.0)
        self.assertAlmostEqual(math.degrees(state.alpha), 120.0, places=6)

    def test_alpha_still_normal_in_flight_regime(self):
        state = _state_at_alpha(6.0)
        self.assertAlmostEqual(math.degrees(state.alpha), 6.0, places=6)


class TestPostStallAero(unittest.TestCase):
    def setUp(self):
        self.params = Aircraft6DOFParams(post_stall=True)
        self.plain = Aircraft6DOFParams()
        self.trim = SurfaceCommand(0.5, 0.0, 0.0, 0.0)

    def test_disabled_by_default(self):
        # below stall the extension changes nothing
        pre = _state_at_alpha(12.0)
        _, _, snap_plain = aerodynamics(pre, self.trim, self.plain)
        _, _, snap_ext = aerodynamics(pre, self.trim, self.params)
        self.assertEqual(snap_plain.coefficients[0], snap_ext.coefficients[0])
        # above stall the two models genuinely differ
        post = _state_at_alpha(25.0)
        _, _, snap_plain_post = aerodynamics(post, self.trim, self.plain)
        _, _, snap_ext_post = aerodynamics(post, self.trim, self.params)
        self.assertNotEqual(snap_plain_post.coefficients[0], snap_ext_post.coefficients[0])

    def test_blend_continuous_at_stall(self):
        eps = 0.05
        _, _, before = aerodynamics(
            _state_at_alpha(math.degrees(self.params.stall_alpha) - eps), self.trim, self.params
        )
        _, _, after = aerodynamics(
            _state_at_alpha(math.degrees(self.params.stall_alpha) + eps), self.trim, self.params
        )
        self.assertAlmostEqual(before.coefficients[0], after.coefficients[0], delta=0.05)
        self.assertAlmostEqual(before.coefficients[1], after.coefficients[1], delta=0.05)

    def test_drag_grows_toward_flat_plate(self):
        _, _, at_stall = aerodynamics(_state_at_alpha(16.0), self.trim, self.params)
        _, _, deep = aerodynamics(_state_at_alpha(80.0), self.trim, self.params)
        self.assertGreater(deep.coefficients[1], at_stall.coefficients[1] + 0.5)

    def test_flat_plate_lift_shape(self):
        # flat-plate teaching model: CL keeps growing to a peak near 45 deg,
        # then decays toward zero at 90 deg
        _, _, at_stall = aerodynamics(_state_at_alpha(16.0), self.trim, self.params)
        _, _, at_peak = aerodynamics(_state_at_alpha(45.0), self.trim, self.params)
        _, _, near_vertical = aerodynamics(_state_at_alpha(88.0), self.trim, self.params)
        self.assertGreater(at_peak.coefficients[0], at_stall.coefficients[0])
        self.assertLess(near_vertical.coefficients[0], 0.3 * at_stall.coefficients[0])

    def test_validation_guards(self):
        with self.assertRaises(ValueError):
            Aircraft6DOFParams(post_stall=True, stall_alpha=math.radians(5.0)).validate()
        with self.assertRaises(ValueError):
            Aircraft6DOFParams(post_stall=True, post_stall_cm_factor=0.0).validate()


class TestCobraScenario(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = simulate_cobra_6dof()

    def test_reaches_cobra_alpha(self):
        self.assertTrue(self.result.metrics["cobra_reached"])
        self.assertGreaterEqual(self.result.metrics["peak_alpha_deg"], 90.0)
        self.assertLess(self.result.metrics["peak_alpha_deg"], 140.0)

    def test_recovers_level_flight(self):
        self.assertTrue(self.result.metrics["recovered"])
        self.assertGreater(self.result.metrics["final_speed_mps"], 100.0)
        self.assertLess(abs(self.result.metrics["final_altitude_m"] - self.result.metrics["entry_altitude_m"]), 400.0)

    def test_speed_bleeds_and_rebuilds(self):
        self.assertLess(self.result.metrics["min_speed_mps"], 0.8 * self.result.metrics["entry_speed_mps"])

    def test_deterministic(self):
        again = simulate_cobra_6dof()
        self.assertEqual(self.result.metrics["peak_alpha_deg"], again.metrics["peak_alpha_deg"])
        self.assertEqual(self.result.metrics["recovered"], again.metrics["recovered"])

    def test_requires_post_stall_params(self):
        with self.assertRaises(ValueError):
            simulate_cobra_6dof(params=Aircraft6DOFParams())

    def test_history_consistent(self):
        n = len(self.result.times)
        self.assertEqual(self.result.positions.shape, (n, 3))
        self.assertEqual(self.result.quaternions.shape, (n, 4))
        # altitude history stays plausible for an air demo
        self.assertGreater(np.min(-self.result.positions[:, 2]), 250.0)


if __name__ == "__main__":
    unittest.main()
