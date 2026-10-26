import math
import unittest

from flight_control_lab.airwake import CarrierAirwake
from flight_control_lab.scenarios6dof import (
    simulate_carrier_challenge_6dof,
    simulate_maneuver_6dof,
    simulate_moving_carrier_6dof,
    simulate_takeoff_6dof,
)


class SixDOFScenarioTests(unittest.TestCase):
    def test_takeoff_rotates_and_captures_altitude(self):
        result = simulate_takeoff_6dof(dt=0.05)
        self.assertTrue(result.metrics["rotated"])
        self.assertTrue(result.metrics["target_altitude_reached"])
        self.assertLess(result.metrics["rotation_distance_m"], 1500)

    def test_maneuver_heading_capture(self):
        result = simulate_maneuver_6dof(dt=0.05)
        self.assertTrue(result.metrics["heading_captured"])
        self.assertLess(result.metrics["max_alpha_deg"], 14)
        self.assertLess(result.metrics["max_abs_beta_deg"], 11)

    def test_moving_carrier_touchdown(self):
        result, touchdown = simulate_moving_carrier_6dof(dt=0.05)
        self.assertTrue(touchdown.touched)
        self.assertTrue(touchdown.success, touchdown.reason)
        self.assertLessEqual(touchdown.relative_sink_rate, 4.5)
        self.assertLessEqual(abs(touchdown.lineup_error), 16)
        self.assertTrue(result.metrics["airwake_enabled"])
        self.assertGreater(result.metrics["max_airwake_downwash_mps"], 0.5)
        self.assertGreater(result.metrics["landing_quality_score"], 65.0)
        self.assertIn(result.metrics["landing_quality_grade"], {"A", "B"})

    def test_harder_carrier_challenges_stay_inside_teaching_limits(self):
        for challenge in ("rough_seas", "offset_recovery", "strong_airwake"):
            with self.subTest(challenge=challenge):
                result, touchdown = simulate_carrier_challenge_6dof(challenge, dt=0.05)
                self.assertTrue(touchdown.success, touchdown.reason)
                self.assertTrue(result.metrics["arrestment_success"], result.metrics["arrestment_reason"])
                self.assertEqual(result.metrics["challenge"], challenge)

    def test_airwake_can_be_disabled_for_baseline_comparison(self):
        result, touchdown = simulate_moving_carrier_6dof(
            airwake=CarrierAirwake(enabled=False),
            dt=0.05,
        )
        self.assertTrue(touchdown.success, touchdown.reason)
        self.assertFalse(result.metrics["airwake_enabled"])
        self.assertEqual(result.metrics["max_airwake_downwash_mps"], 0.0)


if __name__ == "__main__":
    unittest.main()
