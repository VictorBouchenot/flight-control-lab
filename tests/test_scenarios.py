import math
import unittest

from flight_control_lab.carrier import CarrierDeck, assess_touchdown
from flight_control_lab.dynamics import PointMassState
from flight_control_lab.scenarios import (
    simulate_carrier_approach,
    simulate_maneuver_assist,
    simulate_takeoff,
)


class ScenarioTests(unittest.TestCase):
    def test_takeoff_rotates_and_reaches_altitude(self):
        result = simulate_takeoff(dt=0.05)
        self.assertTrue(result.metrics["rotated"])
        self.assertTrue(result.metrics["target_altitude_reached"])
        self.assertLess(result.metrics["rotation_distance_m"], 1200)
        self.assertGreaterEqual(result.final_state().speed, 65)

    def test_maneuver_captures_heading_inside_limits(self):
        result = simulate_maneuver_assist(dt=0.05)
        self.assertTrue(result.metrics["heading_captured"])
        self.assertLess(result.metrics["max_abs_bank_deg"], 76)
        self.assertLess(result.metrics["max_load_factor"], 7.1)
        self.assertGreater(result.metrics["min_load_factor"], -2.6)

    def test_carrier_approach_lands_on_fixed_deck(self):
        result, touchdown = simulate_carrier_approach(dt=0.05)
        self.assertTrue(touchdown.touched)
        self.assertTrue(touchdown.success, touchdown.reason)
        self.assertLessEqual(touchdown.sink_rate, 4.5)
        self.assertLessEqual(abs(touchdown.lineup_error_m), 15)
        self.assertLessEqual(abs(result.metrics["final_lineup_error_m"]), 15)

    def test_touchdown_classifier_rejects_high_sink(self):
        deck = CarrierDeck()
        state = PointMassState(
            deck.touchdown_x,
            0,
            deck.altitude,
            74,
            math.radians(-12),
            0,
        )
        touchdown = assess_touchdown(state, deck, approach_speed=74)
        self.assertFalse(touchdown.success)
        self.assertIn("sink", touchdown.reason)

    def test_carrier_scenario_is_deterministic(self):
        first, touchdown_first = simulate_carrier_approach(dt=0.05)
        second, touchdown_second = simulate_carrier_approach(dt=0.05)
        self.assertEqual(first.states.shape, second.states.shape)
        self.assertAlmostEqual(touchdown_first.x, touchdown_second.x)
        self.assertAlmostEqual(touchdown_first.sink_rate, touchdown_second.sink_rate)

    def test_invalid_scenario_timing_rejected(self):
        with self.assertRaises(ValueError):
            simulate_takeoff(dt=0.0)
        with self.assertRaises(ValueError):
            simulate_carrier_approach(timeout=0.0)


if __name__ == "__main__":
    unittest.main()
