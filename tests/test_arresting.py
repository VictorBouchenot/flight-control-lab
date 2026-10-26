import unittest

import numpy as np

from flight_control_lab.aircraft6dof import trim_level_flight
from flight_control_lab.arresting import ArrestingGear, evaluate_arrestment
from flight_control_lab.carrier6dof import MovingCarrier, assess_moving_touchdown
from flight_control_lab.scenarios6dof import simulate_moving_carrier_6dof


class ArrestingGearTests(unittest.TestCase):
    def test_invalid_gear_rejected(self):
        with self.assertRaises(ValueError):
            ArrestingGear(target_wire=5).validate()
        with self.assertRaises(ValueError):
            ArrestingGear(max_cable_payout_m=0).validate()

    def test_successful_scenario_scores_three_wire_rollout(self):
        result, touchdown = simulate_moving_carrier_6dof(dt=0.05)
        self.assertTrue(touchdown.success, touchdown.reason)
        self.assertTrue(result.metrics["arrestment_engaged"], result.metrics["arrestment_reason"])
        self.assertTrue(result.metrics["arrestment_success"], result.metrics["arrestment_reason"])
        self.assertEqual(result.metrics["arrestment_wire_number"], 3)
        self.assertGreater(result.metrics["arrestment_stop_margin_m"], 10.0)
        self.assertGreater(result.metrics["arrestment_score"], 55.0)

    def test_missed_touchdown_is_not_an_arrestment(self):
        carrier = MovingCarrier()
        state, _ = trim_level_flight(speed=90.0, altitude=18.0)
        state = type(state)(
            np.array([carrier.touchdown_x(0.0), 20.0, -18.0]),
            state.velocity_body,
            state.quaternion,
            state.rates_body,
        )
        touchdown = assess_moving_touchdown(state, carrier, 0.0, 70.0)
        arrestment = evaluate_arrestment(state, touchdown, carrier, 0.0)
        self.assertFalse(touchdown.success)
        self.assertFalse(arrestment.engaged)
        self.assertFalse(arrestment.success)
        self.assertIn("touchdown", arrestment.reason)


if __name__ == "__main__":
    unittest.main()
