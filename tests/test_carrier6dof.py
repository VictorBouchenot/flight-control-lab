import math
import unittest

from flight_control_lab.aircraft6dof import Aircraft6DOFParams, RigidBodyState, euler_to_quat, trim_level_flight
from flight_control_lab.carrier6dof import MovingCarrier, assess_moving_touchdown
import numpy as np


class MovingCarrierTests(unittest.TestCase):
    def test_carrier_moves_and_heaves(self):
        ship = MovingCarrier()
        first = ship.deck_state(0.0)
        later = ship.deck_state(5.0)
        self.assertGreater(later["touchdown_x"], first["touchdown_x"])
        self.assertNotEqual(first["z_down"], later["z_down"])
        self.assertNotEqual(first["roll"], later["roll"])

    def test_desired_altitude_intercepts_deck(self):
        ship = MovingCarrier()
        t = 3.0
        deck = ship.deck_state(t)
        self.assertAlmostEqual(ship.desired_altitude(deck["touchdown_x"], t), -deck["z_down"])
        behind = ship.desired_altitude(deck["touchdown_x"] - 100, t)
        self.assertGreater(behind, -deck["z_down"])

    def test_touchdown_assessment_accepts_good_state(self):
        ship = MovingCarrier()
        t = 0.0
        deck = ship.deck_state(t)
        params = Aircraft6DOFParams()
        approach_speed = 79.0
        state = RigidBodyState(
            np.array([deck["touchdown_x"], 0.0, deck["z_down"]]),
            np.array([approach_speed, 0.0, 3.0]),
            euler_to_quat(0, 0, 0),
            np.zeros(3),
        )
        touchdown = assess_moving_touchdown(state, ship, t, approach_speed)
        self.assertTrue(touchdown.success, touchdown.reason)

    def test_touchdown_assessment_rejects_offset_and_sink(self):
        ship = MovingCarrier()
        t = 0.0
        deck = ship.deck_state(t)
        state = RigidBodyState(
            np.array([deck["touchdown_x"], ship.width, deck["z_down"]]),
            np.array([79.0, 0.0, 10.0]),
            euler_to_quat(0, 0, 0),
            np.zeros(3),
        )
        touchdown = assess_moving_touchdown(state, ship, t, 79.0)
        self.assertFalse(touchdown.success)

    def test_invalid_motion_rejected(self):
        with self.assertRaises(ValueError):
            MovingCarrier(speed=100).validate()


if __name__ == "__main__":
    unittest.main()
