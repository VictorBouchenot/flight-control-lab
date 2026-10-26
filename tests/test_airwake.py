import math
import unittest

import numpy as np

from flight_control_lab.aircraft6dof import Aircraft6DOFParams, aerodynamics, trim_level_flight
from flight_control_lab.airwake import CarrierAirwake
from flight_control_lab.carrier6dof import MovingCarrier


class CarrierAirwakeTests(unittest.TestCase):
    def setUp(self):
        self.carrier = MovingCarrier()
        self.wake = CarrierAirwake()

    def test_disabled_airwake_is_zero(self):
        wake = CarrierAirwake(enabled=False)
        wind = wake.wind_nav(np.array([4_350.0, 0.0, -60.0]), self.carrier, 3.0)
        np.testing.assert_allclose(wind, np.zeros(3))

    def test_burble_is_localized_downwash_behind_carrier(self):
        deck = self.carrier.deck_state(2.0)
        near = np.array([deck["x_start"] - self.wake.peak_distance_m, 0.0, deck["z_down"] - 20.0])
        far = np.array([deck["x_start"] + 250.0, 0.0, deck["z_down"] - 20.0])
        wind = self.wake.wind_nav(near, self.carrier, 2.0)
        far_wind = self.wake.wind_nav(far, self.carrier, 2.0)
        self.assertGreater(wind[2], 1.0)  # NED +z is downward
        self.assertLess(wind[0], -0.5)
        np.testing.assert_allclose(far_wind, np.zeros(3))

    def test_airwake_is_deterministic(self):
        position = np.array([4_390.0, 4.0, -45.0])
        first = self.wake.wind_nav(position, self.carrier, 7.25)
        second = self.wake.wind_nav(position, self.carrier, 7.25)
        np.testing.assert_allclose(first, second)

    def test_headwind_increases_relative_dynamic_pressure(self):
        state, surfaces = trim_level_flight(speed=110.0)
        params = Aircraft6DOFParams()
        calm = aerodynamics(state, surfaces, params)[2]
        headwind = aerodynamics(state, surfaces, params, wind_body=np.array([-8.0, 0.0, 0.0]))[2]
        self.assertGreater(headwind.dynamic_pressure, calm.dynamic_pressure)

    def test_invalid_airwake_rejected(self):
        with self.assertRaises(ValueError):
            CarrierAirwake(distance_scale_m=0.0).validate()
        with self.assertRaises(ValueError):
            CarrierAirwake(unsteadiness=0.9).validate()


if __name__ == "__main__":
    unittest.main()
