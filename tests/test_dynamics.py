import math
import unittest

from flight_control_lab.dynamics import (
    AircraftParams,
    PointMassState,
    RawControl,
    derivatives,
    drag_force,
    rk4_step,
)


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.params = AircraftParams()

    def test_level_equilibrium_is_stable(self):
        state = PointMassState(0, 0, 1000, 150, 0, 0)
        drag = drag_force(state, self.params, 1.0)
        control = RawControl(drag / self.params.max_thrust, 1.0, 0.0)
        out = derivatives(state, control, self.params)
        self.assertAlmostEqual(out[3], 0.0, places=5)
        self.assertAlmostEqual(out[4], 0.0, places=5)
        self.assertAlmostEqual(out[5], 0.0, places=7)

    def test_climb_increases_altitude(self):
        state = PointMassState(0, 0, 1000, 150, math.radians(5), 0)
        drag = drag_force(state, self.params, 1.1)
        control = RawControl(min(1.0, (drag + 12000) / self.params.max_thrust), 1.1, 0.0)
        new_state = rk4_step(state, control, self.params, 0.05)
        self.assertGreater(new_state.z, state.z)

    def test_bank_turns_heading(self):
        state = PointMassState(0, 0, 1000, 150, 0, 0)
        control = RawControl(0.7, 1.2, math.radians(30))
        out = derivatives(state, control, self.params)
        self.assertGreater(out[5], 0.0)

    def test_idle_thrust_slows_aircraft(self):
        state = PointMassState(0, 0, 1000, 180, 0, 0)
        out = derivatives(state, RawControl(0.0, 1.0, 0.0), self.params)
        self.assertLess(out[3], 0.0)

    def test_direct_controls_are_clamped(self):
        state = PointMassState(0, 0, 1000, 150, 0, 0)
        out = derivatives(state, RawControl(2.0, 20.0, math.radians(120)), self.params)
        self.assertLessEqual(out[4], 5.0)
        self.assertLessEqual(out[5], 1.0)

    def test_invalid_dt_rejected(self):
        state = PointMassState(0, 0, 1000, 150, 0, 0)
        with self.assertRaises(ValueError):
            rk4_step(state, RawControl(0.5, 1.0, 0.0), self.params, 0.0)


if __name__ == "__main__":
    unittest.main()
