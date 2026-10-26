import math
import unittest

import numpy as np

from flight_control_lab.actuators import ActuatorState, step_actuators
from flight_control_lab.aircraft6dof import (
    Aircraft6DOFParams,
    RigidBodyState,
    SurfaceCommand,
    aerodynamics,
    euler_to_quat,
    quat_to_euler,
    quat_to_matrix,
    rigid_body_derivatives,
    rk4_step_6dof,
    trim_level_flight,
)


class Aircraft6DOFTests(unittest.TestCase):
    def setUp(self):
        self.params = Aircraft6DOFParams()

    def test_quaternion_round_trip(self):
        q = euler_to_quat(0.2, -0.1, 0.4)
        roll, pitch, yaw = quat_to_euler(q)
        self.assertAlmostEqual(roll, 0.2)
        self.assertAlmostEqual(pitch, -0.1)
        self.assertAlmostEqual(yaw, 0.4)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0)

    def test_level_identity_rotation_maps_gravity_down(self):
        r = quat_to_matrix(euler_to_quat(0, 0, 0))
        gravity_body = r.T @ np.array([0, 0, self.params.mass * 9.80665])
        self.assertAlmostEqual(gravity_body[2], self.params.mass * 9.80665)

    def test_trim_has_near_zero_vertical_and_pitch_acceleration(self):
        state, surfaces = trim_level_flight(self.params)
        out = rigid_body_derivatives(state, surfaces, self.params)
        self.assertLess(abs(out.velocity_dot[2]), 0.25)
        self.assertLess(abs(out.rates_dot[1]), 0.002)
        self.assertGreater(out.aero.lift, 0.9 * self.params.mass * 9.80665)

    def test_positive_aileron_rolls_right(self):
        state, trim = trim_level_flight(self.params)
        command = SurfaceCommand(trim.throttle, trim.elevator, math.radians(5), 0)
        out = rigid_body_derivatives(state, command, self.params)
        self.assertGreater(out.rates_dot[0], 0)

    def test_positive_elevator_pitches_nose_up(self):
        state, trim = trim_level_flight(self.params)
        command = SurfaceCommand(trim.throttle, trim.elevator + math.radians(2), 0, 0)
        out = rigid_body_derivatives(state, command, self.params)
        self.assertLess(out.rates_dot[1], 0)

    def test_positive_rudder_turns_nose_left_in_sign_convention(self):
        state, trim = trim_level_flight(self.params)
        command = SurfaceCommand(trim.throttle, trim.elevator, 0, math.radians(3))
        out = rigid_body_derivatives(state, command, self.params)
        self.assertLess(out.rates_dot[2], 0)

    def test_rk4_preserves_quaternion_norm(self):
        state, trim = trim_level_flight(self.params)
        new_state = rk4_step_6dof(state, trim, self.params, 0.02)
        self.assertAlmostEqual(np.linalg.norm(new_state.quaternion), 1.0, places=12)

    def test_actuator_rate_and_position_limits(self):
        params = self.params
        actual = ActuatorState(0.0, 0.0, 0.0, 0.0)
        desired = SurfaceCommand(1.0, math.radians(90), math.radians(-90), math.radians(90))
        stepped = step_actuators(actual, desired, params, 0.01)
        self.assertAlmostEqual(stepped.throttle, params.throttle_rate * 0.01)
        self.assertAlmostEqual(stepped.elevator, params.surface_rate * 0.01)
        for _ in range(1000):
            stepped = step_actuators(stepped, desired, params, 0.05)
        self.assertLessEqual(stepped.elevator, params.max_surface)
        self.assertGreaterEqual(stepped.aileron, -params.max_surface)
        self.assertLessEqual(stepped.rudder, params.max_rudder)

    def test_invalid_aircraft_params_rejected(self):
        bad = Aircraft6DOFParams(mass=-1)
        with self.assertRaises(ValueError):
            aerodynamics(trim_level_flight(self.params)[0], trim_level_flight(self.params)[1], bad)


if __name__ == "__main__":
    unittest.main()
