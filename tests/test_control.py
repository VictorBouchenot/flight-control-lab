import math
import unittest

from flight_control_lab.control import FlightControlSystem
from flight_control_lab.dynamics import AircraftParams, PointMassState


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.params = AircraftParams()
        self.fcs = FlightControlSystem(self.params)
        self.state = PointMassState(0, 0, 1500, 140, 0, 0)

    def test_speed_error_increases_throttle(self):
        slow = self.fcs.command(self.state, target_speed=170)
        fast = self.fcs.command(self.state, target_speed=120)
        self.assertGreater(slow.throttle, fast.throttle)

    def test_climb_error_increases_load_factor(self):
        level = self.fcs.command(self.state, target_speed=140, target_gamma=0.0)
        climb = self.fcs.command(self.state, target_speed=140, target_gamma=math.radians(8))
        self.assertGreater(climb.load_factor, level.load_factor)

    def test_heading_error_uses_coordinated_bank(self):
        right = self.fcs.command(self.state, target_speed=140, target_heading=math.radians(30))
        left = self.fcs.command(self.state, target_speed=140, target_heading=math.radians(-30))
        self.assertGreater(right.bank, 0)
        self.assertLess(left.bank, 0)

    def test_envelope_limits_are_applied(self):
        low_speed = PointMassState(0, 0, 1500, self.params.min_speed + 1, 0, 0)
        command = self.fcs.command(
            low_speed,
            target_speed=self.params.max_speed,
            target_gamma=math.radians(20),
            target_heading=math.radians(180),
        )
        self.assertLessEqual(command.load_factor, self.params.max_load_factor)
        self.assertLessEqual(abs(command.bank), self.params.max_bank)
        self.assertGreaterEqual(command.load_factor, self.params.min_load_factor)

    def test_carrier_correction_moves_toward_centerline(self):
        right_of_line = PointMassState(100, 20, 250, 75, -0.05, 0)
        command = self.fcs.carrier_approach_command(right_of_line, 240, 20, 74)
        self.assertLess(command.bank, 0)


if __name__ == "__main__":
    unittest.main()
