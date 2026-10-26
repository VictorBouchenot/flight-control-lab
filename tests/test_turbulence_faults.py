import unittest

import numpy as np

from flight_control_lab.aircraft6dof import SurfaceCommand
from flight_control_lab.faults import ActuatorFault
from flight_control_lab.scenarios6dof import simulate_carrier_challenge_6dof
from flight_control_lab.turbulence import StochasticTurbulence


class TurbulenceAndFaultTests(unittest.TestCase):
    def test_turbulence_is_seeded_and_deterministic(self):
        gusts = StochasticTurbulence(seed=42, intensity_mps=1.2)
        position = np.array([120.0, -8.0, -100.0])
        first = gusts.wind_nav(position, 3.2)
        second = gusts.wind_nav(position, 3.2)
        np.testing.assert_allclose(first, second)
        self.assertGreater(np.linalg.norm(first), 0.1)

    def test_turbulence_seed_changes_gusts(self):
        position = np.array([120.0, -8.0, -100.0])
        one = StochasticTurbulence(seed=1).wind_nav(position, 4.0)
        two = StochasticTurbulence(seed=2).wind_nav(position, 4.0)
        self.assertGreater(np.linalg.norm(one - two), 0.1)

    def test_disabled_turbulence_is_zero(self):
        gusts = StochasticTurbulence(enabled=False)
        np.testing.assert_allclose(gusts.wind_nav(np.zeros(3), 1.0), np.zeros(3))

    def test_actuator_fault_scales_only_selected_surface(self):
        fault = ActuatorFault("aileron", start_time=2.0, effectiveness=0.5)
        command = SurfaceCommand(0.8, 0.2, 0.3, -0.1)
        before = fault.apply(command, 1.0)
        after = fault.apply(command, 3.0)
        self.assertEqual(before, command)
        self.assertEqual(after.throttle, command.throttle)
        self.assertEqual(after.elevator, command.elevator)
        self.assertAlmostEqual(after.aileron, 0.15)
        self.assertEqual(after.rudder, command.rudder)

    def test_invalid_fault_rejected(self):
        with self.assertRaises(ValueError):
            ActuatorFault("flaps").validate()
        with self.assertRaises(ValueError):
            ActuatorFault(effectiveness=0.0).validate()

    def test_harder_recovery_and_fault_scenarios(self):
        for challenge in ("turbulent_recovery", "aileron_fault"):
            with self.subTest(challenge=challenge):
                result, touchdown = simulate_carrier_challenge_6dof(challenge, dt=0.05)
                self.assertTrue(touchdown.success, touchdown.reason)
                self.assertTrue(result.metrics["arrestment_success"], result.metrics["arrestment_reason"])
                self.assertIn("landing_quality_score", result.metrics)
                self.assertGreater(result.metrics["landing_quality_score"], 40.0)


if __name__ == "__main__":
    unittest.main()
