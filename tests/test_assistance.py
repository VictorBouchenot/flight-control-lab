import math
import unittest

from flight_control_lab.assistance import (
    envelope_metrics,
    simulate_carrier_assist_comparison,
    simulate_maneuver_assist_comparison,
)
from flight_control_lab.aircraft6dof import SurfaceCommand
from flight_control_lab.carrier6dof import MovingCarrier
from flight_control_lab.pilot import PilotModel
from flight_control_lab.scenarios6dof import simulate_moving_carrier_6dof


class PilotModelTests(unittest.TestCase):
    def test_delay_holds_initial_command(self):
        pilot = PilotModel(delay_s=0.2)
        carrier = MovingCarrier()
        # First command at t=0 must be the pre-delay pending value, not the chase output.
        state = self._hover_state()
        first = pilot.carrier_command(state, carrier, 0.0, 74.0)
        self.assertAlmostEqual(first.elevator, 0.0)

    def test_deterministic(self):
        a = PilotModel()
        b = PilotModel()
        carrier = MovingCarrier()
        state = self._hover_state()
        for k in range(50):
            t = k * 0.05
            ca = a.carrier_command(state, carrier, t, 74.0)
            cb = b.carrier_command(state, carrier, t, 74.0)
            self.assertAlmostEqual(ca.elevator, cb.elevator)
            self.assertAlmostEqual(ca.aileron, cb.aileron)

    def test_warm_start_uses_trim(self):
        pilot = PilotModel()
        trim = SurfaceCommand(0.5, -0.15, 0.0, 0.0)
        pilot.warm_start(trim)
        carrier = MovingCarrier()
        first = pilot.carrier_command(self._hover_state(), carrier, 0.0, 74.0)
        self.assertAlmostEqual(first.elevator, -0.15)

    def test_stick_history_records_every_command(self):
        pilot = PilotModel()
        carrier = MovingCarrier()
        state = self._hover_state()
        for k in range(10):
            pilot.carrier_command(state, carrier, k * 0.05, 74.0)
        self.assertEqual(len(pilot.stick_history()["elevator"]), 10)

    def _hover_state(self):
        from flight_control_lab.aircraft6dof import RigidBodyState
        import numpy as np

        return RigidBodyState(np.array([0.0, 0.0, -300.0]), np.array([74.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3))


class AssistanceComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = simulate_carrier_assist_comparison(dt=0.1)
        cls.maneuver = simulate_maneuver_assist_comparison(dt=0.1)

    def test_both_modes_run(self):
        for mode in ("assisted", "raw"):
            metrics = self.comparison[mode]
            self.assertIn("max_alpha_deg", metrics)
            self.assertIn("max_bank_deg", metrics)
            self.assertIn("lost_control", metrics)
            self.assertIn("landing_quality_score", metrics)

    def test_assisted_holds_near_envelope_limits(self):
        # The limiter acts through rate-limited actuators, so it is not a hard
        # bound: assert the assisted case stays near the limits (small
        # tolerance), far inside the raw pilot's excursions.
        a = self.comparison["assisted"]
        self.assertFalse(a["lost_control"])
        self.assertLessEqual(a["max_alpha_deg"], 13.0 + 2.5)
        self.assertLessEqual(a["max_bank_deg"], 15.0)

    def test_raw_pilot_struggles(self):
        # The point of the demo: the same pilot without the FCS departs.
        r = self.comparison["raw"]
        self.assertTrue(r["lost_control"])
        self.assertGreater(r["max_alpha_deg"], 25.0)

    def test_assisted_scores_higher(self):
        self.assertGreater(
            self.comparison["assisted"]["landing_quality_score"],
            self.comparison["raw"]["landing_quality_score"],
        )

    def test_pilot_stick_recorded(self):
        stick = self.comparison["pilot_stick"]
        self.assertGreater(len(stick["elevator"]), 10)

    def test_shadow_stick_matches_assisted_run_length(self):
        # Maneuver shadow pilot: one stick sample per step of the assisted run.
        stick = self.maneuver["pilot_stick"]
        n_steps = len(self.maneuver["assisted_result"].times)
        self.assertGreaterEqual(len(stick["aileron"]), n_steps - 2)

    def test_envelope_metrics_keys(self):
        m = envelope_metrics(self.comparison["assisted_result"])
        for key in ("max_alpha_deg", "max_abs_beta_deg", "max_bank_deg", "flight_time_s", "lost_control"):
            self.assertIn(key, m)






class ManeuverComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison = simulate_maneuver_assist_comparison(dt=0.1)

    def test_both_modes_run(self):
        for mode in ("assisted", "raw"):
            metrics = self.comparison[mode]
            for key in ("clean_capture", "settle_time_s", "heading_overshoot_deg", "max_abs_altitude_error_m", "max_bank_deg", "max_alpha_deg", "lost_control"):
                self.assertIn(key, metrics)

    def test_assisted_captures_cleanly(self):
        a = self.comparison["assisted"]
        self.assertTrue(a["clean_capture"])
        self.assertFalse(a["lost_control"])
        self.assertLessEqual(a["max_bank_deg"], 50.0)

    def test_raw_pilot_departs(self):
        r = self.comparison["raw"]
        self.assertTrue(r["lost_control"])
        self.assertFalse(r["clean_capture"])
        self.assertGreater(r["max_bank_deg"], 60.0)




if __name__ == "__main__":
    unittest.main()
