import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "run_scenario.py", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_takeoff_json_uses_6dof_by_default(self):
        completed = self.run_cli("takeoff", "--json")
        metrics = json.loads(completed.stdout)
        self.assertTrue(metrics["rotated"])
        self.assertTrue(metrics["target_altitude_reached"])
        self.assertIn("final_speed_mps", metrics)

    def test_moving_carrier_json(self):
        completed = self.run_cli("carrier", "--json")
        metrics = json.loads(completed.stdout)
        self.assertTrue(metrics["touchdown_success"])
        self.assertLessEqual(metrics["relative_sink_rate_mps"], 4.5)
        self.assertIn("deck_roll_deg", metrics)

    def test_turbulent_carrier_challenge_json(self):
        completed = self.run_cli("carrier-turbulent", "--json")
        metrics = json.loads(completed.stdout)
        self.assertTrue(metrics["touchdown_success"])
        self.assertTrue(metrics["arrestment_success"])
        self.assertEqual(metrics["challenge"], "turbulent_recovery")
        self.assertTrue(metrics["turbulence_enabled"])
        self.assertIn("landing_quality_grade", metrics)

    def test_rough_carrier_challenge_json(self):
        completed = self.run_cli("carrier-rough", "--json")
        metrics = json.loads(completed.stdout)
        self.assertTrue(metrics["touchdown_success"])
        self.assertTrue(metrics["arrestment_success"])
        self.assertEqual(metrics["challenge"], "rough_seas")

    def test_carrier_calm_mode_disables_airwake(self):
        completed = self.run_cli("carrier", "--calm", "--json")
        metrics = json.loads(completed.stdout)
        self.assertTrue(metrics["touchdown_success"])
        self.assertFalse(metrics["airwake_enabled"])
        self.assertEqual(metrics["max_airwake_downwash_mps"], 0.0)

    def test_pointmass_baseline_still_available(self):
        completed = self.run_cli("carrier-pointmass", "--json")
        metrics = json.loads(completed.stdout)
        self.assertTrue(metrics["touchdown_success"])

    def test_bad_scenario_rejected(self):
        completed = subprocess.run(
            [sys.executable, "run_scenario.py", "unknown"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
