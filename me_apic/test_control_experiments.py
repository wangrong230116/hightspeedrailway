import unittest

import numpy as np

from run_baseline import Parameters
from run_control_experiments import (
    ControlParameters,
    feedback_force,
    metrics,
    simulate_controller,
    tapic_schedule,
)


class ControlExperimentTests(unittest.TestCase):
    def test_feedback_direction_and_saturation(self):
        c = ControlParameters()
        self.assertLess(feedback_force(80.0, c), 0.0)
        self.assertGreater(feedback_force(20.0, c), 0.0)
        self.assertLessEqual(abs(feedback_force(1e6, c)), c.actuator_limit_N)

    def test_tapic_schedule_constraints(self):
        c = ControlParameters()
        gaps = np.diff(tapic_schedule(1.0, c))
        self.assertTrue(np.all(gaps >= c.tapic_interval_min_s))
        self.assertTrue(np.all(gaps <= c.tapic_interval_min_s+c.tapic_interval_jitter_s))

    def test_passive_controller_has_no_actuation(self):
        result = simulate_controller("Passive", Parameters(), ControlParameters())
        self.assertFalse(result.active.any())
        self.assertTrue(np.allclose(result.actuator_force, 0.0))

    def test_eapic_improves_force_with_low_duty_cycle(self):
        p, c = Parameters(), ControlParameters()
        passive = metrics(simulate_controller("Passive", p, c), c)
        eapic = metrics(simulate_controller("E-APIC", p, c), c)
        self.assertLess(eapic["std_force_N"], passive["std_force_N"])
        self.assertLess(eapic["ATR_percent"], 30.0)


if __name__ == "__main__":
    unittest.main()

