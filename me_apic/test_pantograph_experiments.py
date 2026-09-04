import unittest

import numpy as np

from run_baseline import Parameters
from run_pantograph_experiments import (
    ControllerParameters,
    PantographParameters,
    RobustnessParameters,
    adaptive_burst_width,
    chain_matrix,
    design_modal_lqr,
    metrics,
    moving_contact_vector,
    simulate,
    target_force,
)


class PantographExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.beam = Parameters(elements=20)
        cls.panto = PantographParameters()
        cls.control = ControllerParameters()

    def test_dsa380_target_force(self):
        self.assertAlmostEqual(target_force(300.0), 157.3, places=9)

    def test_adaptive_burst_is_shorter_at_higher_speed(self):
        low = adaptive_burst_width(200.0, self.control)
        high = adaptive_burst_width(350.0, self.control)
        self.assertGreater(low, high)
        self.assertGreaterEqual(high, self.control.adaptive_burst_min_s)
        self.assertLessEqual(low, self.control.adaptive_burst_max_s)

    def test_chain_matrices_are_symmetric(self):
        for values in (
            self.panto.stiffness_N_m,
            self.panto.damping_Ns_m,
        ):
            matrix = chain_matrix(values)
            np.testing.assert_allclose(matrix, matrix.T)
            self.assertGreaterEqual(float(np.linalg.eigvalsh(matrix).min()), -1e-10)

    def test_unilateral_contact_never_returns_tension(self):
        run = simulate("Passive", 300.0, self.beam, self.panto, self.control)
        self.assertGreaterEqual(float(run.force.min()), 0.0)

    def test_noise_delay_and_actuator_dynamics_remain_bounded(self):
        robust = RobustnessParameters(
            measurement_noise_std_N=2.0,
            actuator_delay_s=0.005,
            actuator_time_constant_s=0.015,
            irregularity_scale=1.1,
            irregularity_phase_offsets_rad=(0.2, -0.3, 0.4),
        )
        run = simulate(
            "ME-APIC", 300.0, self.beam, self.panto, self.control,
            feedback_kind="modal_lqr", robustness=robust, random_seed=17,
            controller_model_pantograph=self.panto,
        )
        self.assertTrue(np.all(np.isfinite(run.force)))
        self.assertLessEqual(float(np.max(np.abs(run.actuator))), self.control.limit_N)

    def test_eapic_is_intermittent(self):
        run = simulate("E-APIC", 300.0, self.beam, self.panto, self.control)
        row = metrics(run)
        self.assertGreater(row["ATR_percent"], 0.0)
        self.assertLess(row["ATR_percent"], 100.0)

    def test_modal_lqr_is_finite_and_dimensionally_consistent(self):
        from run_baseline import assemble_beam

        stiffness, mass, free = assemble_beam(self.beam)
        n_beam = len(free)
        size = n_beam+3
        full_stiffness = np.zeros((size, size))
        full_damping = np.zeros_like(full_stiffness)
        full_mass = np.zeros_like(full_stiffness)
        full_stiffness[:n_beam, :n_beam] = stiffness
        full_mass[:n_beam, :n_beam] = mass
        full_stiffness[n_beam:, n_beam:] = chain_matrix(self.panto.stiffness_N_m)
        full_damping[n_beam:, n_beam:] = chain_matrix(self.panto.damping_Ns_m)
        full_mass[n_beam:, n_beam:] = np.diag(self.panto.masses_kg)
        contact = moving_contact_vector(0.5*self.beam.span_length, self.beam, free)
        shapes, gain, riccati = design_modal_lqr(
            full_mass, full_damping, full_stiffness, contact,
            self.panto.contact_stiffness_N_m,
        )
        self.assertEqual(shapes.shape, (size, 16))
        self.assertEqual(gain.shape, (32,))
        self.assertTrue(np.all(np.isfinite(gain)))
        self.assertEqual(riccati.shape, (32, 32))
        self.assertGreater(float(np.linalg.eigvalsh(riccati).min()), 0.0)


if __name__ == "__main__":
    unittest.main()

