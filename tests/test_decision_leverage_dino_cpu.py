"""Oracle-feature experiment contracts; CPU arithmetic without models or envs."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dino_cpu", ROOT / "scripts/run_decision_leverage_dino_cpu.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class OracleFeatureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        torch.set_num_threads(1)

    def arrays(self):
        np = self.np
        # Extremely different initial context must never enter future-error MSE.
        truth = np.ones((6, 5, 2), dtype=np.float32)
        forecast = np.concatenate([np.full((1, 5, 2), 1000, np.float32), truth + 2])
        goal = np.full((1, 1, 2), 9, np.float32)
        return {prefix + k: value.copy() for k in runner.FEATURES
                for prefix, value in (("truth_", truth), ("forecast_", forecast), ("goal_", goal))}

    def test_context_excluded_and_goal_distance_is_not_prediction_error(self):
        scores, errors = runner.tensor_analysis(self.arrays(), self.torch, self.np)
        self.assertEqual(errors["0.0"]["weighted"], [4.4] * 5)
        self.assertEqual(errors["1.0"]["weighted"], [0.] * 5)
        self.assertTrue(all(x > 0 for x in scores["1.0"]))

    def test_oracle_endpoints_and_interpolation_have_expected_error(self):
        _, errors = runner.tensor_analysis(self.arrays(), self.torch, self.np)
        for lam in runner.LAMBDAS:
            self.assertAlmostEqual(errors[str(lam)]["mean_weighted"], (1 - lam) ** 2 * 4.4)

    def test_mis_positive_does_not_imply_a_negative_paired_oracle_effect(self):
        utility = [{h: {"terminal_reward": value, "first_success_control": None} for h in ("h15", "h30")}
                   for value in [0, 4, 6, 2, 1]]
        # Both predictors choose1 although true-utility best is2.
        scores = {str(lam): [4, 0, 3, 6, 7] for lam in runner.LAMBDAS}
        error = {str(lam): {} for lam in runner.LAMBDAS}
        result = runner.summarize_conditions(scores, error, utility)
        metrics = result["0.0"]["metrics"]["h30"]["terminal_reward"]
        self.assertEqual(metrics["Mis"], 2)
        self.assertEqual(metrics["Pred"], 0)
        self.assertEqual(metrics["regret"], 2)


if __name__ == "__main__":
    unittest.main()
