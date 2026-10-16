"""Scientific failure cases for paired-root DINO oracle analysis; no model calls."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_decision_leverage_dino_cpu import (
    EXPECTED_SEEDS, LAMBDAS, analyze_roots, bootstrap_mean,
    validate_complete_panel, wilson,
)
from run_decision_leverage_dino_cpu import summarize_conditions


def artificial_roots(n, same_selection=False):
    """Synthetic fixtures only: never saved as scientific evidence."""
    utilities = []
    for reward in [2., 1., 3., 0., -.5]:
        unit = dict(terminal_reward=reward, cumulative_reward=reward * 10,
                    negative_terminal_distance=reward / 100,
                    terminal_success=int(reward >= 3), ever_success=int(reward >= 3),
                    first_success_control=15 if reward >= 3 else None)
        utilities.append({"h15": unit.copy(), "h30": unit.copy()})
    scores, errors = {}, {}
    for lam in LAMBDAS:
        scores[str(lam)] = [1., 0., 2., 3., 4.] if lam == 1. or same_selection else [0., 1., 2., 3., 4.]
        errors[str(lam)] = {"mean_weighted": (1. - lam) ** 2}
    conditions = summarize_conditions(scores, errors, utilities)
    return [{"seed": 1001 + i, "conditions": conditions, "candidate_utilities": utilities} for i in range(n)]


class StatisticsTests(unittest.TestCase):
    def test_no_samples_or_nonfinite_are_rejected(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                bootstrap_mean(values)

    def test_pairing_precedes_resampling(self):
        original = np.arange(30, dtype=float) * 1000
        oracle = original - 1
        stat = bootstrap_mean(oracle - original)
        self.assertEqual(stat["percentile_95_two_sided"], [-1., -1.])
        self.assertEqual(stat["percentile_95_one_sided_upper"], -1.)
        self.assertEqual(stat, bootstrap_mean(oracle - original))

    def test_calibration_cannot_be_confirmation(self):
        with self.assertRaises(ValueError):
            validate_complete_panel("calibration", EXPECTED_SEEDS, EXPECTED_SEEDS, "completed")

    def test_missing_duplicate_unregistered_and_running_rejected(self):
        cases = [(EXPECTED_SEEDS[:-1], "completed"),
                 (EXPECTED_SEEDS[:-1] + [1001], "completed"),
                 (EXPECTED_SEEDS[:-1] + [1000], "completed"),
                 (EXPECTED_SEEDS, "running")]
        for observed, status in cases:
            with self.assertRaises(ValueError):
                validate_complete_panel("confirmation", EXPECTED_SEEDS, observed, status)
        validate_complete_panel("confirmation", EXPECTED_SEEDS, EXPECTED_SEEDS, "completed")

    def test_wilson_extremes_and_invalid_denominator(self):
        none, all_events = wilson(0, 30), wilson(30, 30)
        self.assertAlmostEqual(none["wilson_95_two_sided"][0], 0.)
        self.assertGreater(none["wilson_95_two_sided"][1], 0.)
        self.assertAlmostEqual(all_events["wilson_95_two_sided"][1], 1.)
        self.assertLess(all_events["wilson_95_two_sided"][0], 1.)
        with self.assertRaises(ValueError):
            wilson(0, 0)

    def test_primary_requires_complete30_and_actual_negative_effect(self):
        valid, root_rows, curves, candidates = analyze_roots(artificial_roots(30), complete=True)
        self.assertTrue(valid["primary"]["p1_offline_fixed_bank_supported"])
        self.assertEqual(valid["primary"]["delta_selected_utility"]["mean"], -1.)
        self.assertEqual(valid["primary"]["selected_action_flip_rate"]["count"], 30)
        self.assertAlmostEqual(valid["primary"]["pairwise_order_flip_fraction"]["mean"], .1)
        self.assertEqual(len(root_rows), 30 * 5 * 2 * 5)
        self.assertEqual(len(candidates), 30 * 5 * 2)
        self.assertEqual(len(curves), 5 * 2 * 5)
        incomplete = analyze_roots(artificial_roots(29), complete=True)[0]
        self.assertIsNone(incomplete["primary"]["p1_offline_fixed_bank_supported"])
        receipt_incomplete = analyze_roots(artificial_roots(30), complete=False)[0]
        self.assertIsNone(receipt_incomplete["primary"]["p1_offline_fixed_bank_supported"])

    def test_mis_positive_cannot_alone_establish_p1(self):
        numbers = analyze_roots(artificial_roots(30, same_selection=True), complete=True)[0]
        self.assertGreater(numbers["lambda_curves"][0]["metrics"]["h30"]["terminal_reward"]["Mis"]["mean"], 0)
        self.assertEqual(numbers["primary"]["delta_selected_utility"]["mean"], 0.)
        self.assertFalse(numbers["primary"]["p1_offline_fixed_bank_supported"])
        self.assertEqual(numbers["primary"]["oracle_reversal_event"]["count"], 0)
        self.assertEqual(numbers["primary"]["selected_action_flip_rate"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
