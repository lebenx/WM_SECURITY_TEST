"""Scientific contracts for the finite recorded-candidate utility panel."""
import copy
import importlib.util
from itertools import product
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("decision_leverage_panel", ROOT / "scripts/compute_decision_leverage_panel.py")
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)


class FiniteSupportContract(unittest.TestCase):
    def test_headroom_is_baseline_specific_not_total_range(self):
        r = panel.finite_candidate_bounds([0, 10], baseline_index=0)
        self.assertEqual(r["leverage"], 10)
        self.assertEqual(r["baseline_harm_headroom"], 0)
        c = panel.finite_candidate_bounds([3, 8], baseline_index=0, higher_is_better=False)
        self.assertEqual(c["baseline_harm_headroom"], 5)

    def test_every_identity_selector_obeys_bound_including_random_mixtures(self):
        banks = [[-3, 2], [7, 8, 11], [-10, -4]]
        low = sum(min(x) for x in banks)
        high = sum(max(x) for x in banks)
        values = [sum(bank[index] for bank, index in zip(banks, indices))
                  for indices in product(*[range(len(bank)) for bank in banks])]
        self.assertEqual(min(values), low)
        self.assertEqual(max(values), high)
        mixture = sum((i + 1) * value for i, value in enumerate(values)) / sum(range(1, len(values) + 1))
        self.assertGreaterEqual(mixture, low)
        self.assertLessEqual(mixture, high)

    def test_elite_mean_or_missing_support_is_rejected(self):
        for candidate in (0.5, -1, 2, "mean", True):
            with self.assertRaisesRegex(ValueError, "not a member"):
                panel.finite_candidate_bounds([1, 3], candidate)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            panel.finite_candidate_bounds([float("nan"), 1], 0)

    def test_incomplete_or_repeated_candidate_bank_is_rejected(self):
        row = panel.entry("example", "root", "cluster", 0, "return", 0, True, "trace", 0)
        for candidate in (0, 2):
            other = {**row, "candidate": candidate, "value": 1}
            with self.assertRaisesRegex(ValueError, "Incomplete bank"):
                panel.summarize([row, other])


class RealCacheChecks(unittest.TestCase):
    def test_cosmos_reopens_real_trace_and_reproduces_known_finite_bounds(self):
        rows, meta = panel.cosmos(panel.Inputs(ROOT))
        roots, aggregate, clusters = panel.summarize(rows)
        success = next(r for r in aggregate if r["metric"] == "success")
        cost = next(r for r in aggregate if r["metric"] == "capped_steps")
        self.assertEqual((meta["roots"], meta["clusters"]), (30, 10))
        self.assertEqual((success["minimum_total"], success["maximum_total"], success["baseline_value_total"]), (28, 30, 29))
        self.assertEqual(success["baseline_harm_headroom_total"], 1)
        self.assertEqual(success["exact_tie_roots"], 28)
        self.assertEqual((cost["minimum_total"], cost["maximum_total"], cost["baseline_value_total"]), (3230, 3646, 3477))
        self.assertEqual(cost["baseline_harm_headroom_total"], 169)
        self.assertEqual(len(clusters), 20)

    def test_dino_trace_mutation_cannot_silently_become_a_new_result(self):
        inputs = panel.Inputs(ROOT)
        original = inputs.lines

        def damaged(path, expected_hash=None):
            rows = copy.deepcopy(original(path, expected_hash))
            rows[0]["reward"] += 1
            return rows

        with patch.object(inputs, "lines", side_effect=damaged):
            with self.assertRaisesRegex(ValueError, "DINO utility mismatch"):
                panel.dino(inputs)

    def test_worldagen_repeats_are_not_counted_as_independent_roots(self):
        rows, meta = panel.worldagen(panel.Inputs(ROOT))
        roots, aggregates, clusters = panel.summarize(rows)
        self.assertEqual(meta["roots"], 3)
        self.assertEqual(meta["clusters"], 1)
        for row in aggregates:
            self.assertEqual((row["roots"], row["clusters"], row["leverage_total"]), (3, 1, 0))


if __name__ == "__main__":
    unittest.main()
