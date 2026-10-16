#!/usr/bin/env python3
"""Exact, offline finite-bank coupling audit; no model or environment imports."""

import argparse
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path


def exact(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator,
            "fraction": str(value), "value": float(value)}


def maximum_matching_permutation(edges):
    """Maximize binary edge count, then complete unmatched rows/columns in order."""
    n = len(edges)
    assert n > 0 and all(len(row) == n for row in edges)
    column_to_row = [-1] * n

    def augment(row, seen):
        for column in range(n):
            if not edges[row][column] or column in seen:
                continue
            seen.add(column)
            if column_to_row[column] < 0 or augment(column_to_row[column], seen):
                column_to_row[column] = row
                return True
        return False

    size = sum(augment(row, set()) for row in range(n))
    permutation = [-1] * n
    for column, row in enumerate(column_to_row):
        if row >= 0:
            permutation[row] = column
    unmatched_rows = [row for row, column in enumerate(permutation) if column < 0]
    unmatched_columns = [column for column, row in enumerate(column_to_row) if row < 0]
    completion = list(zip(unmatched_rows, unmatched_columns))
    for row, column in completion:
        permutation[row] = column
    assert sorted(permutation) == list(range(n))
    # A maximum matching has no favorable edge between two unmatched vertices.
    assert all(not edges[row][column] for row, column in completion)
    assert sum(bool(edges[row][column]) for row, column in enumerate(permutation)) == size
    return permutation, {"maximum_matching_size": size,
                         "unmatched_completion": [list(pair) for pair in completion]}


def ordered_permutation(scores, reverse_second=False):
    first = sorted(range(len(scores[0])), key=lambda i: (scores[0][i], i))
    second = sorted(range(len(scores[1])),
                    key=lambda j: (-scores[1][j] if reverse_second else scores[1][j], j))
    permutation = [-1] * len(first)
    for row, column in zip(first, second):
        permutation[row] = column
    return permutation


def measure(scores, pairs, success, costs, permutation=None):
    pairs = list(pairs)
    denominator = len(pairs)
    strict_bad = sum(scores[0][i] > scores[1][j] for i, j in pairs)
    ties = sum(scores[0][i] == scores[1][j] for i, j in pairs)
    bad = strict_bad + ties
    p_bad = Fraction(bad, denominator)
    result = {
        "equiprobable_pair_count": denominator,
        "strict_candidate0_win_count": strict_bad, "tie_count": ties,
        "candidate0_selected_count": bad, "candidate1_selected_count": denominator - bad,
        "bad_candidate_probability": exact(p_bad),
        "expected_historical_success": exact(p_bad * success[0] + (1 - p_bad) * success[1]),
        "expected_historical_capped_steps": exact(p_bad * costs[0] + (1 - p_bad) * costs[1]),
    }
    if permutation is not None:
        n = len(scores[0])
        assert len(permutation) == n and sorted(permutation) == list(range(n))
        row_counts = [sum(i == row for i, _ in pairs) for row in range(n)]
        column_counts = [sum(j == column for _, j in pairs) for column in range(n)]
        assert row_counts == column_counts == [1] * n
        result.update(permutation=permutation,
                      permutation_convention="candidate1 index = permutation[candidate0 index]",
                      row_counts=row_counts, column_counts=column_counts,
                      each_candidate_index_marginal=exact(Fraction(1, n)),
                      uniform_marginals_exact=True)
    return result


def parse_input(summary):
    scores = summary["scores_by_candidate"]
    if not (isinstance(scores, list) and len(scores) == 2
            and all(isinstance(row, list) for row in scores)
            and len(scores[0]) > 0 and len(scores[0]) == len(scores[1])):
        raise ValueError("scores_by_candidate must contain two nonempty equal-length lists")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for row in scores for v in row):
        raise ValueError("Every score must be a finite JSON number, not a boolean")
    utilities = summary["historical_continuation_utilities"]
    if isinstance(utilities, list):
        if (len(utilities) != 2 or any(not isinstance(row, dict) for row in utilities)
                or sorted(row.get("candidate_index", -1) for row in utilities) != [0, 1]
                or any(type(row["candidate_index"]) is not int for row in utilities)):
            raise ValueError("Historical utility rows must identify candidates 0 and 1 exactly once")
        utilities = sorted(utilities, key=lambda row: row["candidate_index"])
        success = [row["success"] for row in utilities]
        costs = [row["capped_steps"] for row in utilities]
    elif isinstance(utilities, dict):
        success, costs = utilities["success"], utilities["capped_steps"]
    else:
        raise ValueError("historical_continuation_utilities must be a dict or two candidate rows")
    if (not isinstance(success, list) or len(success) != 2
            or any(type(v) is not bool for v in success) or success != [False, True]):
        raise ValueError("This frozen gate requires historical success [false, true]")
    if (not isinstance(costs, list) or len(costs) != 2
            or any(type(v) is not int for v in costs) or costs != [300, 170]):
        raise ValueError("This frozen gate requires historical capped_steps [300, 170]")
    return scores, success, costs


def analyze(summary):
    scores, success, costs = parse_input(summary)
    n = len(scores[0])
    wins = [[a >= b for b in scores[1]] for a in scores[0]]
    maximum, maximum_certificate = maximum_matching_permutation(wins)
    minimum, minimum_certificate = maximum_matching_permutation(
        [[not entry for entry in row] for row in wins])
    product = measure(scores, itertools.product(range(n), repeat=2), success, costs)
    order0 = sorted(range(n), key=lambda i: (scores[0][i], i))
    order1 = sorted(range(n), key=lambda j: (scores[1][j], j))
    ordered_product = measure(scores, itertools.product(order0, order1), success, costs)
    assert product == ordered_product
    controls = {"independent_product": product, "independent_score_ordered": ordered_product}
    for name, permutation in {
        "shared_index": list(range(n)),
        "comonotone_scores": ordered_permutation(scores),
        "countermonotone_scores": ordered_permutation(scores, reverse_second=True),
        "minimum_bad_uniform_coupling": minimum,
        "maximum_bad_uniform_coupling": maximum,
    }.items():
        controls[name] = measure(scores, enumerate(permutation), success, costs, permutation)
    controls["minimum_bad_uniform_coupling"]["matching_certificate"] = minimum_certificate
    controls["maximum_bad_uniform_coupling"]["matching_certificate"] = maximum_certificate
    independent_bad = Fraction(product["candidate0_selected_count"], n * n)
    maximum_bad = Fraction(controls["maximum_bad_uniform_coupling"]["candidate0_selected_count"], n)
    minimum_bad = Fraction(controls["minimum_bad_uniform_coupling"]["candidate0_selected_count"], n)
    assert minimum_bad <= independent_bad <= maximum_bad
    assert maximum_bad == Fraction(maximum_certificate["maximum_matching_size"], n)
    assert minimum_bad == 1 - Fraction(minimum_certificate["maximum_matching_size"], n)
    independent_success = 1 - independent_bad
    increment = maximum_bad - independent_bad
    baseline_pass = independent_success >= Fraction(3, 4)
    effect_pass = increment >= Fraction(1, 4)
    score_ranges = [[min(row), max(row)] for row in scores]
    good_support_margin = min(scores[1]) - max(scores[0])
    all_pairs_good = good_support_margin > 0
    all_pairs_bad = min(scores[0]) >= max(scores[1])
    if all_pairs_good:
        assert maximum_bad == minimum_bad == independent_bad == 0
    if all_pairs_bad:
        assert maximum_bad == minimum_bad == independent_bad == 1
    return {
        "status": "completed_exact_finite_bank_coupling_analysis", "bank_size_per_candidate": n,
        "scores_by_candidate": scores,
        "historical_continuation_utilities": {"success": success, "capped_steps": costs},
        "selection_rule": "Select candidate0 iff score0 >= score1; exact ties select candidate0.",
        "bad_candidate_index": 0, "W_bad_candidate_win": [[int(v) for v in row] for row in wins],
        "scalar_support_check": {
            "candidate_min_max": score_ranges,
            "candidate1_min_minus_candidate0_max": good_support_margin,
            "all_pairs_strictly_select_candidate1": all_pairs_good,
            "all_pairs_select_candidate0_including_ties": all_pairs_bad,
            "interpretation": "A strictly positive candidate1-min minus candidate0-max proves every possible pair selects candidate1, independently of the matching implementation. This is observed finite-bank support, not generator support.",
        },
        "score_ordering_rule": "Ascending score then original index; descending uses negative score then original index.",
        "controls": controls,
        "feasibility_gate": {
            "independent_success_at_least_three_quarters": baseline_pass,
            "max_bad_minus_independent_bad_at_least_one_quarter": effect_pass,
            "independent_success": exact(independent_success),
            "max_bad_minus_independent_bad": exact(increment),
            "passed": baseline_pass and effect_pass,
        },
        "optimality_proof": [
            "Let P be a K-by-K joint law with each row and column sum 1/K. Q=K*P is doubly stochastic.",
            "By the Birkhoff-von Neumann theorem, Q is a convex combination of permutation matrices. A linear expected win objective therefore attains both extrema at permutations.",
            "For binary W, maximum matching on W=1 gives the largest possible number of bad-win edges. Completing unmatched rows/columns cannot add a favorable edge, or the matching was not maximum.",
            "Matching on W=0 analogously maximizes good-win edges; minimum bad probability is 1 minus that matching size/K.",
            "Each reported permutation, with a fresh uniform row index, uses each stored score once per candidate and exactly preserves both finite empirical marginal distributions, including duplicates.",
        ],
        "scope": {
            "utility_source": "Expectations splice supplied historical real continuation outcomes; no new environment execution is performed.",
            "conditioning": "All bounds condition on this observed finite bank and its known historical candidate utilities; they do not prove an unconditional continuous forecast-generator law or an admissible deployed attack.",
            "bank_identity_requirement": "Input producer must bind every score to its fixed candidate, forecast sample, original Value goal and the same scorer. This analyzer only validates the supplied JSON.",
            "sampling_controls": "Shared-index pairs original bank indices. Score-ordered independent is still product law; comonotone/countermonotone pair score ranks. Matching extrema are separate hindsight feasibility bounds.",
            "statistics": "One posthoc root. Exact finite-bank combinatorics only; no p-values, confidence intervals, independent-root generalization or new success-rate experiment.",
            "model_calls": 0, "environment_steps": 0, "api_calls": 0,
        },
    }


def self_test():
    cases = {
        "single_tie": [[0], [0]], "all_win": [[3, 4], [1, 2]],
        "all_lose": [[0, 1], [2, 3]], "all_ties": [[2, 2, 2], [2, 2, 2]],
        "mixed_ties": [[0, 1, 1], [0, 1, 2]],
        "disjoint_score_ranges": [[-2, -1, 8, 9], [0, 1, 2, 3]],
        "unordered_four": [[3, 0, 2, 1], [2, 1, 4, 0]],
    }
    for name, scores in cases.items():
        n = len(scores[0])
        result = analyze({"scores_by_candidate": scores,
                          "historical_continuation_utilities": {"success": [False, True], "capped_steps": [300, 170]}})
        all_counts = [sum(scores[0][i] >= scores[1][j] for i, j in enumerate(p))
                      for p in itertools.permutations(range(n))]
        controls = result["controls"]
        assert controls["minimum_bad_uniform_coupling"]["candidate0_selected_count"] == min(all_counts), name
        assert controls["maximum_bad_uniform_coupling"]["candidate0_selected_count"] == max(all_counts), name
        assert controls["independent_product"]["tie_count"] == sum(a == b for a in scores[0] for b in scores[1]), name
        assert controls["independent_product"] == controls["independent_score_ordered"], name
        for row in controls.values():
            bad = Fraction(row["candidate0_selected_count"], row["equiprobable_pair_count"])
            assert Fraction(row["expected_historical_success"]["fraction"]) == 1 - bad, name
            assert Fraction(row["expected_historical_capped_steps"]["fraction"]) == 170 + 130 * bad, name
    # Disjoint edges and an augmenting-path case exercise the generic matching routine.
    matrices = [[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
                [[1, 1, 0], [1, 0, 0], [0, 1, 1]]]
    for edges in matrices:
        permutation, certificate = maximum_matching_permutation(edges)
        optimum = max(sum(edges[i][j] for i, j in enumerate(p))
                      for p in itertools.permutations(range(len(edges))))
        assert certificate["maximum_matching_size"] == optimum
        assert sorted(permutation) == list(range(len(edges)))
    list_input = {"scores_by_candidate": [[0, 0], [0, 1]], "historical_continuation_utilities": [
        {"candidate_index": 1, "success": True, "steps": 170, "capped_steps": 170},
        {"candidate_index": 0, "success": False, "steps": 300, "capped_steps": 300}]}
    assert analyze(list_input)["controls"]["independent_product"]["tie_count"] == 2
    # Boundary values are inclusive and exact: bad=1/4, max bad=1/2.
    gate = analyze({"scores_by_candidate": [[0, 2], [1, 3]],
                    "historical_continuation_utilities": {"success": [False, True], "capped_steps": [300, 170]}})
    assert gate["feasibility_gate"]["passed"]
    print(json.dumps({"self_test": "passed", "score_cases_exhaustively_checked": len(cases),
                      "additional_binary_matching_cases": len(matrices), "largest_K": 4}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.summary is None and args.output is None and args.self_test:
        return 0
    if args.summary is None or args.output is None:
        parser.error("--summary and --output are required together")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    summary = json.loads(args.summary.read_text())
    result = analyze(summary)
    result["source_summary"] = str(args.summary)
    result["source_summary_status"] = summary.get("status")
    result["reproduction"] = ["python", "scripts/analyze_cosmos_forecast_coupling_gate.py",
                              "--summary", str(args.summary), "--output", "NEW_OUTPUT_PATH"]
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(serialized)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "bank_size_per_candidate": result["bank_size_per_candidate"],
                      "gate": result["feasibility_gate"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
