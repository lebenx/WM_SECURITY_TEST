#!/usr/bin/env python3
"""Recompute first-choice utility bounds from the existing audited Cosmos pair cache."""
from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import combinations_with_replacement, permutations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/cosmos_first_candidate_utility_recovery_analysis_v1/analysis.json"
OUTPUT = ROOT / "reports/cosmos_first_choice_identifiability_bounds_20261015.json"
REPRODUCTION = (
    "python scripts/analyze_cosmos_first_choice_bounds.py --verify-only; "
    "use --out <new-json-path> to write without overwriting. "
    "success_lower/upper=sum(min/max(s0,s1)); cost_lower/upper=sum(min/max(c0,c1)); "
    "subtract native totals or symbolic independent-scorer expectations. Standard-library JSON arithmetic only."
)


def conditional_k4_bound(success, costs):
    """Exact Ferrers enumeration; no measured or invented prediction scores."""
    k = 4
    assignments = list(permutations(range(k)))
    patterns = list(combinations_with_replacement(range(k + 1), k))
    max_increase = max_decrease = Fraction(0)
    witnesses = {}
    for counts in patterns:
        independent = Fraction(sum(counts), k * k)
        for assignment in assignments:
            coupled = Fraction(sum(column < counts[row] for row, column in enumerate(assignment)), k)
            delta = coupled - independent
            for name, value in (("increase", delta), ("decrease", -delta)):
                previous = max_increase if name == "increase" else max_decrease
                if value > previous:
                    if name == "increase":
                        max_increase = value
                    else:
                        max_decrease = value
                    witnesses[name] = {"row_counts": list(counts), "column_assignment": list(assignment),
                        "independent_choice_probability": str(independent),
                        "coupled_choice_probability": str(coupled), "signed_change": str(delta)}
    bound = max(max_increase, max_decrease)
    binary_range = sum(abs(s[1] - s[0]) for s in success)
    cost_range = sum(abs(c[1] - c[0]) for c in costs)
    binary_bound, cost_bound = bound * binary_range, bound * cost_range
    return {
        "status": "exact_finite_enumeration_of_conditional_mathematical_bound",
        "candidate_design_only": "K4 is a proposed design condition, not a registered or executed Cosmos attack/scorer.",
        "assumptions": [
            "Two fixed candidate identities, each with exactly four equally weighted scalar scores; hold the full bank, original action controls and real continuation utility fixed.",
            "Selection uses fixed argmax with a candidate-identity tie rule independent of bank-row identities; original Cosmos ties select candidate0.",
            "The reference chooses independent uniform column indices. The alternative joint law keeps both column-index marginals uniform; no score/action edits or utility-dependent bank selection.",
            "Sort score rows/columns consistently; candidate1 winning cells form a Ferrers matrix, represented up to orientation by nondecreasing row counts in0..4."],
        "enumeration": {"bank_size": k, "row_count_patterns": len(patterns), "permutations_per_pattern": len(assignments),
            "comparisons": len(patterns) * len(assignments), "arithmetic": "fractions.Fraction, exact rational arithmetic",
            "independent_probability_formula": "sum(row_counts)/16",
            "coupled_probability_formula": "sum_i[permutation[i]<row_counts[i]]/4",
            "max_choice_probability_increase": str(max_increase), "max_choice_probability_decrease": str(max_decrease),
            "max_absolute_choice_probability_change": str(bound),
            "proposed_three_eighths_confirmed": max_increase == max_decrease == Fraction(3, 8), "witnesses": witnesses},
        "extension_to_all_uniform_marginal_couplings": "Any4x4 nonnegative joint table with row/column sums1/4 is a convex mixture of uniform permutation tables. Selection probability is linear in that table, so the permutation extrema bound all such couplings. This is a finite application of the doubly-stochastic matrix decomposition, not a new theorem.",
        "cache_effect_bounds_relative_to_same_bank_independent_scorer": {
            "sum_absolute_binary_branch_differences": binary_range, "sum_absolute_capped_step_branch_differences": cost_range,
            "maximum_expected_success_loss": float(binary_bound), "exact_expected_success_loss_bound": str(binary_bound),
            "maximum_success_rate_drop_percentage_points": float(binary_bound * 100 / len(success)),
            "maximum_expected_capped_step_increase_total": float(cost_bound),
            "exact_capped_step_increase_total_bound": str(cost_bound),
            "maximum_mean_capped_steps_increase": float(cost_bound / len(costs)),
            "formulas": "success_loss <= choice_probability_bound*sum|s1-s0|; cost_increase <= choice_probability_bound*sum|c1-c0|. Also retain the tighter applicable support bounds S_ind-28 and3646-C_ind.",
            "meaning": "Bounds on expected aggregate utility, not maximum realized failures from one random row draw, nor a confidence bound."},
        "limits": [
            "A global extremal Ferrers pattern need not occur in an actual Cosmos bank. The actual restricted permutation family may be weaker, and simultaneous per-root harm is not shown achievable without access to true utility.",
            "This is relative to the independent uniform sampler on the same fixedK4 bank; it is not relative to native seed31/32 point-mass forecasts or the old deterministic native controller.",
            "The bounds do not transfer unchanged to nonuniform bank weights, more than two candidates, row-dependent tie rules, changed scores, differentK, or repeated closed-loop interventions.",
            "No forecast bank or attack output was generated; the only new computation is the1680-case mathematical enumeration."],
    }


def analyze(source):
    data = json.loads(source.read_text())
    rows = data["pairs"]
    n = len(rows)
    assert n == 30 and len({(r["task_id"], r["init_id"]) for r in rows}) == n
    assert data["validated_branches"] == 60 and data["complete_amended_grid"] is True
    assert data["selected_arm_full_source_parity_count"] == 30
    assert data["status"] == "completed_amended_development_validated"
    success = [[int(r[f"candidate{i}_success"]) for i in (0, 1)] for r in rows]
    costs = [[r[f"candidate{i}_capped_steps"] for i in (0, 1)] for r in rows]
    for r, s, c in zip(rows, success, costs):
        assert all(v in (0, 1) for v in s) and all(0 <= v <= 300 for v in c)
        selected = r["source_native_selected_index"]
        assert r["native_success"] == s[selected] and r["native_capped_steps"] == c[selected]
    smin, smax = (sum(fn(s) for s in success) for fn in (min, max))
    cmin, cmax = (sum(fn(c) for c in costs) for fn in (min, max))
    ns, nc = sum(r["native_success"] for r in rows), sum(r["native_capped_steps"] for r in rows)
    us, uc = sum(sum(s) / 2 for s in success), sum(sum(c) / 2 for c in costs)
    assert cmin == data["contribution"]["policies"]["oracle"]["capped_steps_total_or_expectation"]
    assert us == sum(r["uniform_success"] for r in rows) and uc == sum(r["uniform_capped_steps"] for r in rows)
    discordant = [r for r in rows if r["candidate0_success"] != r["candidate1_success"]]
    harmful = [r for r, s in zip(rows, success) if r["native_success"] > min(s)]
    assert len(harmful) == 1  # This entry describes this frozen cache, not arbitrary panels.
    p_expression = str(sum(s[0] for s in success))
    for r, s in zip(rows, success):
        if s[0] != s[1]:
            p_expression += ("+" if s[1] > s[0] else "-") + f"p_(task{r['task_id']},init{r['init_id']})"
    source_path = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
    return {
        "schema_version": 1, "status": "completed_cache_only_identifiability_bounds", "date": "2026-10-15",
        "source": source_path,
        "scope": "Fixed development cache:30 roots,2 first16-control candidates, then the unchanged original-Goal fixed-seed native continuation. These are bounds on the recorded branch outcomes, not population or stochastic-environment guarantees.",
        "prior_coverage": {
            "existing": "The source already reports native/fixed/uniform utilities, two discordant-success roots and hindsight best-branch totals.",
            "incremental": "This calculation adds explicit worst-branch totals and maximum possible first-choice damage against native or an unknown randomized scorer; no same-named prior bound artifact was found in the narrow report check. No new model or task evidence."},
        "source_validation": {"validated_branches": data["validated_branches"], "complete_amended_grid": True,
            "selected_arm_full_source_parity_count": data["selected_arm_full_source_parity_count"], "source_status": data["status"],
            "raw_trace_audit": "Reuse the existing independent branch audit; no repeated raw audit or hashes."},
        "identification": {"binary_discordant_roots": len(discordant), "binary_tie_roots": n-len(discordant),
            "capped_cost_discordant_roots": sum(c[0] != c[1] for c in costs),
            "binary_cases": [{"task_id": r["task_id"], "init_id": r["init_id"],
                "candidate_success": [r["candidate0_success"], r["candidate1_success"]],
                "candidate_capped_steps": [r["candidate0_capped_steps"], r["candidate1_capped_steps"]],
                "native_index": r["source_native_selected_index"]} for r in discordant]},
        "any_first_choice_selector": {"success_count_or_expectation_interval": [smin, smax],
            "success_rate_interval": [smin/n, smax/n], "capped_step_total_or_expectation_interval": [cmin, cmax],
            "mean_capped_steps_interval": [cmin/n, cmax/n],
            "definition": "Sum per-root min/max across the two recorded real continuation outcomes. Any stochastic selector is a convex combination within each root. Extremizers use hindsight outcome knowledge and are not feasible attacks."},
        "relative_to_recorded_native": {"native_success_count": ns, "native_capped_steps_total": nc,
            "maximum_success_loss_count": ns-smin, "maximum_success_rate_drop_percentage_points": 100*(ns-smin)/n,
            "maximum_loss_fraction_among_native_successes": (ns-smin)/ns,
            "maximum_number_of_tasks_with_a_new_success_loss": len({r["task_id"] for r in harmful}),
            "only_possible_new_binary_loss_root": {k: harmful[0][k] for k in ("task_id", "init_id")},
            "maximum_capped_steps_increase_total": cmax-nc, "maximum_mean_capped_steps_increase": (cmax-nc)/n,
            "maximum_relative_capped_cost_increase": (cmax-nc)/nc, "capped_steps_change_interval": [cmin-nc, cmax-nc]},
        "relative_to_unknown_independent_finite_bank_scorer": {"measured": False,
            "p_definition": "p_r = P(independent scorer selects candidate1 at root r), including the declared score tie rule. No p_r or forecast bank has been measured here.",
            "baseline_success_expression": f"S_ind = sum_r [(1-p_r)*s_r0+p_r*s_r1]; for this cache S_ind={p_expression}.",
            "baseline_cost_expression": "C_ind = sum_r [(1-p_r)*c_r0+p_r*c_r1].",
            "success_loss_outer_bound_expression": f"S_ind-{smin}; no more than{smax-smin} expected successes across the panel without knowing S_ind.",
            "maximum_distribution_free_success_loss_count": smax-smin,
            "maximum_distribution_free_success_rate_drop_percentage_points": 100*(smax-smin)/n,
            "cost_increase_outer_bound_expression": f"{cmax}-C_ind; no more than{cmax-cmin} total capped steps without knowing C_ind.",
            "maximum_distribution_free_capped_steps_increase_total": cmax-cmin,
            "maximum_distribution_free_mean_capped_steps_increase": (cmax-cmin)/n,
            "coupling_limit": "These are outer bounds for arbitrary first selectors. Keeping finite score-column marginals and using a restricted permutation can only narrow reachability; no unknown bank, selection probabilities, seed mixture or attainable permutation is invented. Independent forecast-column indices do not imply uniform candidate choice."},
        "uniform_candidate_reference_separate": {"success_count_expectation": us, "capped_steps_total_expectation": uc,
            "maximum_success_loss_count": us-smin, "maximum_capped_steps_increase_total": cmax-uc,
            "maximum_mean_capped_steps_increase": (cmax-uc)/n,
            "scope": "Existing fair coin over the two candidate identities, not the unmeasured independent finite-bank scoring baseline."},
        "decision": {
            "before_new_inference": "Exclude any claim of three or more new failed roots, or losses across two tasks, relative to the recorded native controller under this first-choice-only design. Relative to any new randomized scorer, more than two lost successes is impossible on these stored branches.",
            "existing_first_choice_effect_already_reaches_binary_floor": "reports/cosmos_first_selection_mediation_20261015.json already derives29 native successes versus28 under the recorded Goal-Swap first choice using these same real continuations. Its one local loss reaches this cache's28-success floor; a different first-choice scorer cannot establish greater binary harm against the recorded native baseline here. This does not make the previous failed extension gate pass or establish a finite-bank operation.",
            "what_remains_testable": "A bounded local or capped-step effect remains possible, but requires a pre-frozen operation blind to the real branch-utility table and a correctly labeled added stochastic scorer. Bounds alone give no attack signal.",
            "not_generalized": "No claim about interventions at later decisions, different candidates, other roots, environmental repetitions, full closed-loop randomized planners or population ASR."},
        "activity": {"new_model_calls": 0, "new_environment_steps": 0, "api_requests": 0, "training_updates": 0, "hashes_recomputed": 0},
        "conditional_uniform_k4_coupling_bound": conditional_k4_bound(success, costs),
        "reproduction": REPRODUCTION,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--verify-only", action="store_true", help="Compare the existing report without writing")
    args = parser.parse_args()
    result = analyze(args.source.resolve())
    if args.verify_only:
        saved = json.loads(args.out.read_text())
        assert saved == result, "Recomputed report differs from existing report"
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"status": "verified_exact" if args.verify_only else "written",
        "report": str(args.out), "success_interval": result["any_first_choice_selector"]["success_count_or_expectation_interval"],
        "cost_interval": result["any_first_choice_selector"]["capped_step_total_or_expectation_interval"],
        "writes": int(not args.verify_only), "model_calls": 0, "environment_steps": 0}))


if __name__ == "__main__":
    main()
