"""Audit saved CartPole CEM logs without importing a model or environment.

Infers the iteration supplying the returned best candidate from native CEM's
strict-improvement rule. It does not reconstruct unrecorded candidates or test
a counterfactual optimizer. All paths are local; no network or weight IO.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/cem_dispersion_action_source_audit_20261015.json"
MODES = ("random", "boundary", "interior")
SEEDS = (810, 811, 812, 813)
PIN = "3f93cccfc8d635f74e335a2f07aab6e9a48fc021"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads((ROOT / path).read_text())


def anchor(path, snippet):
    text = (ROOT / path).read_text()
    require(snippet in text, f"Expected source contract absent: {path}: {snippet}")
    return {"path": path, "line": text[:text.index(snippet)].count("\n") + 1,
            "required_source": snippet}


def source_contract():
    path = "vendor/mbrl-lib/mbrl/planning/trajectory_opt.py"
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CEMOptimizer")
    optimize = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "optimize")
    strict = [n for n in ast.walk(optimize) if isinstance(n, ast.If)
              and isinstance(n.test, ast.Compare) and len(n.test.ops) == 1
              and isinstance(n.test.ops[0], ast.Gt)
              and ast.unparse(n.test.left) == "best_values[0]"
              and ast.unparse(n.test.comparators[0]) == "best_value"]
    require(len(strict) == 1, "CEM must retain best candidate unless value strictly increases")
    require(any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "best_solution"
                for t in n.targets) for n in strict[0].body), "Strict branch must assign best_solution")
    ret = next(n for n in reversed(optimize.body) if isinstance(n, ast.Return))
    require(ast.unparse(ret.value) == "mu if self.return_mean_elites else best_solution",
            "CEM return contract changed")
    # Compare only three small source files to the archived upstream commit.
    # No model/checkpoint files or repeated full-tree hashes are read.
    git_args = ["git", "-C", str(ROOT / "vendor/mbrl-lib"), "diff", "--quiet", PIN, "--",
                "mbrl/planning/trajectory_opt.py", "mbrl/env/reward_fns.py", "mbrl/models/model_env.py"]
    checked = subprocess.run(git_args, capture_output=True, text=True)
    require(checked.returncode == 0, "Native source differs from the recorded MBRL pin: " + checked.stderr)
    return {
        "native_source_matches_recorded_commit": True, "recorded_commit": PIN,
        "checks": [
            anchor(path, "if best_values[0] > best_value:"),
            anchor(path, "return mu if self.return_mean_elites else best_solution"),
            anchor(path, "constrained_var = torch.min(mv, dispersion)"),
            anchor(path, "new_dispersion = torch.var(elite, dim=0)"),
            anchor(path, "self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)"),
            anchor("vendor/mbrl-lib/mbrl/env/reward_fns.py",
                   "return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)"),
            anchor("vendor/mbrl-lib/mbrl/models/model_env.py", "return total_rewards.mean(dim=1)"),
            anchor("scripts/pilot_mbrl_rank_general.py", "cov = torch.cov(flat.T)"),
            anchor("scripts/pilot_mbrl_rank_general.py", "elif args.collection_mode == 'boundary':"),
        ],
        "score_upper_bound": 15,
        "reason": "The registered horizon is15; each native model reward is0 or1, and scores are particle-mean accumulated rewards.",
    }


def covariance_audit():
    episodes, training = [], []
    for mode in MODES:
        base = f"runs/mbrl_covariance_pinching_{mode}_v0"
        train = read_json(base + "/summary.json")
        plan = read_json(base + "/plan.json")
        args = plan["arguments"]
        require(train["status"] == "trained" and train["task"] == "cartpole", "Wrong trained carrier")
        require(args["collection_mode"] == mode and args["transitions"] == 3000
                and args["train_seed"] == 18 and args["epochs"] == 40, "Unexpected training panel")
        training.append({"mode": mode, "source": base + "/summary.json",
                         "carrier": "Locally trained MBRL-Lib CartPole GaussianMLP ensemble",
                         **{k: train[k] for k in ("native_collection_steps", "training_seed", "epochs",
                                                "gradient_updates", "final_validation", "collection_resets")}})
        for version in (0, 1):
            run = f"runs/mbrl_covariance_pinching_eval_{mode}_metrics_v{version}"
            summary, plan = read_json(run + "/summary.json"), read_json(run + "/plan.json")
            args = plan["arguments"]
            require(summary["status"] == "completed" and plan["mbrl_pin"] == PIN, "Incomplete or wrong-pin run")
            expected = {"task": "cartpole", "condition": "native", "population": 100,
                        "iterations": 3, "horizon": 15, "particles": 10, "elite_ratio": 0.1,
                        "alpha": 0.1, "return_mean_elites": False, "model": base}
            require(all(args[k] == v for k, v in expected.items()), "Unexpected native CEM protocol")
            require(args["seeds"] == list(SEEDS[2*version:2*version+2]), "Changed old evaluation seeds")
            for saved in summary["rows"]:
                seed = saved["seed"]
                file = run + f"/episode_{seed}.json"
                data = read_json(file)
                require(data["summary"] == saved, "Episode and aggregate summaries disagree")
                trace = data["trajectory"]
                require(len(trace) == saved["steps"], "Incomplete real trajectory")
                require([r["step"] for r in trace] == list(range(len(trace))), "Noncontiguous real steps")
                require(all(len(r["obs"]) == 4 and len(r["action"]) == 1 for r in trace), "Wrong state/action dimension")
                total = math.fsum(float(r["reward"]) for r in trace)
                require(total == saved["return_"], "Return differs from raw rewards")
                grouped = defaultdict(list)
                for row in saved["cem_stats"]:
                    grouped[row["step"]].append(row)
                require(sorted(grouped) == list(range(len(trace))), "Missing planner queries")
                provenance = []
                for step, records in sorted(grouped.items()):
                    records.sort(key=lambda r: r["iteration"])
                    require([r["iteration"] for r in records] == [0, 1, 2], "Incomplete CEM iterations")
                    maxima = [float(r["value_max"]) for r in records]
                    require(all(math.isfinite(x) and 0 <= x <= 15 for x in maxima), "Invalid score maxima")
                    best, winner = -math.inf, None
                    for iteration, maximum in enumerate(maxima):
                        if maximum > best:
                            best, winner = maximum, iteration
                    provenance.append({"query_step": step, "iteration_maxima": maxima,
                                       "returned_candidate_origin_iteration_inferred": winner,
                                       "first_iteration_reaches_reward_ceiling": maxima[0] == 15,
                                       "later_iterations_do_not_replace_observed_first_winner": winner == 0})
                first = grouped[0]
                counts = Counter(r["returned_candidate_origin_iteration_inferred"] for r in provenance)
                ceiling_count = sum(r["first_iteration_reaches_reward_ceiling"] for r in provenance)
                episodes.append({"mode": mode, "seed": seed, "source": file,
                                 "return_from_real_rewards": total, "real_steps": len(trace),
                                 "initial_observation": trace[0]["obs"],
                                 "first_executed_action": trace[0]["action"],
                                 "first_query": {**provenance[0],
                                     "population_cov_trace": [r["population_cov_trace"] for r in first],
                                     "elite_cov_trace": [r["elite_cov_trace"] for r in first],
                                     "population_iteration2_over_iteration0":
                                         first[2]["population_cov_trace"] / first[0]["population_cov_trace"]},
                                 "origin_iteration_counts": {str(i): counts[i] for i in (0, 1, 2)},
                                 "first_iteration_ceiling_count": ceiling_count,
                                 "first_iteration_ceiling_fraction": ceiling_count / len(provenance),
                                 "every_query_reaches_ceiling_in_first_iteration": ceiling_count == len(provenance),
                                 "queries": provenance})
    require(len(episodes) == 12, "Expected twelve completed old episodes")
    paired = []
    for seed in SEEDS:
        rows = {r["mode"]: r for r in episodes if r["seed"] == seed}
        same_obs = all(r["initial_observation"] == rows["random"]["initial_observation"] for r in rows.values())
        same_cov = len({r["first_query"]["population_cov_trace"][0] for r in rows.values()}) == 1
        require(same_obs and same_cov, "Initial common-state query summaries do not match")
        paired.append({"seed": seed, "initial_observations_exactly_equal": same_obs,
                       "initial_population_cov_trace_exactly_equal": same_cov,
                       "first_actions_by_mode": {m: rows[m]["first_executed_action"] for m in MODES},
                       "first_query_contraction_by_mode": {
                           m: rows[m]["first_query"]["population_iteration2_over_iteration0"] for m in MODES}})
    totals = {mode: {"queries": sum(r["real_steps"] for r in episodes if r["mode"] == mode),
                     "mean_real_return": math.fsum(r["return_from_real_rewards"] for r in episodes if r["mode"] == mode) / 4,
                     "first_iteration_ceiling_count": sum(r["first_iteration_ceiling_count"] for r in episodes if r["mode"] == mode),
                     "origin_iteration_counts": {
                         str(i): sum(r["origin_iteration_counts"][str(i)] for r in episodes if r["mode"] == mode)
                         for i in (0, 1, 2)}} for mode in MODES}
    return {"training": training, "episodes": episodes, "initial_query_pairing": paired,
            "descriptive_totals_not_independent_samples": totals,
            "conditional_whole_recorded_episode_invariance": {
                "qualifying_episodes": [{"mode": r["mode"], "seed": r["seed"],
                                         "real_steps": r["real_steps"]}
                                        for r in episodes if r["every_query_reaches_ceiling_in_first_iteration"]],
                "scope": "A source-and-log conditional deduction, not an executed intervention.",
                "assumptions": [
                    "Only within-query dispersion updates after iteration0 change; initial dispersion, first population, objective, score bound and winner/tie rule remain fixed.",
                    "All candidate/model RNG streams consume the same numbers of draws in the same order; no additional stateful model/cache side effect changes future queries.",
                    "Warm starts use returned best_solution, as the pinned wrapper does, not final mu/dispersion; real environment evolution is unchanged for identical controls.",
                ],
                "deduction": "If every observed query first reaches15, induction preserves each returned best solution, next warm start, executed action and next real state for the entire recorded episode under these assumptions. This does not extend to the other episodes or interventions changing first-round sampling, RNG consumption, mean-dependent score, or model outputs.",
            }}


def parity_audit():
    rows = []
    for version in (0, 1):
        values = defaultdict(dict)
        sources = []
        for arm in ("independent", "parity_control", "parity_attack"):
            path = f"runs/mbrl_parity_cartpole_v{version}/{arm}/summary.json"
            if not (ROOT / path).exists():
                continue
            data = read_json(path)
            require(data["status"] == "completed", "Incomplete parity source")
            sources.append(path)
            for row in data["rows"]:
                values[row["seed"]][arm] = row["return_"]
        for seed, arms in sorted(values.items()):
            require("parity_attack" in arms and "parity_control" in arms, "Missing parity arm")
            independent = arms.get("independent")
            rows.append({"seed": seed, "returns": arms, "sources": sources,
                         "attack_minus_control": arms["parity_attack"] - arms["parity_control"],
                         "attack_minus_independent": None if independent is None else arms["parity_attack"] - independent,
                         "lower_than_both_original_comparators": None if independent is None else
                             arms["parity_attack"] < min(independent, arms["parity_control"])})
    return {"rows": rows,
            "control_definition": "parity_control adaptively maximizes expected maximum score within each triple; it is an opposite-objective operation, not neutral independent sampling.",
            "source": anchor("src/wm_security/parity_coupling.py",
                             "orientation = 0 if even_score >= odd_score else 1"),
            "interpretation": "Seed704 has lower return than parity_control but higher return than independent. Seed705 is lower than both. Held-out706 ties parity_control;707 improves. Missing v1 independent runs are untested, not failures."}


def audit():
    return {
        "schema_version": 1, "status": "completed_offline_source_and_saved_log_audit",
        "activity": {"model_imports": 0, "checkpoint_loads": 0, "model_calls": 0,
                     "environment_calls": 0, "api_calls": 0, "training_updates": 0,
                     "new_real_episodes": 0},
        "reproduction": ".venv-opine/bin/python scripts/audit_cem_dispersion_action_source.py --verify-only",
        "source_contract": source_contract(),
        "covariance": covariance_audit(), "parity": parity_audit(),
        "scientific_boundaries": [
            "The covariance carrier is locally trained MBRL-Lib GaussianMLP CartPole, not TD-MPC2 or an externally released checkpoint. Training data, normalization and weights differ between collection modes.",
            "Own-distribution validation scores are not cross-distribution forecast fidelity. Boundary's smaller validation score does not establish superior victim-state accuracy, and cannot support a claim of worse predictive accuracy either.",
            "Logged population covariance is sampled-candidate dispersion. Native effective sampling variance additionally depends on distance from the action bounds; it is not identical to the stored optimizer dispersion.",
            "Across all four seeds, the three models have equal initial physical observations and first-iteration covariance summaries. Subsequent pooled covariance differs across visited states, warm starts and episode durations, so it is not a matched causal mediation estimate.",
            "When iteration0 already reaches the bounded score15, later CEM iterations cannot strictly improve the best value and therefore cannot replace that query's returned candidate. This statement concerns the unchanged observed first iteration, not arbitrary changes to the first iteration or model.",
            "An inferred origin0 without reaching15 only proves that the observed later iterations did not replace it. A different dispersion could have produced a better later candidate; no counterfactual invariance follows.",
            "Many later queries return candidates from iteration1 or2. A first-query counterexample therefore does not disprove a whole-episode covariance-mediated effect. Query counts are dependent observations within twelve episodes, not independent task samples.",
            "Saved logs lack full candidate populations, elite identities, per-coordinate means and dispersion. The audit infers the winning iteration under the pinned source; it cannot verify the identity of the returned candidate or recompute a counterfactual CEM action from logs alone.",
            "Parity's exact finite-bank pairwise-law construction remains valid; its comparison against an actively optimized parity control is not equivalent to harm against independent sampling. No extra parity rollout is warranted by these records alone.",
        ],
        "next_decisive_control": {
            "status": "proposed_only_audit_does_not_execute",
            "mechanism": "Necessity of native learned-dispersion updates for the existing boundary-data return gap",
            "new_reason": "The old screen pooled sampled covariance over different trajectories and did not establish that covariance updates reached the executed action; the pinned optimizer returns a strictly improving best candidate rather than the final elite mean.",
            "operation": "On the three existing frozen CartPole checkpoints and the original810..813 panel only, compare native CEM to one dispersion-only intervention: retain each query's native initial optimizer dispersion through its three iterations while preserving native mean updates, legal bound constraint, winner/tie rule,100 candidates,15-step horizon,10 particles and RNG draw counts. No retraining, new seed, action-source data or reward changes.",
            "necessary_measurements": "Reproduce historical native traces first. Record actual per-coordinate optimizer dispersion, bound-constrained sampling variance, winner iteration and real returns. Confirm that the intervention changes effective sampling variance on relevant non-ceiling queries; otherwise it did not manipulate the hypothesized mediator.",
            "interpretation_if_selective_recovery": "Boundary-specific recovery beyond the random/interior changes would support a local learned-dispersion mediator. It would not establish a new security principle, a realistic attacker interface or an externally pretrained-system effect.",
            "interpretation_if_no_selective_recovery": "Do not promote the claim that learned-dispersion collapse explains the existing harm. Preserve the real data-distribution effect and do not search seeds or variance floors for a positive outcome.",
            "limits": "This distinguishes the elite-dispersion update from mean-dependent bound clipping, not every possible source of candidate covariance change. Normal-control returns may be ceiling-censored; report the complete original panel and do not call generic recovery selective. This is a mechanism eligibility control, not a sufficient paper acceptance test.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true", help="Recompute all checks without writing files")
    args = parser.parse_args()
    result = audit()
    if not args.verify_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print(json.dumps({"status": result["status"], "files_written": int(not args.verify_only),
                      "covariance_totals": result["covariance"]["descriptive_totals_not_independent_samples"],
                      "parity": [{k: v for k, v in r.items() if k != "sources"} for r in result["parity"]["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
