#!/usr/bin/env python3
"""Read-only grouped-CV audit of saved private-query feature summaries.

Uses only JSON and NumPy arithmetic. Does not import a world model, simulator,
planner, training stack, or network client. Original runs remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/private_query_grouped_cv_audit_20261015"


def read_json(relative_path):
    return json.loads((ROOT / relative_path).read_text())


def validate_rows(rows):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["rep"] for row in rows], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Expected exactly two original preference labels")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two paired repetition groups are required")
    for group in unique_groups:
        selected = labels[groups == group]
        if len(selected) != 2 or set(selected.tolist()) != {0, 1}:
            raise ValueError("Each repetition must contain exactly one row per label")
    query = np.asarray([row["summary"] for row in rows], dtype=np.float64)
    action = np.asarray([row["first_action"] for row in rows], dtype=np.float64)
    if query.ndim != 2 or action.ndim != 2:
        raise ValueError("Features must be rectangular matrices")
    if not np.isfinite(query).all() or not np.isfinite(action).all():
        raise ValueError("Nonfinite saved feature")
    return labels, groups, query, action


def nearest_centroid_cv(features, labels, groups, *, grouped):
    """Match original raw-scale Euclidean centroids; change only held-out rows."""
    classes = np.unique(labels)  # Original implementation's sorted class order.
    predictions = []
    rows = []
    for index in range(len(labels)):
        train = groups != groups[index] if grouped else np.arange(len(labels)) != index
        centers = np.stack([features[train & (labels == c)].mean(0) for c in classes])
        distances = np.square(features[index] - centers).mean(1)
        minimum = distances.min()
        tied_classes = classes[distances == minimum]
        # The original min(dict, key=...) chooses the first class on exact ties.
        predicted = int(tied_classes[0])
        predictions.append(predicted)
        rows.append({
            "row": index,
            "rep": int(groups[index]),
            "label": int(labels[index]),
            "prediction": predicted,
            "class_mean_squared_distances": {str(int(c)): float(d) for c, d in zip(classes, distances)},
            "exact_tie": len(tied_classes) > 1,
            "train_rows_per_class": {str(int(c)): int(np.sum(train & (labels == c))) for c in classes},
            "test_pair_present_in_training": bool(np.any(train & (groups == groups[index]))),
        })
    predictions = np.asarray(predictions)
    recalls = {str(int(c)): float(np.mean(predictions[labels == c] == c)) for c in classes}
    return {
        "split": "leave_one_repetition_pair_out" if grouped else "original_leave_one_row_out",
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "recall_by_class": recalls,
        "correct_rows": int(np.sum(predictions == labels)),
        "test_rows": len(labels),
        "folds": len(np.unique(groups)) if grouped else len(labels),
        "exact_tie_rows": sum(row["exact_tie"] for row in rows),
        "test_rows_with_paired_seed_in_training": sum(row["test_pair_present_in_training"] for row in rows),
        "predictions": rows,
    }


def paired_feature_differences(features, labels, groups):
    rows = []
    for group in np.unique(groups):
        x0 = features[(groups == group) & (labels == 0)][0]
        x1 = features[(groups == group) & (labels == 1)][0]
        difference = np.abs(x0 - x1)
        rows.append({"rep": int(group), "exactly_equal": bool(np.array_equal(x0, x1)),
                     "mean_absolute_difference": float(difference.mean()),
                     "max_absolute_difference": float(difference.max())})
    return {
        "pairs": len(rows),
        "exactly_equal_pairs": sum(row["exactly_equal"] for row in rows),
        "all_pairs_exactly_equal": all(row["exactly_equal"] for row in rows),
        "mean_paired_absolute_difference": float(np.mean([row["mean_absolute_difference"] for row in rows])),
        "maximum_paired_absolute_difference": max(row["max_absolute_difference"] for row in rows),
        "tolerance": "Exact equality; no data-dependent tolerance or feature filtering",
        "rows": rows,
    }


def audit_dataset(identifier, source, rows, reported_accuracy, scope):
    labels, groups, query, action = validate_rows(rows)
    feature_sets = {"query_summary": query, "first_action": action,
                    "query_and_first_action": np.concatenate([query, action], axis=1)}
    results = {}
    for name, features in feature_sets.items():
        results[name] = {
            "dimensions": int(features.shape[1]),
            "pair_equality": paired_feature_differences(features, labels, groups),
            "original_row_loo": nearest_centroid_cv(features, labels, groups, grouped=False),
            "grouped_pair_cv": nearest_centroid_cv(features, labels, groups, grouped=True),
        }
    reproduced = results["query_summary"]["original_row_loo"]["accuracy"]
    if not np.isclose(reproduced, reported_accuracy, atol=1e-12, rtol=0):
        raise ValueError(f"Original query accuracy did not reproduce for {identifier}")
    return {
        "id": identifier,
        "source_path": source,
        "sample_rows": len(labels),
        "repetition_groups": len(np.unique(groups)),
        "rows_per_group": 2,
        "labels": [0, 1],
        "label_counts": {str(int(c)): int(np.sum(labels == c)) for c in np.unique(labels)},
        "independent_environment_roots": 1,
        "distinct_preference_pairs": 1,
        "scope": scope,
        "reported_query_row_loo_accuracy": reported_accuracy,
        "original_query_accuracy_reproduced": True,
        "features": results,
        "descriptive_joined_minus_action_accuracy": (
            results["query_and_first_action"]["grouped_pair_cv"]["accuracy"]
            - results["first_action"]["grouped_pair_cv"]["accuracy"]),
        "incremental_privacy_identified": False,
    }


def datasets():
    for version in (0, 1):
        directory = f"runs/private_counterfactual_query_v{version}"
        summary = read_json(f"{directory}/summary.json")
        yield (f"private_counterfactual_query_v{version}", f"{directory}/traces.json",
               read_json(f"{directory}/traces.json"), summary["query_only_goal_classification_accuracy"],
               {"task": "walker-walk", "state_seed": 31,
                "preference": "Added random terminal latent target; no validated private physical task",
                "objective": "Terminal latent MSE only" if version == 0 else "Non-native reward-logit proxy plus weight0.05 latent preference",
                "development_seed_formula": "9000 + attempted_pair_index",
                "validation_seed_formula": "12000 + rep",
                "selected_action_physically_executed": False})
    for version in (0, 1):
        path = f"runs/private_query_geometry_search_v{version}/summary.json"
        summary = read_json(path)
        for result in summary["results"]:
            yield (f"private_query_geometry_search_v{version}/{result['task']}", path,
                   result["validation_rows"], result["validation"]["accuracy"],
                   {"task": result["task"], "state_seed": 31,
                    "preference": "Added random terminal latent target; pair selected on development seeds",
                    "objective": "Non-native reward-logit proxy plus latent preference",
                    "private_weight": result["weight"],
                    "development_seed_formula": "900000 + attempted_pair_index",
                    "validation_seed_formula": "910000 + rep",
                    "selected_action_physically_executed": False})


def counterexample():
    # X is identically distributed for both labels: each label has X=0 and X=1.
    labels = np.asarray([0, 0, 1, 1])
    groups = np.asarray([0, 1, 0, 1])
    features = np.asarray([[0.0], [1.0], [0.0], [1.0]])
    row = nearest_centroid_cv(features, labels, groups, grouped=False)
    paired = nearest_centroid_cv(features, labels, groups, grouped=True)
    assert row["accuracy"] == 0.0 and paired["accuracy"] == 0.5
    assert paired["exact_tie_rows"] == 4
    return {
        "rows": [{"label": int(y), "rep": int(g), "feature": float(x[0])}
                 for y, g, x in zip(labels, groups, features)],
        "equal_class_feature_distributions": True,
        "original_row_loo": row,
        "grouped_pair_cv": paired,
        "general_identity": "For n pairs with x_i under both labels and overall mean m, row-LOO gives own centroid (n*m-x_i)/(n-1) and other centroid m. The distance to the own centroid is n/(n-1) times the distance to the other centroid; squared distance has ratio (n/(n-1))**2. If x_i != m, prediction is mechanically the opposite label. Pair holdout instead gives identical class centroids. This is a CV construction effect, not evidence that negating0% predictions yields a valid100% classifier.",
        "explanation": "Each pair has identical features under opposite labels. Row-LOO removes the test feature only from its own-class centroid but retains its identical opposite-label mate. The opposite centroid is closer, so every prediction is wrong. Holding out both mates gives equal class centroids and 50% accuracy under the fixed lowest-label tie rule. Zero percent therefore need not mean an invertible privacy signal or a valid no-leakage estimate.",
    }


def source_bindings():
    pairs = [
        ("scripts/probe_private_counterfactual_queries.py", "runs/private_counterfactual_query_v0/plan.json"),
        ("scripts/probe_private_counterfactual_queries_v1.py", "runs/private_counterfactual_query_v1/plan.json"),
        ("scripts/probe_private_query_geometry.py", "runs/private_query_geometry_search_v0/summary.json"),
        ("scripts/probe_private_query_geometry.py", "runs/private_query_geometry_search_v1/summary.json"),
    ]
    result = []
    current = {}
    for script, record in pairs:
        if script not in current:
            current[script] = hashlib.sha256((ROOT / script).read_bytes()).hexdigest()
        recorded = read_json(record)["source_sha256"]
        result.append({"script": script, "record": record, "recorded_sha256": recorded,
                       "current_sha256": current[script], "matches": current[script] == recorded})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty; preserve existing audit evidence")
    output.mkdir(parents=True, exist_ok=True)
    results = [audit_dataset(*arguments) for arguments in datasets()]
    report = {
        "schema_version": 1,
        "status": "completed_offline_grouped_cv_audit",
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "command": f"{sys.executable} scripts/reanalyze_private_query_grouped_cv.py --output {output.relative_to(ROOT)}",
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "method": {
            "classifier": "Nearest class centroid using mean squared Euclidean distance, exactly the original raw feature scale",
            "features": "Saved query summary; saved selected first action; raw concatenation query_summary followed by first_action",
            "preprocessing": "None: no rescaling, feature selection, regularization, hyperparameter search or best-representation selection",
            "grouping": "Hold out both preference labels with the same rep; rep maps to one shared planner RNG seed",
            "tie_rule": "Exact float64 distance equality; predict lowest numeric class (0), matching original min over sorted classes; report all ties",
            "uncertainty": "Descriptive finite-panel accuracies only. Repetition groups are RNG realizations at one fixed initial state and one selected preference pair, not independent tasks or trajectories",
            "paired_label_swap_test": "Not performed: the saved design supplies no randomized per-pair label assignment or justified conditional exchangeability model for the incremental-leakage null; no post-hoc significance test is substituted",
        },
        "datasets": results,
        "no_information_counterexample": counterexample(),
        "scientific_limits": [
            "Equal selected actions are sufficient for one matched privacy contrast but are not necessary for incremental leakage conditional on publicly observed actions or leakage before any action is visible.",
            "A query/action/joined classifier comparison is a descriptive prediction comparison, not a test of conditional independence or an estimate of conditional mutual information. A weak action classifier can leave apparent gains even when the exact action already identifies the preference.",
            "Only 12 or16 paired RNG groups, one environment initial state and one development-selected preference pair are available per dataset. No independent state/task/private-goal generalization is identified.",
            "Original row-LOO retains the opposite-label shared-seed mate in training. It does not evaluate a held-out repetition and can systematically produce0% on no-information paired features. Grouped CV removes that dependence.",
            "The preference pair searches use different planner seeds from the saved validation rows. This avoids direct validation-seed reuse in pair selection but does not turn the selected pair into an independent distribution of real private objectives.",
            "Saved traces contain feature summaries and selected first actions, not the complete1920-query transcripts or timestamps. Prefix/lead-time leakage and arbitrary query classifiers cannot be reconstructed.",
            "No selected first action was physically executed in these screens. Source control flow places the summaries before returning an action, so a result can only concern pre-action computation in the added planner, not measured real-world timing or utility.",
            "v0 replaces native MPC scoring with MSE to added random latent targets. v1 and geometry use reward-head logit means instead of native two-hot reward decoding and omit native terminal Q; these are added planners, not native TD-MPC2 private-task evaluations.",
            "In v1 and geometry, base_returns/public is initialized outside the CEM iteration loop and accumulates across unrelated candidate populations. This implementation defect affects the original proxy objective; statistical reanalysis cannot repair its generated queries.",
            "Exact equality or chance accuracy of the saved summaries does not prove equality of the unsaved full query transcripts or universal privacy. High query accuracy alone does not establish leakage beyond actions.",
        ],
        "source_code_review": [
            "scripts/probe_private_counterfactual_queries.py",
            "scripts/probe_private_counterfactual_queries_v1.py",
            "scripts/probe_private_query_geometry.py",
        ],
        "source_code_bindings": source_bindings(),
        "activity": {"pretrained_model_loads": 0, "model_forwards": 0, "environment_steps": 0,
                     "training_updates": 0, "api_calls": 0, "original_files_modified": 0,
                     "checkpoint_hashes_recomputed": 0, "statistical_work": "JSON feature equality and arithmetic centroid CV only"},
        "decision": "Correct the historical classifier interpretation; do not revive a native privacy claim or infer real incremental leakage from this audit",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    compact = []
    for dataset in results:
        compact.append({"id": dataset["id"], "groups": dataset["repetition_groups"],
                        "row_loo_query": dataset["features"]["query_summary"]["original_row_loo"]["accuracy"],
                        "paired_cv": {key: value["grouped_pair_cv"]["accuracy"] for key, value in dataset["features"].items()},
                        "equal_query_pairs": dataset["features"]["query_summary"]["pair_equality"]["exactly_equal_pairs"]})
    print(json.dumps({"status": report["status"], "results": compact}, indent=2))


if __name__ == "__main__":
    main()
