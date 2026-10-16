#!/usr/bin/env python3
"""Audit existing GenSafeNav entity binding using two JSONL logs only.

No predictor, actor, simulator, random generator, or tensor framework is imported.
Current-point ownership uses exact recorded FP32 observation arithmetic, not a
nearest-person heuristic. The unobserved ordering among dummy rows is unresolved.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct


DEFAULT_RUN = "runs/gensafenav_feedback_selection_gate_20261015"
DEFAULT_OUTPUT = "reports/gensafenav_entity_binding_audit_20261016.json"


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def load_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(rows, field):
    result = {row[field]: row for row in rows}
    assert len(result) == len(rows), ("duplicate", field)
    return result


def analyze(run):
    issues = load_jsonl(run / "issues.jsonl")
    steps = load_jsonl(run / "physical_steps.jsonl")
    by_time = {(s["before"]["reset_index"], s["before"]["physical_index"]): s
               for s in steps}
    assert len(by_time) == len(steps) == len(issues)
    counts = Counter()
    issue_events, rejection_events = [], []
    lifetimes = set()
    minimum_center_gap = None
    counts["issue_frames"] = len(issues)
    counts["physical_steps"] = len(steps)
    sampled_positions = {}
    first_seen = {}
    for step in steps:
        for side in ("before", "after"):
            snap = step[side]
            for h in snap["humans"]:
                first_seen.setdefault(h["lifetime_id"], snap["physical_index"])
                key = (snap["reset_index"], snap["physical_index"], h["lifetime_id"])
                point = (snap["robot_position"], h["position"])
                assert key not in sampled_positions or sampled_positions[key] == point
                sampled_positions[key] = point
    nearest_key, nearest_points = min(sampled_positions.items(), key=lambda item: math.dist(*item[1]))
    minimum_center_distance = math.dist(*nearest_points)
    for issue in issues:
        key = (issue["reset_index"], issue["physical_index"])
        step = by_time.pop(key)
        before = step["before"]
        assert before["time"] == issue["time"]
        physical = keyed(before["humans"], "lifetime_id")
        logged = keyed(issue["humans"], "lifetime_id")
        assert set(physical) == set(logged)
        lifetimes.update(physical)
        rows = sorted(issue["humans"], key=lambda h: h["sorted_row"])
        assert [h["sorted_row"] for h in rows] == list(range(len(rows)))
        robot = before["robot_position"]
        distance = {lid: math.dist(h["position"], robot) for lid, h in physical.items()}
        center_order = sorted(physical, key=lambda lid: distance[lid])
        stored_order = [h["lifetime_id"] for h in rows]
        counts["all_entity_center_order_matches"] += center_order == stored_order
        assert center_order == stored_order, ("talk center order", key)
        gaps = [distance[b] - distance[a] for a, b in zip(center_order, center_order[1:])]
        if gaps:
            minimum_center_gap = min(gaps + ([] if minimum_center_gap is None else [minimum_center_gap]))
        visible = [lid for lid in center_order if physical[lid]["visibility_observation"]]
        counts["visible_entity_rows"] += len(visible)
        counts["invisible_entity_rows"] += len(rows) - len(visible)
        # Native DummyVecEnv's spatial Box is FP32. The recorder adds the real
        # robot position in FP64 to the delivered FP32 relative current point.
        expected = {}
        for lid in visible:
            h = physical[lid]
            xy = tuple(f32(h["position"][j] - robot[j]) + robot[j] for j in range(2))
            expected.setdefault(xy, []).append(lid)
        dummy = tuple(15.0 + robot[j] for j in range(2))
        owners, mismatches, rejected = [], [], []
        for row in rows:
            lid = row["lifetime_id"]
            actual = physical[lid]
            assert actual["human_id"] == row["human_id"]
            assert actual["issue_valid"] == row["issue_valid"]
            counts["visibility_fields_match"] += row["visible"] == actual["visibility_observation"]
            assert row["visible"] == actual["visibility_observation"], ("visibility", key, lid)
            prediction = row["forecast_world_xy"]
            assert len(prediction) == 6 and all(len(p) == 2 for p in prediction)
            assert all(math.isfinite(v) for p in prediction for v in p)
            matches = expected.get(tuple(prediction[0]), [])
            owner = matches[0] if len(matches) == 1 else None
            owners.append(owner)
            if owner is not None:
                counts["delivered_current_point_unique_exact_visible_owner"] += 1
            elif tuple(prediction[0]) == dummy:
                counts["delivered_current_point_exact_invisible_dummy"] += 1
            else:
                counts["delivered_current_point_unresolved"] += 1
            admission_distance = math.dist(prediction[0], actual["position"])
            assert (admission_distance < 0.1) == row["issue_valid"], ("native admission", key, lid)
            counts["native_admission_rule_matches"] += 1
            counts["accepted_entity_rows"] += row["issue_valid"]
            counts["rejected_entity_rows"] += not row["issue_valid"]
            counts["assigned_entity_gst_mask_true_rows"] += row["gst_out_mask"]
            if row["gst_out_mask"]:
                counts["assigned_entity_gst_mask_true_accepted"] += row["issue_valid"]
                counts["assigned_entity_gst_mask_true_rejected"] += not row["issue_valid"]
                if row["issue_valid"]:
                    counts["accepted_gst_full_future_exact_same_id"] += row["gst_delivered_relative_exact"]
                    counts["accepted_gst_full_future_not_exact_same_id"] += not row["gst_delivered_relative_exact"]
            elif row["issue_valid"]:
                counts["accepted_non_gst_dummy_rows"] += 1
            interval = row["native_interval"]
            assert (interval is not None) == row["issue_valid"]
            if interval is not None:
                assert len(interval) == 5 and all(0 <= r <= 1 for r in interval)
                counts["accepted_entity_interval_vectors"] += 1
            # This identifies which delivered row contains a current point of a
            # visible human. Its five future points are not reconstructed here.
            if owner is not None and owner != lid:
                detail = {"sorted_row": row["sorted_row"], "assigned_lifetime_id": lid,
                          "delivered_current_point_owner": owner,
                          "assigned_visible": row["visible"],
                          "delivered_owner_gst_mask": logged[owner]["gst_out_mask"],
                          "native_issue_valid": row["issue_valid"],
                          "admission_distance": admission_distance,
                          "native_interval_of_assigned_entity": interval}
                mismatches.append(detail)
                counts["visible_row_entity_mismatches"] += 1
                counts["accepted_visible_row_entity_mismatches"] += row["issue_valid"]
                counts["mismatched_visible_row_has_nonzero_interval"] += bool(interval and any(interval))
            if owner is not None and logged[owner]["gst_out_mask"]:
                counts["delivered_rows_with_gst_eligible_visible_owner"] += 1
                if not row["issue_valid"]:
                    counts["rejected_rows_with_gst_eligible_visible_owner"] += 1
                    rejected.append({"sorted_row": row["sorted_row"],
                                     "assigned_lifetime_id": lid, "visible_owner": owner,
                                     "owner_gst_mask": True, "same_identity": lid == owner,
                                     "admission_distance": admission_distance})
        # No guessed order is assigned to identical invisible (15,15) rows.
        actual_visible_prefix = owners[:len(visible)]
        prefix_exact = actual_visible_prefix == visible
        counts["visible_prefix_exact_expected_center_order_frames"] += prefix_exact
        counts["visible_prefix_contains_all_visible_entities_frames"] += (
            None not in actual_visible_prefix and set(actual_visible_prefix) == set(visible))
        prefix_mismatch = stored_order[:len(visible)] != actual_visible_prefix
        counts["visible_prefix_vs_all_entity_order_mismatch_frames"] += prefix_mismatch
        counts["frames_with_visible_row_entity_mismatches"] += bool(mismatches)
        if mismatches or not prefix_exact:
            new_lifetimes = [lid for lid in physical if first_seen[lid] == issue["physical_index"]
                             and issue["physical_index"] != 0]
            issue_events.append({"physical_index": issue["physical_index"], "time": issue["time"],
                                 "new_lifetimes_first_seen_at_this_frame": new_lifetimes,
                                 "all_entity_center_order": stored_order,
                                 "visible_center_order": visible,
                                 "delivered_current_point_owners": owners,
                                 "visible_prefix_expected_exact": prefix_exact,
                                 "row_mismatches": mismatches})
        if rejected:
            rejection_events.append({"physical_index": issue["physical_index"],
                                     "time": issue["time"], "rows": rejected})
    assert not by_time
    for name in ("delivered_current_point_unresolved", "visible_row_entity_mismatches",
                 "accepted_visible_row_entity_mismatches", "accepted_gst_full_future_not_exact_same_id",
                 "rejected_rows_with_gst_eligible_visible_owner", "mismatched_visible_row_has_nonzero_interval",
                 "visible_prefix_vs_all_entity_order_mismatch_frames"):
        counts.setdefault(name, 0)
    counts["unique_human_lifetimes"] = len(lifetimes)
    counts["new_lifetimes_after_initial_frame"] = sum(t != 0 for t in first_seen.values())
    counts["mismatch_frames_coincident_with_new_lifetime"] = sum(
        bool(e["new_lifetimes_first_seen_at_this_frame"]) for e in issue_events)
    observed = counts["visible_row_entity_mismatches"] > 0
    return {
        "status": "completed_two_log_offline_entity_binding_audit",
        "sources": [str(run / "issues.jsonl"), str(run / "physical_steps.jsonl")],
        "execution": {"model_calls": 0, "environment_steps": 0, "api_calls": 0,
                      "new_random_draws": 0, "new_trajectories": 0,
                      "input_policy": "Only the two listed JSONL files are read; source-code interpretations below are explicit annotations."},
        "method": {
            "population": "All recorded pre-action frames; no root, seed, or error-based selection.",
            "binding": "Join by reset_index/physical_index, then lifetime_id, checking human_id and visibility.",
            "ownership": "Exact FP32 relative-current-point plus FP64 real robot position; unique visible-owner match only, no tolerance or nearest-neighbor identity assignment.",
            "admission": "Recompute the native current-point Euclidean distance < .1 gate for each assigned human.",
            "invisible_rows": "Identical (15,15) placeholders cannot identify the hidden-row permutation; no such identity is claimed.",
            "minimum_distinct_center_order_gap": minimum_center_gap,
            "gst_scope": "gst_out_mask and gst_delivered_relative_exact are previously recorded checks, not freshly recomputed forecasts. A row whose current-point owner has GST out_mask=True is GST-eligible; this script does not reopen raw predictor NPZ files."
        },
        "counts": dict(sorted(counts.items())),
        "ordering_events": issue_events,
        "gst_eligible_rejection_events": rejection_events,
        "entity_replacement_boundary": {
            "new_lifetimes": [{"lifetime_id": lid, "first_physical_index": t}
                              for lid, t in first_seen.items() if t != 0],
            "observation": "Every observed mismatch frame coincides with a newly replaced Human lifetime; the leading intervening invisible assigned entity is that new object. Not every replacement causes a mismatch.",
            "source_sequence": "CrowdSimPred.step generates the observation before its end_goal_changing branch replaces reached-goal Human objects; talk2Env then sorts the current objects by their new actual positions. See crowd_sim_pred.py:205, :213-218. The existing observation can therefore retain the previous object's visibility/order context at that frame.",
            "causal_limit": "This coincidence plus the native sequence supports an observation/replacement/sort boundary diagnosis. It does not establish that heterogeneous physical radii caused these five events. No alternative-order replay or actor counterfactual was run."
        },
        "radius_and_actor_binding": {
            "physical_human_radius_identifiable": False,
            "physical_robot_radius_identifiable": False,
            "reason": "Neither JSONL records physical radii. ACI interval is an uncertainty radius, not the human body's radius.",
            "saved_config_source_facts": {
                "config": "vendor/GenSafeNav/trained_models/Ours_GST/configs/config.py",
                "env_randomize_attributes": True, "humans_random_radii": False,
                "robot_radius": 0.3,
                "human_radius_closed_envelope": [0.3, 0.5],
                "interpretation": "random_radii=False is the documented switch for changing radius on goal arrival, not initial randomization. generate_circle_crossing_human calls sample_random_attributes under randomize_attributes, whose radius draw is uniform(.3,.5). Human replacement in CrowdSimPred.step calls this same generator, so the same bounds apply to new lifetimes. Actual draws cannot be recovered from these logs.",
                "robot_initialization": "Agent.__init__ sets radius=config.robot.radius=.3; Robot inherits it. Active holonomic generate_robot_humans calls robot.set without a radius override; Agent.set changes radius only when supplied. No robot sample_random_attributes call occurs on this initialization path. This is source-supported, not a logged radius measurement."
            },
            "native_interval_binding": "The recorder saves each distance-sorted Human object's post-talk interval. Native talk constructs its returned conformity-score row at this same index. Exact actor input conformity_scores is not separately cached in the two logs, so its return/assignment is source-supported rather than an independent tensor equality check.",
            "invalid_interval": "native_interval=None corresponds in native source to a zero returned conformity row; that returned tensor is not independently logged.",
            "mismatch_implication": "All observed mismatched visible current-point rows were rejected and have native_interval=None. By native construction these actor-row locations receive zero conformity scores, rather than another entity's nonzero interval. The spatial future row remains present. Whether the resulting actor action changes is unmeasured.",
            "unused_actor_fields": "Ours_GST actor reads spatial_edges plus conformity_scores and detected_human_num. It does not read visible_masks/aggressiveness_factor. The latter is a robot-level scalar, not per-human metadata."
        },
        "sampled_physical_clearance_bound": {
            "unique_time_entity_position_pairs": len(sampled_positions),
            "minimum_sampled_center_distance": minimum_center_distance,
            "minimum_center_distance_at": {"reset_index": nearest_key[0], "physical_index": nearest_key[1],
                                            "lifetime_id": nearest_key[2]},
            "fixed_point3_both_radii_proxy": minimum_center_distance - 0.3 - 0.3,
            "true_minimum_sampled_surface_clearance_closed_bound": [minimum_center_distance - 0.3 - 0.5,
                                                                    minimum_center_distance - 0.3 - 0.3],
            "derivation": "At each recorded position, c=d-r_robot-r_human, r_robot=.3 by the active native initialization and .3<=r_human<=.5 by the generator. Taking the minimum preserves [min(d)-.8,min(d)-.6]. This is a conservative bound, not the unknown actual radius assignment.",
            "scope": "All distinct before/after sampled physical positions; no interpolation or swept-volume collision certificate. No human radii were inferred from behavior, RNG, or images."
        },
        "historical_clearance_correction": {
            "previous_reported_value": 0.44927675174873716,
            "previous_label": "minimum_sampled_surface_clearance",
            "correct_label": "fixed_point3_both_radii_proxy; an upper bound, not the actual clearance",
            "reason": "The old statement that both radii are fixed .3 incorrectly treats random_radii=False as disabling initial randomize_attributes=True.",
            "locations_for_parent_to_correct": [
                {"path": "reports/gensafenav_feedback_selection_decision_20261015.json", "json_path": "native_clean.minimum_sampled_surface_clearance / native_clean.clearance_scope", "observed_lines": [98, 99]},
                {"path": "reports/gensafenav_feedback_selection_review_20261015.json", "json_path": "physical_episode.minimum_sampled_surface_clearance / physical_episode.clearance_scope", "observed_lines": [71, 72]}
            ],
            "historical_reports_are_inputs": False,
            "changes_made_to_historical_reports": False
        },
        "recorded_episode_context": {
            "steps": len(steps), "terminal_done": steps[-1]["done"],
            "terminal_info_type": steps[-1]["info_type"],
            "scope": "Original realized trajectory only; no modified ordering, repair, or counterfactual action was executed. Case0/seed42 provenance comes from the fixed run, not a seed field in these two logs."
        },
        "decision": {
            "observed_current_point_entity_order_mismatch": observed,
            "qualification": ("Concrete ordering mismatch observed in the fixed trajectory; this does not establish changed decisions or harm."
                              if observed else "No concrete ordering mismatch observed in this fixed trajectory; static possibility is not an experimental harm signal."),
            "actual_safety_harm_established": False,
            "limitations": [
                "No actor counterfactual, action-change attribution, collision consequence, or repair effect was tested.",
                "Physical radii and the full actor input tensor are absent, so physical-radius binding/clearance cannot be independently established.",
                "Stable identity renaming with all associated state consistently permuted is distinct from changing the order of only one feature array.",
                "The row mismatch, if present, is a native data-binding observation; it is not evidence of an external attacker interface or a novel general security mechanism."
            ]
        },
        "source_code_anchors": [
            "scripts/probe_gensafenav_feedback_selection.py:80 (saved config), :159 (physical recorder), :252 (talk recorder)",
            "vendor/GenSafeNav/crowd_sim/envs/crowd_sim_var_num.py:143 (initial attributes), :275 (visible-only rows)",
            "vendor/GenSafeNav/crowd_sim/envs/utils/agent.py:44 (initial random physical radius)",
            "vendor/GenSafeNav/crowd_sim/envs/utils/agent.py:19, :52 (robot radius initialization/optional override)",
            "vendor/GenSafeNav/crowd_sim/envs/crowd_sim_var_num.py:119 (active robot reset without radius override)",
            "vendor/GenSafeNav/crowd_sim/envs/crowd_sim_pred.py:217 (new Human object uses the same generator)",
            "vendor/GenSafeNav/trained_models/Ours_GST/configs/config.py:48, :120, :135",
            "vendor/GenSafeNav/crowd_sim/envs/crowd_sim_pred_real_gst.py:52 (FP32 spatial observation), :82 (all-human ordering), :93 (admission), :105 (conformity rows)",
            "vendor/GenSafeNav/rl/vec_env/dummy_vec_env.py:25, :73 (FP32 observation buffer)",
            "vendor/GenSafeNav/rl/vec_env/vec_pretext_normalize.py:191 (whole-row distance sorting)",
            "vendor/GenSafeNav/rl/networks/selfAttn_srnn_temp_node.py:359, :387 (actual actor consumers)"
        ],
        "reproduction_command": "python scripts/analyze_gensafenav_entity_binding.py --verify-only"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = analyze(args.run)
    if args.verify_only:
        assert json.loads(args.output.read_text()) == result, "Saved report differs from recomputation"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"output": str(args.output), "verified_only": args.verify_only,
                      "counts": result["counts"], "decision": result["decision"]["qualification"]}, indent=2))


if __name__ == "__main__":
    main()
