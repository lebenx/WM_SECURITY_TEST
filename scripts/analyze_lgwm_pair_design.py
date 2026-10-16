"""Audit the official LGWM test manifest as a context/outcome identity graph.

No images, learned models, APIs, or training data are loaded. The optional
forest potentials are diagnostic TEST-label lookup constructions, never a
generalization baseline or a prediction from the released harm head.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "runs/lgwm_resource_qualification_20261016/source/benchmark/rswt_bench_test.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/lgwm_pair_design_audit_20261016.json"
EXPECTED_SHA256 = "5f22f4848b5221970142babb7abc572b68cb1119f770266f787a786817dcb707"
COMMIT = "4e760ed5f44f645e1bf81cdb171b01056dc30f3a"
CLASSES = {
    1: "1_hijack", 2: "2_paired_safe", 3: "3_legit_cred",
    4: "4_surprising", 5: "5_benign_misplace",
}


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def graph_audit(records, positive_classes):
    """Exact integer graph statistics; duplicate records are not parallel edges."""
    edge_records = defaultdict(list)
    for record in records:
        edge = (record["base"]["transition_id"], record["obs"]["transition_id"])
        edge_records[edge].append(record)
    adjacency = defaultdict(list)
    context_neighbors = defaultdict(set)
    context_labels, outcome_labels = defaultdict(set), defaultdict(set)
    conflicts = []
    edge_targets = {}
    for (context, outcome), rows in sorted(edge_records.items()):
        labels = {1 if row["cls"] in positive_classes else -1 for row in rows}
        if len(labels) != 1:
            conflicts.append({"base": context, "obs": outcome,
                              "sids": [row["sid"] for row in rows]})
        target = next(iter(labels)) if len(labels) == 1 else None
        edge_targets[(context, outcome)] = target
        cnode, onode = ("context", context), ("outcome", outcome)
        adjacency[cnode].append((onode, target))
        adjacency[onode].append((cnode, target))
        context_neighbors[context].add(outcome)
        context_labels[context].update(labels)
        outcome_labels[outcome].update(labels)

    visited, components = set(), []
    for first in sorted(adjacency):
        if first in visited:
            continue
        queue, vertices, degrees = deque([first]), 0, 0
        visited.add(first)
        while queue:
            node = queue.popleft()
            vertices += 1
            degrees += len(adjacency[node])
            for neighbor, _ in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append({"vertices": vertices, "edges": degrees // 2})
    cycle_rank = len(edge_records) - len(adjacency) + len(components)

    complete_2x2 = checkerboards = 0
    contexts = sorted(context_neighbors)
    for i, context in enumerate(contexts):
        for other in contexts[i + 1:]:
            common = sorted(context_neighbors[context] & context_neighbors[other])
            complete_2x2 += len(common) * (len(common) - 1) // 2
            for j, outcome in enumerate(common):
                for second in common[j + 1:]:
                    labels = [edge_targets[(c, o)]
                              for c, o in ((context, outcome), (context, second),
                                           (other, outcome), (other, second))]
                    if None not in labels and labels[0] == labels[3] and labels[1] == labels[2] and labels[0] != labels[1]:
                        checkerboards += 1

    construction = {
        "scope": "Diagnostic test-label identity lookup; not train/test generalization, a deployable classifier, or the released model's learned readout.",
        "labels": "Positive edge target +1; negative edge target -1.",
        "algorithm": "If the simple bipartite graph is a forest and duplicate-edge labels agree, anchor the lexicographically first vertex of each component at zero. Traverse its tree; assign unseen neighbor potential = edge target - current potential. Each edge then satisfies u(context)+v(outcome)=target.",
        "executed": cycle_rank == 0 and not conflicts,
    }
    if construction["executed"]:
        potential = {}
        for first in sorted(adjacency):
            if first in potential:
                continue
            potential[first] = 0
            queue = deque([first])
            while queue:
                node = queue.popleft()
                for neighbor, target in sorted(adjacency[node]):
                    proposed = target - potential[node]
                    if neighbor not in potential:
                        potential[neighbor] = proposed
                        queue.append(neighbor)
                    elif potential[neighbor] != proposed:
                        raise AssertionError("Forest potential construction is inconsistent")
        residuals = [potential[("context", c)] + potential[("outcome", o)] - target
                     for (c, o), target in edge_targets.items()]
        correct = sum((potential[("context", row["base"]["transition_id"])]
                       + potential[("outcome", row["obs"]["transition_id"])] > 0)
                      == (row["cls"] in positive_classes) for row in records)
        serialized = json.dumps([[*node, potential[node]] for node in sorted(potential)],
                                ensure_ascii=False, separators=(",", ":"))
        construction.update(
            unique_edges_verified=len(residuals), max_absolute_edge_residual=max(map(abs, residuals), default=0),
            diagnostic_test_records_correct=correct, diagnostic_test_records_total=len(records),
            potential_min=min(potential.values(), default=0), potential_max=max(potential.values(), default=0),
            deterministic_potentials_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
        )
    return {
        "records": len(records), "unique_edges": len(edge_records),
        "duplicate_record_count": len(records) - len(edge_records),
        "context_vertices": len(context_neighbors), "outcome_vertices": len(outcome_labels),
        "vertices": len(adjacency), "connected_components": len(components),
        "cycle_rank": cycle_rank, "is_forest": cycle_rank == 0,
        "complete_2x2_submatrices": complete_2x2, "binary_checkerboards": checkerboards,
        "context_vertices_with_both_labels": sum(len(v) == 2 for v in context_labels.values()),
        "outcome_vertices_with_both_labels": sum(len(v) == 2 for v in outcome_labels.values()),
        "conflicting_binary_label_edges": conflicts,
        "forest_additive_test_label_construction": construction,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = args.manifest.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Expected the pinned official test manifest; no replacement panel allowed")
    records = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if len({r["sid"] for r in records}) != len(records):
        raise ValueError("Official sample identities must be unique")
    comparisons = {}
    aliases = {(1, 3): ["RSWT"], (1, 5): ["Harm_AUC"],
               (5, 2): ["Swap"], (5, 3): ["Cred"]}
    for positive, negative in [(1, 2), (1, 3), (1, 4), (1, 5), (5, 2), (5, 3), (5, 4)]:
        names = {CLASSES[positive], CLASSES[negative]}
        rows = [r for r in records if r["cls"] in names]
        comparisons[f"{positive}v{negative}"] = {
            "native_aliases": aliases.get((positive, negative), []),
            "positive_class": CLASSES[positive], "negative_class": CLASSES[negative],
            **graph_audit(rows, {CLASSES[positive]}),
        }
    source_base = f"https://github.com/jiamingzhang94/lgwm/blob/{COMMIT}/"
    report = {
        "status": "completed_source_and_official_test_manifest_design_audit",
        "manifest": relative(args.manifest), "manifest_sha256": digest,
        "script": relative(Path(__file__)), "upstream_commit": COMMIT,
        "identity_definition": "Context node = base.transition_id; outcome node = obs.transition_id. Roles are separate even when the same ID appears on both sides. This does not establish unique pixels, action inputs, or actual UI states across IDs.",
        "total_graph": graph_audit(records, {CLASSES[1]}),
        "native_pairwise_metrics": comparisons,
        "native_aggregate_metric_definitions": {
            "Integrity_mAUC": ["1v2", "1v3", "1v4", "5v2", "5v3", "5v4"],
            "Hijack_mAUC": ["1v2", "1v3", "1v4", "1v5"],
            "note": "These average pairwise AUCs; graph statistics are reported for their actual component comparisons, without inventing a new pooled binary target.",
        },
        "verified_native_head": {
            "feature_formula": "concat(unit(mean observed_j)-unit(mean predicted_i), mean valid action tokens_i)",
            "head": "Fixed StandardScaler followed by LogisticRegression; no absolute residual or joint residual L2 normalization.",
            "logit_formula": "L_ij = beta_o^T o_j - beta_o^T p_i + beta_a^T a_i + c = u_i + v_j",
            "mixed_contrast": "L_11 - L_12 - L_21 + L_22 = 0 in exact arithmetic; numerical rounding may leave small residuals. Sigmoid probabilities need not have zero mixed contrast.",
            "interpretation": "Outcome score ordering cannot reverse with context. A complete checkerboard cannot be classified correctly at all four cells by a common threshold.",
            "source_paths": ["runs/lgwm_readout_source_20261016/src/lgwm/inference.py",
                             "runs/lgwm_readout_source_20261016/src/lgwm/features.py",
                             "runs/lgwm_readout_source_20261016/src/lgwm/evaluation.py"],
            "primary_sources": [source_base + "src/lgwm/inference.py#L40-L51",
                                source_base + "src/lgwm/evaluation.py#L71-L88"],
        },
        "integrity_exception": {
            "formula": "integrity_ij = 1 - dot(p_i, o_j)",
            "scope": "This score has a context-outcome interaction and is not subject to the additive harm-head limitation. No full-verifier bypass follows.",
        },
        "conclusions": [
            "At official transition-ID resolution the test graph is a forest, with no complete 2x2 panel. The diagnostic additive lookup construction fits its binary labels exactly.",
            "Donor pairing excludes outcome-only detection for RSWT but does not by itself identify context-outcome interactions over additive effects.",
            "The construction uses test labels and unrestricted ID potentials. It neither reproduces the learned head nor measures held-out generalization, so its exact fit is not a performance baseline.",
            "The linear-head expressivity fact is elementary; the source-grounded benchmark qualification is useful but is not evidence of a new security mechanism or actual agent harm.",
            "A model pilot needs independently grounded complete relational comparisons and a concrete consumer consequence; downloading the WM merely to verify the algebra is unnecessary.",
        ],
        "limitations": [
            "Different transition IDs may contain identical images or actions. Image deduplication and actual input equivalence were not inspected.",
            "No missing edge has been assigned a safe/harmful label. A structurally completed matrix does not supply safety ground truth.",
            "The native score CLI reports scores; this audit does not establish a downstream accept/block policy or physical consequence.",
        ],
        "resource_use_for_this_audit": {"model_calls": 0, "api_calls": 0, "gpu_jobs": 0,
                                        "environment_steps": 0, "weight_downloads": 0,
                                        "training_runs": 0, "new_downloads": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    graph = report["total_graph"]
    print(json.dumps({"report": relative(args.output), **{k: graph[k] for k in
                     ("records", "unique_edges", "vertices", "connected_components",
                      "cycle_rank", "complete_2x2_submatrices")}}))


if __name__ == "__main__":
    main()
