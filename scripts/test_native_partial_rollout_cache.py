#!/usr/bin/env python3
"""R002: analyze paired native deadline outcomes only when genuine caches exist.

No inference, generated outcomes, concurrent load or network requests. Missing
native evidence is a resource gap, never an experimentally demonstrated null.
Input schema: system; native_evidence containing source paths for service,
deadline, partial_return and partial_score_consumer; roots containing id,
candidate_utilities (candidate_id -> actual measured scalar utility),
utility_trace_paths, higher_is_better, and arms clean/loaded_native/
loaded_complete_only (candidate_id, completed, load_schedule_id).
Both loaded arms must replay the same precommitted load schedule.
"""
import argparse
import json
import math
from pathlib import Path


def analyze(doc, base):
    evidence = doc.get("native_evidence", {})
    required = ("service", "deadline", "partial_return", "partial_score_consumer")
    missing = [name for name in required if not evidence.get(name)]
    if missing:
        return {"status": "resource_gap_not_an_experiment", "missing": missing,
                "actual_utility_tested": False, "security_claim_supported": False}
    # A supplied source reference is not automatically proof of its semantics.
    # Require independent review before admitting a dataset as a native test.
    if doc.get("native_contract_review_status") != "confirmed_from_native_trace_and_source":
        raise ValueError("Native trace/source contract has not been confirmed")
    for name in required:
        if not (base / evidence[name]).is_file():
            raise ValueError(f"Missing native evidence file: {name}")
    roots = doc.get("roots", [])
    if not roots:
        return {"status": "resource_gap_no_paired_real_utilities",
                "actual_utility_tested": False, "security_claim_supported": False}
    if len({r["id"] for r in roots}) != len(roots):
        raise ValueError("Duplicate physical-root identifier")
    rows = []
    for r in roots:
        values = r["candidate_utilities"]
        if not values or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()):
            raise ValueError("Actual finite candidate utilities are required")
        if set(r["utility_trace_paths"]) != set(values):
            raise ValueError("Every candidate needs its actual outcome trace")
        for path in r["utility_trace_paths"].values():
            if not (base / path).is_file():
                raise ValueError("Missing actual environment outcome trace")
        arms = r["arms"]
        clean, loaded, guarded = [arms[k] for k in ("clean", "loaded_native", "loaded_complete_only")]
        if not clean["completed"] or not guarded["completed"]:
            raise ValueError("Clean and completion-aware selected results must be complete")
        if not loaded["load_schedule_id"] or loaded["load_schedule_id"] != guarded["load_schedule_id"]:
            raise ValueError("Different load schedules confound the comparison")
        selected = [values[a["candidate_id"]] for a in (clean, loaded, guarded)]
        sign = 1 if r["higher_is_better"] else -1
        rows.append({"root": r["id"], "candidate_span": max(values.values()) - min(values.values()),
                     "clean_utility": selected[0], "loaded_native_utility": selected[1],
                     "loaded_complete_only_utility": selected[2],
                     "native_harm": sign * (selected[0] - selected[1]),
                     "completion_control_recovery": sign * (selected[2] - selected[1]),
                     "native_selected_partial": not loaded["completed"]})
    signal = any(r["native_selected_partial"] and r["native_harm"] > 0 and
                 r["completion_control_recovery"] > 0 for r in rows)
    return {"status": "completed_descriptive_native_cache_test", "system": doc["system"],
            "actual_utility_tested": True, "rows": rows, "conditional_signal": signal,
            "security_claim_supported": False,
            "interpretation": "A signal warrants an independent controlled replication; source references, descriptive recovery and a single system do not establish novelty or cross-system causality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a fresh output path; prior negatives are preserved")
    doc = json.loads(args.input.read_text())
    result = analyze(doc, Path(__file__).resolve().parents[1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
