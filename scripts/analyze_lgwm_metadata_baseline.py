"""Fixed train-only categorical baselines; no WM, API or environment execution."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/lgwm_resource_qualification_20261016/source"
PROTOCOL = ROOT / "configs/lgwm_metadata_baseline_20261016.json"
OUTPUT = ROOT / "runs/lgwm_metadata_baseline_20261016"


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def auc(positive, negative):
    # Mid-rank definition; exact treatment of tied categorical predictions.
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def features(row, names):
    return tuple(min(9, max(0, math.floor(10 * row["step_frac"]))) if n == "step_decile" else row[n] for n in names)


def main():
    start = time.monotonic()
    config = json.loads(PROTOCOL.read_text())
    train, test = (records(SOURCE / config[k]) for k in ("training_file", "test_file"))
    assert len(train) == 6786 and len(test) == 966
    OUTPUT.mkdir(exist_ok=False)
    positive = sum(r["cls"] == "1_hijack" for r in train)
    prior = (positive + 1) / (len(train) + 2)
    rows, scores = [], []
    for names in config["baseline_features"]:
        counts = defaultdict(lambda: [0, 0])
        for r in train:
            counts[features(r, names)][r["cls"] == "1_hijack"] += 1
        predictions = []
        for r in test:
            cell = counts.get(features(r, names))
            predictions.append((cell[1] + 1) / (sum(cell) + 2) if cell else prior)
        by_class = {cls: [p for r, p in zip(test, predictions) if r["cls"] == cls] for cls in {r["cls"] for r in test}}
        donor = defaultdict(dict)
        for r, p in zip(test, predictions):
            if r["cls"] in ["1_hijack", "3_legit_cred"]:
                donor[r["donor_id"]][r["cls"]] = p
        assert len(donor) == 222 and all(len(v) == 2 for v in donor.values())
        wins = sum((v["1_hijack"] > v["3_legit_cred"]) + 0.5 * (v["1_hijack"] == v["3_legit_cred"]) for v in donor.values())
        result = {"features": names, "trained_cells": len(counts),
                  "unseen_test_cells_rows": sum(features(r, names) not in counts for r in test),
                  "RSWT_1v3": auc(by_class["1_hijack"], by_class["3_legit_cred"]),
                  "Harm_1v5": auc(by_class["1_hijack"], by_class["5_benign_misplace"]),
                  "donor_pair_order_accuracy": wins / len(donor)}
        result["followup_gate"] = max(result["RSWT_1v3"], result["Harm_1v5"]) >= 0.90
        rows.append(result)
        scores.extend({"features": names, "sid": r["sid"], "score": p} for r, p in zip(test, predictions))
    overlap = {}
    for name in ["base", "obs", "clean"]:
        train_ids = {r[name]["transition_id"] for r in train}
        test_ids = {r[name]["transition_id"] for r in test}
        overlap[name] = {"transition_ids": len(train_ids & test_ids),
                         "episode_prefixes": len({s.rsplit("/", 1)[0] for s in train_ids} & {s.rsplit("/", 1)[0] for s in test_ids})}
    all_train = {r[k]["transition_id"] for r in train for k in ["base", "obs", "clean"]}
    all_test = {r[k]["transition_id"] for r in test for k in ["base", "obs", "clean"]}
    overlap["across_all_roles"] = {"transition_ids": len(all_train & all_test),
                                    "episode_prefixes": len({s.rsplit("/", 1)[0] for s in all_train} & {s.rsplit("/", 1)[0] for s in all_test})}
    report = {"status": "completed", "protocol": str(PROTOCOL.relative_to(ROOT)),
              "scope": "Train-only low-dimensional metadata baseline on a published fixed benchmark; no WM, attack or task-outcome experiment.",
              "protocol_timing": "Frozen before training-manifest download/fitting; test manifest source and structure had already been inspected. This is a development screening analysis.",
              "train_records": len(train), "test_records": len(test),
              "training_class_counts": dict(Counter(r["cls"] for r in train)),
              "test_class_counts": dict(Counter(r["cls"] for r in test)),
              "overlap": overlap, "results": rows,
              "any_followup_gate": any(r["followup_gate"] for r in rows),
              "limits": config["limits"], "elapsed_seconds": time.monotonic() - start,
              "activity": {"model_api_calls": 0, "pretrained_model_loads": 0, "world_model_forwards": 0,
                           "environment_steps": 0, "gpu_jobs": 0, "categorical_estimators_fitted": len(rows)}}
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (OUTPUT / "scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
    print(json.dumps({"results": rows, "any_followup_gate": report["any_followup_gate"], "overlap": overlap}))


if __name__ == "__main__":
    main()
