#!/usr/bin/env python3
"""Audit local resources needed for an AlphaBrain Cosmos/LIBERO fine-tune.

This deliberately does not download data or start training.  It records which
released inference artifacts are present and whether a compatible training
dataset is available locally, so a missing full dynamics baseline is explicit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _files(root: Path, pattern: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            out.append({"path": str(path), "bytes": path.stat().st_size})
    return out


def audit(repo_root: Path) -> dict[str, Any]:
    model_dir = repo_root / "models" / "AlphaBrain" / "cosmos_policy_libero"
    base_artifacts = _files(model_dir, "*")

    env_root = os.environ.get("LIBERO_DATA_ROOT")
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            repo_root / "data" / "datasets" / "LIBERO-Cosmos-Policy",
            repo_root / "playground" / "Datasets" / "LIBERO-Cosmos-Policy",
            repo_root / "playground" / "Datasets" / "LEROBOT_LIBERO_DATA",
            repo_root / "data" / "LIBERO-Cosmos-Policy",
        ]
    )

    datasets = []
    for path in candidates:
        entry: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "success_only": (path / "success_only").is_dir(),
            "all_episodes": (path / "all_episodes").is_dir(),
            "hdf5_files": len(list(path.rglob("*.hdf5"))) if path.exists() else 0,
            "parquet_files": len(list(path.rglob("*.parquet"))) if path.exists() else 0,
        }
        entry["compatible_cosmos_policy_layout"] = bool(
            entry["success_only"] and entry["all_episodes"]
        )
        datasets.append(entry)

    probe_data = _files(repo_root / "data", "worldagen_goal_probe*/**/*.parquet")
    official_entry = repo_root / "vendor" / "AlphaBrain" / "configs" / "finetune_config.yaml"
    loader_entry = repo_root / "vendor" / "AlphaBrain" / "AlphaBrain" / "dataloader" / "cosmos_datasets.py"

    compatible = any(item["compatible_cosmos_policy_layout"] for item in datasets)
    base_present = any(item["path"].endswith("model.safetensors") for item in base_artifacts)
    return {
        "schema_version": 1,
        "repo_root": str(repo_root),
        "base_inference_artifacts": {
            "directory": str(model_dir),
            "present": base_present,
            "files": base_artifacts,
        },
        "training_dataset_search": {
            "LIBERO_DATA_ROOT": env_root,
            "candidates": datasets,
            "compatible_cosmos_policy_dataset_found": compatible,
            "required_layout": ["success_only/", "all_episodes/", "success_only/t5_embeddings.pkl", "success_only/dataset_statistics.json"],
        },
        "available_probe_data": {
            "files": probe_data,
            "compatible_with_cosmos_policy_loader": False,
            "reason": "WorldAgen probe parquet files do not provide the required LIBERO-Cosmos-Policy episode layout.",
        },
        "official_training_entrypoints": {
            "config": str(official_entry),
            "loader": str(loader_entry),
            "config_present": official_entry.is_file(),
            "loader_present": loader_entry.is_file(),
        },
        "conclusion": (
            "released inference artifacts are available, but no compatible local training dataset was found"
            if not compatible
            else "a compatible local training dataset was found"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.repo_root.resolve())
    output = args.output or (args.repo_root / "runs" / "alphabrain_training_resource_audit_v0" / "summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
