#!/usr/bin/env python3
"""Compare an empty Goal with the Clean Goal on the same physical scene.

This is a forecast/target comparator only.  It reuses the same three shared
initial states and four fixed actions as the same-scene crossing, but does not
rerun every legal Goal replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pilot_worldagen_counterfactual_action_ranking as r
from pilot_worldagen_same_scene_goal import make_shared_env, physical_record, scene_contract, sha
from pilot_worldagen_reproducible_state_target import restore_rng, snapshot_rng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-indices", default="0,1,2")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    suite = r.libero.benchmark.get_benchmark_dict()["libero_goal"]()
    tasks = sorted([suite.get_task(i) for i in range(suite.get_num_tasks())], key=lambda t: t.name)
    clean_task = next(t for t in tasks if t.name == "put_the_bowl_on_the_plate")
    contracts = [scene_contract(t) for t in tasks]
    if len(set(contracts)) != 1:
        raise RuntimeError("Goal suite physical BDDL definitions differ")
    init_file = r.libero.LIBERO_ROOT / "libero/libero/init_files" / clean_task.problem_folder / clean_task.init_states_file
    initial_states = torch.load(init_file, map_location="cpu", weights_only=False)
    actions = torch.zeros(4, 6, 7)
    actions[1, :, 0] = 0.25
    actions[2, :, 0] = -0.25
    actions[3, :, 1] = 0.25
    protocol = {
        "status": "registered_before_inference",
        "seed": args.seed,
        "initial_indices": [int(i) for i in args.init_indices.split(",")],
        "clean_goal": clean_task.language,
        "empty_goal": "",
        "action_names": ["zero", "positive_x", "negative_x", "positive_y"],
        "actions_sha256": sha(actions.numpy()),
        "forecast_horizon": 3,
        "model_weights_frozen": True,
        "api_requests": 0,
        "training_updates": 0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Same physical scene and fixed actions; empty instruction is a prompt-independent diagnostic, not a trained goal-free dynamics head.",
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    model = r.probe.build_model()
    native, cut = r.libero.masks(model)
    rows = []
    for init_idx in protocol["initial_indices"]:
        rng = snapshot_rng()
        restore_rng(rng)
        env, obs = make_shared_env(clean_task, initial_states[init_idx], args.seed)
        try:
            initial = physical_record(env, obs)
            preds = {}
            for label, text in [("clean", clean_task.language), ("empty", "")]:
                batch = r.batch_from_observation(model, obs, text, actions)
                raw, _ = r.probe.infer(model, batch, native)
                preds[label] = r.primary_forecasts(raw, 4)[:, 0].cpu().float()
            targets = []
            for action in actions:
                restore_rng(rng)
                live_env, live_obs = make_shared_env(clean_task, initial_states[init_idx], args.seed)
                try:
                    if physical_record(live_env, live_obs) != initial:
                        raise RuntimeError(f"initial replay mismatch init={init_idx}")
                    for a in action[:3]:
                        live_obs, *_ = live_env.step(a.numpy().astype(np.float32))
                    image = Image.fromarray(live_obs["agentview_image"]).convert("RGB").transpose(Image.Transpose.ROTATE_180)
                    targets.append(r.patchify(model.image_processor(image).unsqueeze(0))[0])
                finally:
                    live_env.close()
            target = torch.stack(targets)
        finally:
            env.close()
        clean_mse = (preds["clean"] - target).square().mean((1, 2)).numpy()
        empty_mse = (preds["empty"] - target).square().mean((1, 2)).numpy()
        row = {
            "init_index": init_idx,
            "clean_target_mse": clean_mse.tolist(),
            "empty_target_mse": empty_mse.tolist(),
            "empty_minus_clean_target_mse": (empty_mse - clean_mse).tolist(),
            "clean_empty_forecast_mse": (preds["empty"] - preds["clean"]).square().mean((1, 2)).numpy().tolist(),
        }
        rows.append(row)
        (args.output_dir / "summary.json").write_text(json.dumps({**protocol, "status": "running", "rows": rows}, indent=2) + "\n")
        print(json.dumps({"init": init_idx, "empty_minus_clean_mean": float(np.mean(row["empty_minus_clean_target_mse"]))}), flush=True)
    deltas = np.asarray([r["empty_minus_clean_target_mse"] for r in rows], dtype=float)
    state_means = deltas.mean(axis=1)
    rng = np.random.default_rng(0)
    boot = np.asarray([
        rng.choice(state_means, size=len(state_means), replace=True).mean()
        for _ in range(10000)
    ])
    result = {**protocol, "status": "completed_same_scene_empty_goal", "rows": rows,
              "empty_minus_clean_target_mse_mean": float(deltas.mean()),
              "empty_minus_clean_target_mse_by_state": state_means.tolist(),
              "empty_minus_clean_target_mse_state_cluster_bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
              "empty_minus_clean_target_mse_positive_state_fraction": float(np.mean(state_means > 0)),
              "initial_replay_exact_count": len(rows) * len(actions),
              "interpretation": "The empty Goal is a prompt-independent diagnostic. Positive target-MSE deltas mean Clean is a better comparator on average; mixed deltas mean the comparison is not a universal fidelity result."}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
