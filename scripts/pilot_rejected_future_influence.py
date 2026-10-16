"""Bounded decision-coupling probe with released MiniGrid forward/inverse models.

This is a researcher-composed one-step selector, NOT an ACID/WAV reproduction.
The adaptive spread ratio comes from ACID Eq. 5; categorical action confidence
comes from WAV's RLIR comparator. Only a Pareto-dominated forecast is replaced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from wav_minigrid.config import WM_ACTIVE_LEARNING
from wav_minigrid.evaluate_generation import MiniGridPhysicsOracle
from wav_minigrid.models import SparseIDM, WorldModel


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/wav_minigrid"
RUN = ROOT / "runs/rejected_future_influence_v1"
SEED = 20261009


def save(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def distance(frame, carried, goal_frame, goal_carried):
    # Half the squared distance between categorical one-hot encodings, with
    # the upstream carried-feature weight of 10. No geometry for object IDs.
    return ((frame != goal_frame).sum(axis=(-3, -2, -1))
            + 10 * (carried != goal_carried).sum(axis=-1)).astype(float)


def nondominated(g, a):
    dominates = ((g[:, None] <= g[None, :]) & (a[:, None] <= a[None, :])
                 & ((g[:, None] < g[None, :]) | (a[:, None] < a[None, :])))
    return ~dominates.any(axis=0)


def coefficient(g, a, pareto=False):
    mask = nondominated(g, a) if pareto else np.ones(len(g), dtype=bool)
    return float(np.std(g[mask]) / max(np.std(a[mask]), 1e-8))


def choose(g, a, weight):
    return int(np.argmin(g + weight * a))


def inverse_prob(model, frames, carried, next_frames, next_carried, seed):
    torch.manual_seed(seed)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(frames), 256):
            s = slice(start, start + 256)
            inputs = {
                "frame": torch.from_numpy(np.stack([frames[s], next_frames[s]])).float(),
                "carried_col": torch.from_numpy(np.stack([carried[s, :1], next_carried[s, :1]])).float(),
                "carried_obj": torch.from_numpy(np.stack([carried[s, 1:], next_carried[s, 1:]])).float(),
            }
            logits, _, _ = model(inputs)
            outputs.append(logits.softmax(-1).numpy())
    return np.concatenate(outputs)


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=False)
    wm_path = VENDOR / "checkpoints/afs_test_checkpoints/wm_WAV_afs_test.pth"
    inverse_path = Path(WM_ACTIVE_LEARNING["INVERSE_MODEL_PATH"])
    test_path = Path(WM_ACTIVE_LEARNING["TEST_SET_PATH"])
    donor_path = Path(WM_ACTIVE_LEARNING["DATA_PATH"])
    data = np.load(test_path)
    states = data["states"].reshape(-1, 6, 6, 3).astype(np.float32)
    carried = data["carried"].reshape(-1, 2).astype(np.float32)
    actions = data["actions"]
    oracle = MiniGridPhysicsOracle(box_toggle_mode="deposit")
    indices, goal_frames, goal_carried = [], [], []
    examined = 0
    # Eligibility uses only recorded inputs and the native physical oracle.
    for idx in np.random.default_rng(SEED).permutation(len(states)):
        examined += 1
        f, c0, c1 = oracle.step(states[idx], *carried[idx], actions[idx])
        c = np.array([c0, c1], dtype=np.float32)
        if np.array_equal(f, states[idx]) and np.array_equal(c, carried[idx]):
            continue
        indices.append(int(idx))
        goal_frames.append(f)
        goal_carried.append(c)
        if len(indices) == 128:
            break
    assert len(indices) == 128
    frames, carries = states[indices], carried[indices]
    goals, goal_carries = np.stack(goal_frames), np.stack(goal_carried)
    donors = np.load(donor_path)
    donor_indices = np.random.default_rng(SEED + 1).choice(len(donors["states"]), 256, replace=False)
    donor_frames = donors["next_states"].reshape(-1, 6, 6, 3)[donor_indices].astype(np.float32)
    donor_carries = donors["next_carried"].reshape(-1, 2)[donor_indices].astype(np.float32)
    plan = {
        "candidate": "Changing only a rejected forecast can steer other choices through batch-adaptive cost weights.",
        "scope": "Researcher-composed exhaustive one-step selector with released learned MiniGrid components; not native WAV acquisition, ACID reproduction, full task episodes, or a novelty claim.",
        "upstream_commit": subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [wm_path, inverse_path, test_path, donor_path]},
        "indices": indices, "donor_indices": donor_indices.tolist(),
        "goals": "Native oracle successor of each original recorded action; skip physical no-ops before querying learned models.",
        "candidates": "All seven actions, fixed across arms; no action proposal or ordering edit.",
        "clean_gate": "At least 80% exact one-step goal achievement by BOTH goal-only and adaptive selectors; otherwise stop before donor search.",
        "goal_cost": "Categorical mismatch count, carried channels weighted by 10.",
        "action_cost": "1 minus released SparseIDM probability of the conditioning action.",
        "weight": "lambda=1; std(goal_cost)/max(std(action_cost),1e-8), recalculated per candidate pool.",
        "edit": "Replace one originally Pareto-dominated prediction with a recorded donor next-state; replacement must also be Pareto-dominated by an unchanged candidate.",
        "attack_selection": "Maximize or minimize the resulting spread ratio among legal donors; no oracle outcome used for donor selection. Random legal donor comparator.",
        "controls": "Identical cached predictions for all untouched actions; restore clean coefficient after same edit; Pareto-front-only scale computation.",
        "idm_randomness": "Native stochastic Gumbel mask retained. Reset a fixed seed per full inference pass; freeze all untouched prediction/confidence outputs across edits.",
        "oracle_calls_for_goal_selection": examined,
        "training_updates": 0, "api_requests": 0, "gpu_use": False,
        "sources": ["https://arxiv.org/html/2607.02403v1#S3.SS3", "https://github.com/world-action-verifier/wav_minigrid"],
    }
    save("plan.json", plan)
    wm = WorldModel((6, 6, 3), num_actions=7)
    checkpoint = torch.load(wm_path, map_location="cpu", weights_only=True)
    wm.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    inv = SparseIDM(6, 6, num_actions=7)
    inv.load_state_dict(torch.load(inverse_path, map_location="cpu", weights_only=True), strict=True)
    wm.eval().requires_grad_(False)
    inv.eval().requires_grad_(False)
    hashes = [digest(wm), digest(inv)]
    flat_frames, flat_carries = np.repeat(frames, 7, axis=0), np.repeat(carries, 7, axis=0)
    candidate_actions = np.tile(np.arange(7), len(frames))
    inputs = {"frame": torch.from_numpy(flat_frames), "carried_col": torch.from_numpy(flat_carries[:, :1]),
              "carried_obj": torch.from_numpy(flat_carries[:, 1:])}
    with torch.inference_mode():
        out = wm(inputs, mode="predict_with_action", gt_actions=torch.from_numpy(candidate_actions))
        pred_frames = torch.stack([out["logits_obj"].argmax(1), out["logits_col"].argmax(1),
                                   out["logits_state"].argmax(1)], dim=-1).float().numpy()
        pred_carries = torch.cat([out["carried_col"].round(), out["carried_obj"].round()], dim=-1).numpy()
    probs = inverse_prob(inv, flat_frames, flat_carries, pred_frames, pred_carries, SEED + 2)
    repeat_probs = inverse_prob(inv, flat_frames, flat_carries, pred_frames, pred_carries, SEED + 2)
    assert np.array_equal(probs, repeat_probs)
    g = distance(pred_frames, pred_carries, np.repeat(goals, 7, axis=0), np.repeat(goal_carries, 7, axis=0)).reshape(-1, 7)
    a = (1 - probs[np.arange(len(probs)), candidate_actions]).reshape(-1, 7).astype(float)
    actual_costs = []
    for frame, carry, goal, gc in zip(frames, carries, goals, goal_carries):
        row = []
        for action in range(7):
            nf, nc0, nc1 = oracle.step(frame, *carry, action)
            row.append(float(distance(nf, np.array([nc0, nc1]), goal, gc)))
        actual_costs.append(row)
    actual = np.asarray(actual_costs)
    weights = np.array([coefficient(gi, ai) for gi, ai in zip(g, a)])
    clean = np.array([choose(gi, ai, w) for gi, ai, w in zip(g, a, weights)])
    goal_only = g.argmin(-1)
    pareto_weights = np.array([coefficient(gi, ai, True) for gi, ai in zip(g, a)])
    pareto = np.array([choose(gi, ai, w) for gi, ai, w in zip(g, a, pareto_weights)])
    def metrics(selected):
        costs = actual[np.arange(len(frames)), selected]
        return {"successes": int((costs == 0).sum()), "instances": len(frames),
                "success_rate": float((costs == 0).mean()), "mean_actual_goal_cost": float(costs.mean())}
    result = {"clean": metrics(clean), "goal_only": metrics(goal_only), "pareto_scale": metrics(pareto),
              "model_hashes": hashes, "native_world_model_forward_calls": 1,
              "native_inverse_evaluated_pairs": 2 * len(probs), "training_updates": 0,
              "environment_oracle_calls": examined + len(frames) * 7, "api_requests": 0, "gpu_use": False,
              "exact_paired_inverse_repeat": True, "scope": plan["scope"]}
    np.savez_compressed(RUN / "clean_predictions.npz", indices=indices, frames=frames, carried=carries,
                        goals=goals, goal_carried=goal_carries, pred_frames=pred_frames, pred_carried=pred_carries,
                        probabilities=probs, goal_cost=g, action_cost=a, actual_cost=actual, weights=weights,
                        clean_selected=clean, goal_only_selected=goal_only, pareto_selected=pareto)
    gate = min(result["clean"]["success_rate"], result["goal_only"]["success_rate"]) >= .8
    if not gate:
        result.update(status="clean_competence_gate_failed", donor_search_performed=False,
                      interpretation="This component composition is not a competent enough clean controller on the fixed gate. Do not tune or scale this construction, or count a later loss as a supported attack.")
    else:
        rows = []
        rng = np.random.default_rng(SEED + 3)
        for i, (gi, ai) in enumerate(zip(g, a)):
            dominated = np.flatnonzero(~nondominated(gi, ai))
            if not len(dominated):
                rows.append({"index": indices[i], "eligible": False})
                continue
            edited = int(dominated[np.argmax((gi + weights[i] * ai)[dominated])])
            donor_prob = inverse_prob(inv, np.repeat(frames[i:i+1], len(donor_frames), axis=0),
                                     np.repeat(carries[i:i+1], len(donor_frames), axis=0),
                                     donor_frames, donor_carries, SEED + 100 + i)
            result["native_inverse_evaluated_pairs"] += len(donor_frames)
            dg = distance(donor_frames, donor_carries, goals[i], goal_carries[i])
            da = 1 - donor_prob[:, edited].astype(float)
            possibilities = []
            for j in range(len(dg)):
                gj, aj = gi.copy(), ai.copy()
                gj[edited], aj[edited] = dg[j], da[j]
                if nondominated(gj, aj)[edited]:
                    continue
                w = coefficient(gj, aj)
                selected = choose(gj, aj, w)
                restored = choose(gj, aj, weights[i])
                repaired = choose(gj, aj, coefficient(gj, aj, True))
                assert selected != edited and restored == clean[i] and repaired == pareto[i]
                possibilities.append({"donor_local_index": j, "donor_index": int(donor_indices[j]),
                                      "weight": w, "selected": selected,
                                      "goal_cost": float(actual[i, selected]),
                                      "restored_selected": restored, "pareto_selected": repaired})
            if not possibilities:
                rows.append({"index": indices[i], "eligible": False})
                continue
            up = max(possibilities, key=lambda x: x["weight"])
            down = min(possibilities, key=lambda x: x["weight"])
            rand = possibilities[int(rng.integers(len(possibilities)))]
            rows.append({"index": indices[i], "eligible": True, "edited_action": edited,
                         "clean_selected": int(clean[i]), "clean_weight": weights[i],
                         "clean_goal_cost": float(actual[i, clean[i]]), "legal_donors": len(possibilities),
                         "maximize_weight": up, "minimize_weight": down, "random_donor": rand})
        save("interventions.json", rows)
        arms = {}
        for name in ["maximize_weight", "minimize_weight", "random_donor"]:
            choices = np.array([r[name]["selected"] if r["eligible"] else int(clean[i]) for i, r in enumerate(rows)])
            clean_cost = actual[np.arange(len(frames)), clean]
            new_cost = actual[np.arange(len(frames)), choices]
            arms[name] = {**metrics(choices), "choices_changed": int((choices != clean).sum()),
                          "cost_worsened": int((new_cost > clean_cost).sum()),
                          "cost_improved": int((new_cost < clean_cost).sum()),
                          "clean_success_to_failure": int(((clean_cost == 0) & (new_cost > 0)).sum()),
                          "failure_to_success": int(((clean_cost > 0) & (new_cost == 0)).sum())}
        result.update(status="bounded_component_screen_completed", donor_search_performed=True,
                      eligible_instances=sum(r["eligible"] for r in rows), arms=arms,
                      coefficient_restoration_preserves_every_clean_choice=True,
                      pareto_scale_preserves_every_own_clean_choice=True,
                      interpretation="Only a dominated, never-selected forecast is edited. Any change among untouched actions is mediated by the pool coefficient. This is a component-composition feasibility result, not a native full-task attack or demonstrated novelty.")
    assert [digest(wm), digest(inv)] == hashes
    result.update(models_unchanged=True, elapsed_s=time.monotonic() - start)
    save("summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
