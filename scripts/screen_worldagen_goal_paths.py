"""Evaluate native WorldAgen mask reachability, without weights or training.

An allowed path proves lack of architectural noninterference, not that a
trained model uses the path or that any task fails. The exact upstream mask
function is extracted with AST to avoid loading unrelated model dependencies.
"""
from __future__ import annotations

import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/WorldAgen/models/model.py"
RUN = ROOT / "runs/worldagen_goal_paths_v0"


def shortest_path(allowed, sources, targets, max_depth):
    q = deque((s, [s]) for s in sources)
    seen = set(sources)
    targets = set(targets)
    while q:
        current, path = q.popleft()
        if current in targets:
            return path
        if len(path)-1 == max_depth:
            continue
        for dest in torch.where(allowed[:, current])[0].tolist():
            if dest not in seen:
                seen.add(dest)
                q.append((dest, path+[dest]))
    return None


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=False)
    source = UPSTREAM.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "generate_attention_mask")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(UPSTREAM), "exec"), namespace)
    build_mask = namespace["generate_attention_mask"]
    plan = {
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM.parent.parent), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_model_sha256": hashlib.sha256(UPSTREAM.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_lines": [node.lineno, node.end_lineno],
        "scopes": ["Released LIBERO evaluation configuration", "One-chunk structural control"],
        "depth_limit": 24,
        "intervention": "Mask input-action query rows from all instruction key columns, preserving policy output access to instructions.",
        "fixed_inputs": "All action, state and image input tokens fixed; only instruction tokens are considered a source of influence.",
        "not_tested": ["pretrained weights", "decoded images", "real task decisions", "task returns", "attack optimization"],
    }
    (RUN/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    rows = []
    # Native LIBERO settings: 2 chunks, 1 state, 3 actions, 1 image per view,
    # 6 resampler queries, 9 prediction queries per image, 1 action pred step.
    for chunks in [1, 2]:
        num_a, num_b = 1+1+3+6*2+2, 9*2+3
        stride = num_a+num_b
        goals = [i*stride for i in range(chunks)]
        actions = [i*stride+j for i in range(chunks) for j in range(2, 5)]
        worlds = [i*stride+j for i in range(chunks) for j in range(num_a, num_a+18)]
        policies = [i*stride+j for i in range(chunks) for j in range(num_a+18, stride)]
        mask = build_mask(K=chunks, num_A=num_a, num_B=num_b,
                          mask_l_obs_ratio=0., num_obs_token=18, resampler_query=6,
                          image_seq_len=1, action_seq_len=3, state_seq_len=1,
                          action_pred_steps=1, pred_obs_token=18, atten_goal=0,
                          pred_state=False)
        allowed = torch.isfinite(mask)
        repaired = allowed.clone()
        repaired[torch.tensor(actions)[:, None], torch.tensor(goals)[None, :]] = False
        for name, edges in [("native", allowed), ("cut_goal_to_input_action", repaired)]:
            reach = torch.zeros(len(mask), dtype=torch.bool)
            reach[goals] = True
            counts = []
            for depth in range(1, 25):
                reach = reach | edges[:, reach].any(dim=1)
                if depth in [1, 2, 3, 24]:
                    counts.append({"layers": depth, "reachable_world_outputs": int(reach[worlds].sum()),
                                   "reachable_policy_outputs": int(reach[policies].sum())})
            path = shortest_path(edges, goals, worlds, 24)
            def label(index):
                chunk, offset = divmod(index, stride)
                kind = ("goal" if offset == 0 else "input_state" if offset == 1 else
                        "input_action" if offset < 5 else "input_image" if offset < num_a else
                        "world_output" if offset < num_a+18 else "policy_output")
                return {"index": index, "chunk": chunk, "kind": kind}
            row = {"chunks": chunks, "condition": name, "tokens": len(mask),
                   "world_output_count": len(worlds), "policy_output_count": len(policies),
                   "direct_goal_to_world_edges": int(edges[worlds][:, goals].sum()),
                   "removed_edges": int((allowed & ~edges).sum()),
                   "reachability": counts,
                   "shortest_world_path": [label(i) for i in path] if path else None}
            if name == "native":
                assert row["direct_goal_to_world_edges"] == 0
                assert path is not None and len(path)-1 == 2
            else:
                assert path is None
                assert counts[-1]["reachable_policy_outputs"] == len(policies)
            rows.append(row)
    result = {
        "status": "native_mask_allows_indirect_goal_path_pretrained_effect_unmeasured",
        "results": rows,
        "interpretation": "The direct goal-to-world mask does not close the two-layer path through contextualized input-action tokens. The action values themselves remain fixed. Cutting only goal-to-input-action edges removes structural world-output reachability while leaving goal-conditioned policy outputs reachable. This is an architecture-level result, not a trained-model effect or security success.",
        "elapsed_s": time.monotonic()-started,
        "native_mask_calls": 2, "pretrained_model_loads": 0,
        "native_environment_steps": 0, "training_updates": 0, "api_requests": 0,
        "gpu_use": False,
    }
    (RUN/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
