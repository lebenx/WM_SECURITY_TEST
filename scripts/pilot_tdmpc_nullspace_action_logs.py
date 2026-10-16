#!/usr/bin/env python3
"""Fixed, bounded executor-null action-log continuation experiment.

Use an existing real source trajectory and preserve its complete physical
transition multiset and order. This is an added offline writer and a dynamics
ablation, not a native online exploit or a native full-training experiment.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOL = ROOT / "configs/tdmpc_nullspace_action_logs_20261015.json"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    protocol = json.loads(PROTOCOL.read_text())
    out = ROOT / protocol["output"]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PROTOCOL, out / "frozen_protocol.json")
    shutil.copyfile(__file__, out / "launched_script.py")
    start = time.monotonic()
    activity = {"checkpoint_loads": 0, "training_updates": 0,
                "source_replay_controls": 0, "evaluation_controls": 0,
                "explicit_environment_resets": 0, "evaluation_episodes_completed": 0,
                "model_api_requests": 0}
    summary = {"status": "running", "protocol": str(PROTOCOL.relative_to(ROOT)),
               "activity": activity, "conditions": {}, "scope": protocol["scope"]}

    def persist():
        summary["elapsed_seconds"] = time.monotonic() - start
        write(out / "summary.json", summary)

    def deadline():
        require(time.monotonic() - start <= protocol["resources"]["deadline_seconds"],
                "Registered wall-time budget exceeded; this is incomplete, not task failure")

    persist()
    try:
        metadata = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"
        ], text=True, timeout=15)
        rows = [[int(p.strip()) for p in row.split(",")] for row in metadata.strip().splitlines()]
        selected = next(r for r in rows if r[0] == protocol["resources"]["gpu_index"])
        summary["gpu_before"] = {"index": selected[0], "free_mib": selected[1], "total_mib": selected[2]}
        require(selected[1] >= protocol["resources"]["minimum_free_memory_mib"], "Insufficient free memory")

        import torch
        import torch.nn.functional as F
        from wm_security import tdmpc_native as native
        torch.cuda.set_per_process_memory_fraction(
            protocol["resources"]["allocator_cap_gib"] * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
        source = torch.load(ROOT / protocol["source"]["path"], map_location="cpu", weights_only=False)
        require(tuple(source["obs"].shape) == (301, 24) and tuple(source["action"].shape) == (300, 6), "Unexpected source shapes")
        require(int(source["task"]) == 4 and int(source["seed"]) == 21, "Unexpected source task/seed")
        require(bool((source["action"][:, 2:] == 0).all()), "Source is not zero padded")
        require(bool(torch.isfinite(source["action"]).all()) and float(source["action"].abs().max()) <= 1, "Invalid source actions")
        correlated = source["action"].clone()
        correlated[:, 2:] = source["action"][:, :2].repeat(1, 2)
        permutation = torch.randperm(300, generator=torch.Generator().manual_seed(protocol["source"]["permutation_seed"]))
        permuted = correlated.clone()
        permuted[:, 2:] = correlated[permutation, 2:]
        variants = {"zero": source["action"], "correlated": correlated, "permuted": permuted}
        require(torch.equal(permuted[torch.argsort(permutation), 2:], correlated[:, 2:]), "Joint-tail multiset changed")
        mask = torch.tensor([1., 1., 0., 0., 0., 0.])
        for name, action in variants.items():
            require(torch.equal(action[:, :2], source["action"][:, :2]), "Active action changed")
            require(float(action.abs().max()) <= 1, "Edited action outside Box")
            require(torch.equal(action * mask, source["action"]), "Task-mask identity failed")
        torch.save({"actions": variants, "permutation": permutation}, out / "action_variants.pt")
        summary["data_control"] = {"joint_tail_multiset_exact": True, "active_actions_exact": True,
            "mask_recovers_zero_exact": True, "mask_recovery_episode_executed": False,
            "protected_data": "All arms reference the identical source obs/reward/task/seed tensors; only action tail is substituted.",
            "tail_range": [float(correlated[:, 2:].min()), float(correlated[:, 2:].max())]}

        # Invoke actual upstream slicing code, not a local emulation of it.
        native.imports()
        from envs.wrappers.multitask import MultitaskWrapper
        replay_reference = None
        for condition in ["zero", "correlated"]:
            deadline()
            native.set_seed(21)
            base_env = native.make_one("reacher-easy", 21)
            # Recent Gymnasium no longer forwards arbitrary attributes through
            # wrappers. Restore only the metadata expected by upstream __init__.
            base_env.max_episode_steps = base_env.env.max_episode_steps
            env = MultitaskWrapper(SimpleNamespace(tasks=["reacher-easy"]), [base_env])
            replay_obs, replay_rewards, replay_done = [], [], []
            try:
                obs = env.reset(task_idx=0)
                activity["explicit_environment_resets"] += 1
                replay_obs.append(F.pad(obs, (0, 24 - obs.shape[0])))
                for action in variants[condition]:
                    deadline()
                    obs, reward, done, _ = env.step(action)
                    activity["source_replay_controls"] += 1
                    replay_obs.append(F.pad(obs, (0, 24 - obs.shape[0])))
                    replay_rewards.append(float(reward))
                    replay_done.append(bool(done))
                    require(not done, "Unexpected terminal inside cached300-control prefix")
            finally:
                env.unwrapped.close()
            replay = {"obs": torch.stack(replay_obs), "reward": torch.tensor(replay_rewards),
                      "native_rewards": replay_rewards, "done": replay_done}
            torch.save(replay, out / f"{condition}_physical_replay.pt")
            require(torch.equal(replay["obs"], source["obs"]), "Replay observations do not reconstruct cached source")
            require(torch.equal(replay["reward"], source["reward"]), "Replay rewards do not reconstruct cached source")
            if replay_reference is not None:
                require(torch.equal(replay["obs"], replay_reference["obs"])
                        and replay["native_rewards"] == replay_reference["native_rewards"]
                        and replay["done"] == replay_reference["done"], "Physical equivalence failed")
            replay_reference = replay
        summary["physical_equivalence"] = {"passed": True, "controls_per_replay": 300,
            "cached_float32_obs_and_rewards_exact": True, "paired_native_rewards_and_done_exact": True,
            "compatibility": "Expose max_episode_steps metadata through TensorWrapper; upstream reset/step unchanged. Rewards already pass through native float32 TensorWrapper.",
            "scope": "Two actual upstream-wrapper replays; no new source states selected."}
        persist()

        deadline()
        activity["checkpoint_loads"] += 1
        agent, provenance = native.load_agent(ROOT / protocol["checkpoint"])
        write(out / "provenance.json", provenance)
        cfg, model = agent.cfg, agent.model
        require(cfg.action_dims[4] == 2 and cfg.action_dim == 6, "Action dimensions mismatch")
        spec = protocol["adaptation"]
        for field in ["horizon", "rho", "consistency_coef", "lr", "grad_clip_norm"]:
            require(getattr(cfg, field) == spec[field], "Native configuration differs: " + field)
        original = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
                    for k, v in model.state_dict().items()}
        generator = torch.Generator(device=agent.device).manual_seed(spec["seed"])
        starts_bank = torch.stack([torch.randint(298, (32,), device=agent.device, generator=generator)
                                   for _ in range(spec["updates"])])
        torch.save(starts_bank.cpu(), out / "training_starts.pt")
        heldout = torch.load(ROOT / "runs/tdmpc_adaptation_screen_v0/clean_reacher-easy.pt",
                             map_location="cpu", weights_only=False)
        historical = json.loads((ROOT / protocol["evaluation"]["historical_learning_log"]).read_text())["updates"]
        require(len(historical) == 40, "Historical training dose differs")

        def evaluate(condition):
            # Record physical controls individually so incomplete runs cannot
            # be reported as completed episodes or lost task outcomes.
            from unittest.mock import patch
            real_make = native.make_one
            partial = {"obs": [], "action": [], "reward": []}

            class RecordingEnv:
                def __init__(self, env):
                    self.env = env
                def __getattr__(self, name):
                    return getattr(self.env, name)
                def reset(self, *args, **kwargs):
                    obs = self.env.reset(*args, **kwargs)
                    activity["explicit_environment_resets"] += 1
                    partial["obs"].append(native.pad_obs(obs, cfg).clone())
                    return obs
                def step(self, action):
                    deadline()
                    result = self.env.step(action)
                    activity["evaluation_controls"] += 1
                    partial["action"].append(action.clone())
                    partial["reward"].append(float(result[1]))
                    partial["obs"].append(native.pad_obs(result[0], cfg).clone())
                    return result

            def progress(info):
                summary["conditions"][condition]["progress"] = info
                persist()
                print(json.dumps({"condition": condition, **info}), flush=True)

            try:
                with patch.object(native, "make_one", lambda task, seed: RecordingEnv(real_make(task, seed))):
                    result, data = native.episode(agent, "reacher-easy", 11, steps=500, progress=progress)
                torch.save(data, out / f"{condition}_evaluation.pt")
                require(result["steps"] == 500, "Unexpected shortened evaluation")
                require(torch.equal(data["obs"][0], heldout["obs"][0]), "Evaluation root changed")
                require(bool((data["action"][:, 2:] == 0).all()), "Native evaluation action tail is nonzero")
                activity["evaluation_episodes_completed"] += 1
                return result
            finally:
                torch.save(partial, out / f"{condition}_physical_trace.pt")

        for condition in protocol["conditions"]:
            deadline()
            model.load_state_dict(original)
            model.eval()
            arm = {"condition": condition, "learning": []}
            summary["conditions"][condition] = arm
            if condition != "clean":
                native.set_seed(spec["seed"])
                for p in model.parameters():
                    p.requires_grad_(False)
                    p.grad = None
                parameters = list(model._dynamics.parameters())
                for p in parameters:
                    p.requires_grad_(True)
                optimizer = torch.optim.Adam(parameters, lr=cfg.lr)
                actions = variants[condition].to(agent.device)
                with torch.no_grad():
                    encoded = model.encode(source["obs"].to(agent.device), 4)
                for u, starts in enumerate(starts_bank):
                    deadline()
                    z = encoded[starts]
                    loss = torch.zeros((), device=agent.device)
                    for t in range(cfg.horizon):
                        z = model.next(z, actions[starts + t], 4)
                        loss = loss + cfg.rho**t * F.mse_loss(z, encoded[starts + t + 1])
                    loss = loss / cfg.horizon * cfg.consistency_coef
                    require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip_norm)
                    require(bool(torch.isfinite(grad)), "Nonfinite gradient")
                    optimizer.step()
                    activity["training_updates"] += 1
                    arm["learning"].append({"update": u + 1, "weighted_consistency_loss": float(loss.detach()),
                                            "grad_norm": float(grad)})
                del optimizer
                torch.save(model._dynamics.state_dict(), out / f"{condition}_dynamics.pt")
                state = model.state_dict()
                unchanged = all(torch.equal(state[k].detach().cpu(), v) if isinstance(v, torch.Tensor)
                                else state[k] == v for k, v in original.items() if not k.startswith("_dynamics."))
                require(unchanged, "Non-dynamics state changed")
                arm["non_dynamics_state_exact"] = True
                if condition == "zero":
                    errors = {key: max(abs(a[key] - b[key]) for a, b in zip(arm["learning"], historical))
                              for key in ["weighted_consistency_loss", "grad_norm"]}
                    matched = all(math.isclose(a[key], b[key], rel_tol=protocol["evaluation"]["history_rtol"],
                                              abs_tol=protocol["evaluation"]["history_atol"])
                                  for a, b in zip(arm["learning"], historical)
                                  for key in errors)
                    arm["historical_learning_reconstruction"] = {"passed": matched, "maximum_absolute_errors": errors}
                    require(matched, "Zero-tail historical learning reconstruction failed")
            with torch.no_grad():
                obs, actions = heldout["obs"].to(agent.device), heldout["action"].to(agent.device)
                z, target = model.encode(obs[:-1], 4), model.encode(obs[1:], 4)
                arm["fixed_trace_one_step_latent_mse"] = float(F.mse_loss(model.next(z, actions, 4), target))
            arm["evaluation"] = evaluate(condition)
            if condition in ["clean", "zero"]:
                ref = protocol["evaluation"][f"historical_{condition}_return"]
                actual = arm["evaluation"]["return"]
                qualified = actual >= 900 and abs(actual - ref) <= 0.01 * abs(ref)
                arm["baseline_gate"] = {"passed": qualified, "reference_return": ref}
                require(qualified, "Baseline competence or historical-return gate failed")
            persist()
            print(json.dumps({"condition": condition, "return": arm["evaluation"]["return"],
                              "mse": arm["fixed_trace_one_step_latent_mse"]}), flush=True)
        returns = {k: v["evaluation"]["return"] for k, v in summary["conditions"].items()}
        # A zero-return control supplies no positive competence reference for
        # relative loss. Preserve its actual outcome and fail this harm gate.
        loss = {k: 1 - returns["correlated"] / returns[k] if returns[k] > 0 else None
                for k in ["zero", "permuted"]}
        passed = all(v is not None and v >= 0.10 for v in loss.values())
        summary["decision"] = {"conditional_harm_gate_passed": passed, "returns": returns,
            "relative_loss_correlated": loss, "permuted_minus_correlated_return": returns["permuted"] - returns["correlated"],
            "conclusion": "Conditional path signal; native-full qualification still required" if passed
                          else "No-Go for this fixed correlation-specific decision-harm candidate; no expansion or retuning",
            "statistics": "One fixed seed; descriptive paired differences only",
            "defense_episode_executed": False}
        deadline()
        summary["status"] = "completed"
        summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        persist()
        print(json.dumps(summary["decision"]), flush=True)
    except Exception as exc:
        summary["status"] = "incomplete_or_gate_failed"
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        persist()
        raise


if __name__ == "__main__":
    main()
