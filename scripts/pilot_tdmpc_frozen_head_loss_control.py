#!/usr/bin/env python3
"""Frozen-head task-loss qualification on existing TD-MPC2 source23 data.

Implementation only until separately launched. This is a bounded researcher
adaptation, not native joint training or an attack. No source data collection.
"""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class HistoricalMismatch(RuntimeError):
    pass


class ResourceDeferred(RuntimeError):
    pass


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gpu_metadata(index):
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,uuid", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    )
    for line in text.splitlines():
        values = [part.strip() for part in line.split(",")]
        if int(values[0]) == index:
            return {"index": index, "free_mib": int(values[1]), "total_mib": int(values[2]), "uuid": values[3]}
    raise ResourceDeferred("Configured physical GPU was not found")


def frozen_equal(model, original, torch):
    current = model.state_dict()
    require(set(current) == set(original), "Model state_dict keys changed")
    changed, checked = [], 0
    for name, reference in original.items():
        if name.startswith("_dynamics."):
            continue
        value = current[name]
        equal = torch.equal(value.detach().cpu(), reference) if isinstance(reference, torch.Tensor) else value == reference
        checked += 1
        if not equal:
            changed.append(name)
    require(not changed, "Frozen state changed: " + repr(changed))
    return {"non_dynamics_state_entries_checked": checked, "exact": True}


def compare_history(history, reference, spec):
    require(len(history) == len(reference) == 300, "Historical log must contain all 300 updates")
    failures, maxima = [], {}
    for field in spec["fields"]:
        errors = []
        for new, old in zip(history, reference):
            require(new["update"] == old["update"], "Historical update index differs")
            error = abs(new[field] - old[field])
            errors.append(error)
            if error > spec["atol"] + spec["rtol"] * abs(old[field]):
                failures.append({"update": new["update"], "field": field, "actual": new[field], "reference": old[field]})
        maxima[field] = max(errors)
    return {"passed": not failures, "updates_checked": 300, "max_absolute_errors": maxima, "failures": failures, "tolerances": spec}


def train_arm(agent, data, starts_bank, condition, protocol, arm, deadline, persist, torch, native_math):
    import torch.nn.functional as F

    model, cfg = agent.model, agent.cfg
    spec = protocol["adaptation"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    parameters = list(model._dynamics.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    model.eval()
    optim_spec = spec["optimizer"]
    optimizer = torch.optim.Adam(parameters, lr=cfg.lr, betas=tuple(optim_spec["betas"]),
                                 eps=optim_spec["eps"], weight_decay=optim_spec["weight_decay"])
    observations, actions = data["obs"].to(agent.device), data["action"].to(agent.device)
    rewards, task = data["reward"].to(agent.device), int(data["task"])
    with torch.no_grad():
        encoded = model.encode(observations, task)
    task_vector = torch.full((32,), task, dtype=torch.long, device=agent.device)
    history = arm["learning"]
    task_loss = condition["loss"] == "consistency_reward_value"
    for update, starts in enumerate(starts_bank):
        deadline()
        require(not model.training and not model._Qs.training and not model._target_Qs.training,
                "Model/head training mode changed")
        z = encoded[starts]
        states = [z]
        loss = torch.zeros((), device=agent.device)
        for step in range(cfg.horizon):
            z = model.next(z, actions[starts + step], task)
            loss = loss + cfg.rho**step * F.mse_loss(z, encoded[starts + step + 1])
            states.append(z)
        # Preserve the old arithmetic order exactly for the reconstruction arms.
        consistency = loss / cfg.horizon * cfg.consistency_coef
        total = consistency
        extra = {}
        if task_loss:
            action_seq = torch.stack([actions[starts + t] for t in range(cfg.horizon)])
            reward_seq = torch.stack([rewards[starts + t] for t in range(cfg.horizon)]).unsqueeze(-1)
            next_z = torch.stack([encoded[starts + t + 1] for t in range(cfg.horizon)])
            before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(agent.device)
            # Native stochastic actor and two-Q subset are paired by update and
            # isolated from starts and subsequent evaluation RNG streams.
            with torch.random.fork_rng(devices=[agent.device.index]):
                torch.random.default_generator.manual_seed(314159 + update)
                torch.cuda.default_generators[agent.device.index].manual_seed(314159 + update)
                with torch.no_grad():
                    targets = agent._td_target(next_z, reward_seq, torch.zeros_like(reward_seq), task_vector)
            require(torch.equal(before_cpu, torch.get_rng_state()) and
                    torch.equal(before_cuda, torch.cuda.get_rng_state(agent.device)), "Target sampling escaped its RNG fork")
            # Freeze head parameters, not their Jacobians with respect to z.
            zs = torch.stack(states[:-1])
            qs = model.Q(zs, action_seq, task_vector, return_type="all")
            reward_preds = model.reward(zs, action_seq, task_vector)
            reward_loss, value_loss = 0, 0
            for t, (prediction, reward, target, all_q) in enumerate(zip(
                    reward_preds.unbind(0), reward_seq.unbind(0), targets.unbind(0), qs.unbind(1))):
                reward_loss = reward_loss + native_math.soft_ce(prediction, reward, cfg).mean() * cfg.rho**t
                for q in all_q.unbind(0):
                    value_loss = value_loss + native_math.soft_ce(q, target, cfg).mean() * cfg.rho**t
            reward_loss = reward_loss / cfg.horizon
            value_loss = value_loss / (cfg.horizon * cfg.num_q)
            auxiliary = cfg.reward_coef * reward_loss + cfg.value_coef * value_loss
            total = consistency + auxiliary
            if update == 0:
                gradients = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
                norm = torch.sqrt(sum((g.detach().float().square().sum() for g in gradients if g is not None),
                                      torch.zeros((), device=agent.device)))
                norm_value = float(norm)
                require(math.isfinite(norm_value) and norm_value > 0, "Auxiliary gradients do not reach dynamics")
                arm["auxiliary_gradient_check"] = {
                    "weighted_auxiliary_dynamics_grad_norm": norm_value,
                    "reward_prediction_requires_grad": reward_preds.requires_grad,
                    "Q_prediction_requires_grad": qs.requires_grad,
                    "target_requires_grad": targets.requires_grad,
                    "separate_optimizer_steps": 0,
                }
                require(reward_preds.requires_grad and qs.requires_grad and not targets.requires_grad,
                        "Incorrect frozen-head input/target gradient boundary")
            extra = {"reward_loss": float(reward_loss.detach()), "value_loss": float(value_loss.detach()),
                     "target_seed": 314159 + update, "target_rng_restored": True}
        require(bool(torch.isfinite(total)), "Nonfinite scheduled loss")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        require(all(p.grad is None for name, p in model.named_parameters() if not name.startswith("_dynamics.")),
                "A frozen parameter received a gradient")
        grad = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip_norm)
        require(bool(torch.isfinite(grad)), "Nonfinite dynamics gradient")
        optimizer.step()
        history.append({"update": update + 1, "weighted_consistency_loss": float(consistency.detach()),
                        "grad_norm": float(grad), "total_loss": float(total.detach()), **extra})
        if (update + 1) % 25 == 0 or update == 0:
            persist()
            print(json.dumps({"arm": condition["id"], "update": update + 1}), flush=True)


def evaluate(agent, spec, destination, arm, native, deadline, persist, torch):
    """Reuse native episode; an observational proxy also preserves partial traces."""
    observations, actions, rewards = [], [], []
    original_make = native.make_one

    class RecordingEnv:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)

        def reset(self, *args, **kwargs):
            obs = self.env.reset(*args, **kwargs)
            observations.append(native.pad_obs(obs, agent.cfg).clone())
            return obs

        def step(self, action):
            deadline()
            result = self.env.step(action)
            obs, reward, _, _ = result
            observations.append(native.pad_obs(obs, agent.cfg).clone())
            actions.append(action.clone())
            rewards.append(float(reward))
            return result

    def make(task, seed):
        return RecordingEnv(original_make(task, seed))

    def progress(info):
        arm["evaluation_progress"] = info
        arm["physical_controls_recorded"] = len(rewards)
        persist()
        print(json.dumps({"arm": arm["id"], **info}), flush=True)
        deadline()

    try:
        with patch.object(native, "make_one", make):
            result, data = native.episode(agent, spec["task"], spec["seed"], spec["policy"], spec["steps"], progress)
        torch.save(data, destination / "evaluation.pt")
        require(result["steps"] == spec["steps"] and result["full_episode"], "Incomplete native episode")
        require(torch.equal(data["obs"], torch.stack(observations)) and
                torch.equal(data["action"], torch.stack(actions)) and
                torch.equal(data["reward"], torch.tensor(rewards)), "Recording proxy differs from native returned trace")
        return result
    finally:
        arm["physical_controls_recorded"] = len(rewards)
        if observations:
            torch.save({"obs": torch.stack(observations),
                        "action": torch.stack(actions) if actions else torch.empty((0, agent.cfg.action_dim)),
                        "reward": torch.tensor(rewards), "task": spec["task_id"], "seed": spec["seed"]},
                       destination / "recorded_trace.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tdmpc_frozen_head_loss_control_20261015.json")
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text())
    out = ROOT / protocol["output"]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, out / "launched_config.json")
    shutil.copyfile(__file__, out / "launched_script.py")
    started = time.monotonic()
    summary = {"status": "starting", "protocol": str(args.config), "arms": {},
               "activity": {"native_load_agent_calls": 0, "source_environment_steps": 0,
                            "network_attempts": 0, "optimizer_updates": 0, "physical_evaluation_controls": 0},
               "scope": protocol["scope"]}

    def persist():
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["activity"]["optimizer_updates"] = sum(len(a["learning"]) for a in summary["arms"].values())
        summary["activity"]["physical_evaluation_controls"] = sum(a.get("physical_controls_recorded", 0) for a in summary["arms"].values())
        for arm_id, arm in summary["arms"].items():
            write(out / arm_id / "learning.json", arm["learning"])
        write(out / "summary.json", summary)

    def deadline():
        if time.monotonic() - started > protocol["resources"]["deadline_seconds"]:
            raise TimeoutError("Frozen whole-process deadline reached; no retry")

    def no_network(*args, **kwargs):
        summary["activity"]["network_attempts"] += 1
        raise RuntimeError("Network access is prohibited in this cached-data pilot")

    try:
        resources = protocol["resources"]
        metadata = gpu_metadata(resources["gpu_index"])
        summary["gpu_at_start"] = metadata
        if metadata["free_mib"] < resources["minimum_free_memory_mib"]:
            raise ResourceDeferred("GPU free memory below the frozen launch threshold")
        revision = subprocess.check_output(["git", "-C", str(ROOT / "vendor/tdmpc2"), "rev-parse", "HEAD"], text=True).strip()
        require(revision == protocol["source_revision"], "Upstream revision differs")
        import torch
        from wm_security import tdmpc_native as native

        torch.cuda.set_device(resources["gpu_index"])
        total_bytes = torch.cuda.get_device_properties(resources["gpu_index"]).total_memory
        cap_bytes = resources["allocator_cap_gib"] * 1024**3
        torch.cuda.set_per_process_memory_fraction(cap_bytes / total_bytes, resources["gpu_index"])
        summary["allocator"] = {"cap_bytes": cap_bytes, "fraction": cap_bytes / total_bytes,
                                "limit": "Allocator only; CUDA context and external allocations are not covered."}
        with ExitStack() as stack:
            stack.enter_context(patch.object(socket.socket, "connect", no_network))
            summary["activity"]["native_load_agent_calls"] += 1
            agent, provenance = native.load_agent(ROOT / protocol["checkpoint"])
            summary["provenance"] = provenance
            from common import math as native_math

            cfg = agent.cfg
            for key in ["horizon", "rho", "consistency_coef", "reward_coef", "value_coef", "num_q", "grad_clip_norm"]:
                require(getattr(cfg, key) == protocol["adaptation"][key], "Native adaptation config mismatch: " + key)
            require(cfg.lr == protocol["adaptation"]["optimizer"]["lr"] and not cfg.episodic, "Native learning/termination contract differs")
            for key in ["horizon", "num_samples", "iterations", "num_elites", "num_pi_trajs", "compile"]:
                require(getattr(cfg, key) == protocol["evaluation"]["native_planner"][key], "Native planner config mismatch: " + key)
            require(cfg.tasks.index(protocol["source_data"]["task"]) == protocol["source_data"]["task_id"] and
                    cfg.tasks.index(protocol["evaluation"]["task"]) == protocol["evaluation"]["task_id"], "Task IDs differ")
            require(cfg.action_dims[protocol["evaluation"]["task_id"]] == cfg.action_dim == 6,
                    "The observational trace proxy requires this frozen task's full six-dimensional controls")
            original = {name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
                        for name, value in agent.model.state_dict().items()}
            sources = {}
            for source in ("random", "cascade"):
                data = torch.load(ROOT / protocol["source_data"][source], map_location="cpu", weights_only=False)
                require(int(data["task"]) == protocol["source_data"]["task_id"] and
                        int(data["seed"]) == protocol["source_data"]["source_seed"], "Cached source identity differs")
                for field, shape in protocol["source_data"]["shapes"].items():
                    require(list(data[field].shape) == shape and bool(torch.isfinite(data[field]).all()), "Invalid source tensor: " + field)
                sources[source] = data
            generator = torch.Generator(device=agent.device).manual_seed(protocol["adaptation"]["learning_seed"])
            # Call shape and count match the historical default-CUDA draws.
            starts_bank = torch.stack([torch.randint(298, (32,), device=agent.device, generator=generator) for _ in range(300)])
            torch.save({"starts": starts_bank.cpu(), "private_generator_final_state": generator.get_state(),
                        "seed": 31, "high": 298, "draw_shape": [32], "draw_calls": 300}, out / "batch_starts.pt")
            historical = protocol["historical_reconstruction"]
            for condition in protocol["conditions_in_order"]:
                deadline()
                if condition["loss"] == "consistency_reward_value":
                    require(all(summary["arms"].get("consistency_" + s, {}).get("learning_reconstruction", {}).get("passed", False)
                                and summary["arms"].get("consistency_" + s, {}).get("historical_return_check", {}).get("passed", False)
                                for s in ("random", "cascade")), "Task loss cannot precede both complete historical gates")
                arm_id = condition["id"]
                destination = out / arm_id
                destination.mkdir()
                arm = {"id": arm_id, "status": "training", "source": condition["source"], "loss": condition["loss"], "learning": []}
                summary["arms"][arm_id] = arm
                agent.model.load_state_dict(original, strict=True)
                agent.model.eval()
                with torch.no_grad():
                    agent._prev_mean.zero_()
                native.set_seed(protocol["adaptation"]["learning_seed"])
                arm["frozen_before"] = frozen_equal(agent.model, original, torch)
                persist()
                try:
                    train_arm(agent, sources[condition["source"]], starts_bank, condition, protocol, arm, deadline, persist, torch, native_math)
                finally:
                    # Preserve the actual current dynamics even on a partial update run.
                    torch.save({k: v.detach().cpu().clone() for k, v in agent.model._dynamics.state_dict().items()}, destination / "dynamics.pt")
                arm["frozen_after_training"] = frozen_equal(agent.model, original, torch)
                if condition["loss"] == "consistency":
                    old = json.loads((ROOT / historical["learning_logs"][condition["source"]]).read_text())
                    arm["learning_reconstruction"] = compare_history(arm["learning"], old, historical["learning_all_300_steps"])
                    persist()
                    if not arm["learning_reconstruction"]["passed"]:
                        raise HistoricalMismatch(arm_id + " does not reproduce the old 300-update log")
                arm["status"] = "evaluating"
                result = evaluate(agent, protocol["evaluation"], destination, arm, native, deadline, persist, torch)
                arm["evaluation"] = result
                arm["frozen_after_evaluation"] = frozen_equal(agent.model, original, torch)
                if condition["loss"] == "consistency":
                    reference = historical["seed31_historical_returns"][condition["source"]]
                    tolerance = historical["return_absolute_tolerance"] + historical["return_relative_tolerance"] * abs(reference)
                    arm["historical_return_check"] = {"reference": reference, "actual": result["return"],
                                                       "absolute_tolerance": tolerance, "passed": abs(result["return"] - reference) <= tolerance}
                    persist()
                    if not arm["historical_return_check"]["passed"]:
                        raise HistoricalMismatch(arm_id + " does not reproduce the old seed31 return")
                arm["status"] = "completed"
                persist()
            returns = {k: a["evaluation"]["return"] for k, a in summary["arms"].items()}
            g0 = returns["consistency_random"] - returns["consistency_cascade"]
            g1 = returns["task_loss_random"] - returns["task_loss_cascade"]
            change_random = returns["task_loss_random"] - returns["consistency_random"]
            change_cascade = returns["task_loss_cascade"] - returns["consistency_cascade"]
            summary["analysis"] = {"returns": returns, "G0": g0, "G1": g1, "interaction": g0 - g1,
                                   "random_source_change": change_random, "cascade_source_change": change_cascade,
                                   "conditional_followup_signal": g0 > 0 and g1 <= 0.5*g0 and change_cascade >= 0.5*g0 and returns["task_loss_random"] >= 0.95*returns["consistency_random"],
                                   "limits": protocol["analysis"]["interpretation"], "statistics": protocol["analysis"]["statistics"]}
            summary["status"] = "completed"
    except ResourceDeferred as exc:
        summary.update(status="resource_deferred_before_model_load", reason=str(exc))
    except HistoricalMismatch as exc:
        summary.update(status="invalid_historical_reconstruction_task_loss_not_authorized", reason=str(exc))
    except TimeoutError as exc:
        summary.update(status="resource_deadline_incomplete", reason=str(exc))
    except Exception as exc:
        summary.update(status="error_preserved_no_retry", reason=str(exc), traceback=traceback.format_exc())
    finally:
        persist()
        print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return 0 if summary["status"] in ("completed", "resource_deferred_before_model_load") else 1


if __name__ == "__main__":
    raise SystemExit(main())
