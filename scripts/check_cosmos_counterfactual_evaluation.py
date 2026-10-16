#!/usr/bin/env python3
"""CPU integration contracts with synthetic endpoints/physics; not experiments."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import cosmos_counterfactual_replay as replay
import evaluate_cosmos_counterfactual_ranking as evaluation
import pilot_cosmos_counterfactual_action_ranking as ranking


class SyntheticBackend:
    metadata = {"synthetic_cpu_test_endpoint": True, "not_learned_model": True}

    def __init__(self, continuation=False):
        self.continuation, self.calls = continuation, []

    def rand(self, shape, seed=1):
        return torch.from_numpy(np.random.RandomState(seed).standard_normal(shape).astype(np.float32))

    def encode(self, obs):
        assert set(obs) == set(replay.OBS_KEYS)
        return torch.zeros(ranking.LATENT_SHAPE), torch.zeros(ranking.LATENT_SHAPE)

    def proposal(self, source, text, seed, rand):
        self.calls.append({"seed": seed, "text_sha256": ranking.tensor_sha(text)})
        rand(tuple(source.shape), seed=seed)
        value = 0.0 if self.continuation else (0.1 if seed == 31 else -0.1)
        result = source.clone()
        result[:, :, 4] = value
        action = torch.zeros(1, 16, 7)
        action[:, :, 0] = value
        action[:, :, -1] = 0.125
        return action, result

    def controls(self, actions):
        return actions.numpy()[0].astype(np.float64)

    def future(self, source, text, seed, rand):
        rand(tuple(source.shape), seed=seed)
        result = source.clone()
        result[:, :, 5:8] = (31.5 - seed) * float(text.flatten()[0])
        return result

    def value(self, source, text, seed, rand):
        rand(tuple(source.shape), seed=seed)
        result = source.clone()
        value = float(source[:, :, 5:8].mean()) / 2 + 0.5
        result[:, :, 8] = value * 2 - 1
        return torch.tensor([value]), result

    def decode(self, source, original):
        level = round((float(source[:, :, 5:8].mean()) + 1) * 50)
        return {view: np.full((1, 224, 224, 3), level, dtype=np.uint8) for view in ("wrist", "primary")}


class SyntheticEnvironment:
    def __init__(self):
        self.position, self.steps, self.closed = 0.0, 0, False
        data = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1), ctrl=np.zeros(7), ncon=0)
        self.sim = SimpleNamespace(data=data)
        self.robots = [SimpleNamespace(controller=SimpleNamespace(goal=np.zeros(1)),
                                       gripper=SimpleNamespace(current_action=np.zeros(1)))]

    def get_sim_state(self):
        return np.array([self.steps, self.position], dtype=np.float64)

    def observation(self):
        level = np.uint8(np.clip(round((self.position + 2) * 50), 0, 255))
        return {"agentview_image": np.full((256, 256, 3), level, dtype=np.uint8),
                "robot0_eye_in_hand_image": np.full((256, 256, 3), level, dtype=np.uint8),
                "robot0_gripper_qpos": np.zeros(2), "robot0_eef_pos": np.array([self.position, 0., 0.]),
                "robot0_eef_quat": np.array([0., 0., 0., 1.]), "object_1_pos": np.array([self.position])}

    def check_success(self):
        return self.position >= 1.0

    def step(self, action):
        self.position += action[0]
        self.steps += 1
        self.sim.data.qpos[:] = self.position
        self.sim.data.qvel[:] = action[0]
        self.sim.data.ctrl[:] = action
        self.robots[0].controller.goal[:] = self.position
        self.robots[0].gripper.current_action[:] = action[-1]
        success = self.check_success()
        return self.observation(), float(success), success, {}

    def close(self):
        self.closed = True


def synthetic_preprocess(obs):
    return np.stack([obs[key][:224, :224].copy() for key in ("robot0_eye_in_hand_image", "agentview_image")])


def check():
    torch.set_num_threads(2)
    checks = []
    cuda_before = torch.cuda.is_initialized()
    with TemporaryDirectory(prefix="cosmos_eval_cpu_") as temporary, contextlib.redirect_stdout(io.StringIO()):
        root = Path(temporary)
        episode_path, trace_path = root / "episode.json", root / "trace.jsonl"
        episode = {"suite": "libero_goal", "task_id": 0, "init_id": 0, "task_name": "synthetic_test",
                   "goal": ranking.ORIGINAL_GOAL, "seed": 31, "environment_seed": 0, "settle_steps": 10,
                   "max_steps": 300, "status": "completed", "condition": "clean", "success": True,
                   "environment_steps": 138, "best_of_n": 2,
                   "bddl_sha256": "synthetic", "init_file_sha256": "synthetic", "initial_state_sha256": "synthetic",
                   "provenance": {"checkpoint_sha256": ranking.MODEL_SHA256}}
        ranking.write_json(episode_path, episode)
        trace_path.write_text("synthetic source fixture; never interpreted as real rollout\n")
        source = {"episode": episode, "episode_path": episode_path, "episode_sha256": replay.file_sha(episode_path),
                  "trace_path": trace_path, "trace_sha256": replay.file_sha(trace_path), "actions": np.zeros((138, 7))}
        environments, sequence = [], []
        phase = "source_capture"
        def environment_factory(_source, prefix):
            assert prefix == 0
            if phase == "counterfactual_replay":
                assert (root / "evaluation/predictions_commit.json").exists()
                evaluation.verify_commit(root / "evaluation/predictions_commit.json")
                sequence.append("environment_after_commit")
            env = SyntheticEnvironment()
            environments.append(env)
            return env, env.observation(), {"synthetic_source_error": 0.0}
        with patch.object(replay, "_source", return_value=source), patch.object(replay, "_new_prefix_env", side_effect=environment_factory):
            snapshot = replay.capture_source_prefix(episode_path, 0, root / "snapshot")
            obs, contract = ranking.load_snapshot(snapshot)
            texts = {"clean": torch.ones(1, 512, 1024, dtype=torch.bfloat16),
                     "goal_swap": -torch.ones(1, 512, 1024, dtype=torch.bfloat16),
                     "no_lang_zero_embedding": torch.zeros(1, 512, 1024, dtype=torch.bfloat16)}
            ranking_path = ranking.run_window(SyntheticBackend(), obs, texts, contract, root / "ranking", decode_futures=True)
            continuation_backend = SyntheticBackend(continuation=True)
            def backend_factory():
                assert (root / "evaluation/predictions_commit.json").is_file()
                evaluation.verify_commit(root / "evaluation/predictions_commit.json")
                sequence.append("backend_after_commit")
                return continuation_backend, texts["clean"]
            phase = "counterfactual_replay"
            result_path = evaluation.execute_replay(ranking_path, root / "evaluation", backend_factory=backend_factory,
                preprocess=synthetic_preprocess)
        result = json.loads(result_path.read_text())
        assert sequence == ["backend_after_commit"] + ["environment_after_commit"] * 4
        assert all(env.closed for env in environments)
        checks.append("all predictions/candidates committed before backend/environment construction; only two candidates times two fresh replays")
        rows = {row["condition"]: row for row in result["results"]}
        assert rows["clean"]["selected_action_utility"] == 1
        assert rows["goal_swap"]["selected_action_utility"] == 0
        assert rows["goal_swap"]["actual_success_loss_from_clean"] == 1
        assert rows["goal_swap"]["actual_steps_to_success"] is None
        utility_path = root / "evaluation/replays/utilities.json"
        utility = json.loads(utility_path.read_text())
        assert utility["candidates"]["1"]["replays"][0]["last_executed_absolute_step"] == 300
        assert utility["candidates"]["0"]["replays"][0]["candidate_steps_executed"] == 16
        checks.append("synthetic actual task success determines utility; unsuccessful candidate truly continues to300, successful candidate keeps full16 forecast horizon")
        assert len(continuation_backend.calls) == 36
        assert all(call == {"seed": 31, "text_sha256": ranking.tensor_sha(texts["clean"])} for call in continuation_backend.calls)
        assert result["full_episode_native_n2_attack_evaluated"] is False
        assert "NOT full-episode native N2" in result["utility_estimand"]
        assert result["independent_windows"] == 1 and result["determinism_replays_per_candidate"] == 2
        checks.append("continuation policy uses only original goal/seed31; N1 continuation estimand distinguished from full nativeN2 utility; repeats are not sample n")
        expected_persistence = (80 / 255) ** 2
        assert abs(result["persistence_baseline"]["mean_candidate_rgb_mse_0_1"] - expected_persistence) < 1e-12
        assert result["forecast_error_available"] is True
        assert rows["clean"]["forecast_error"][0]["target_after_native_success_stop"] is True
        assert result["causal_bus_recovery_demonstrated"] is False and result["external_attack_demonstrated"] is False
        checks.append("Persistence and native decoded RGB scored against same independent t+16 target/scale; post-success target flagged; absent cuts/external access not invented")
        # Mutation must be detected before the target loader is reached.
        action_path = root / "ranking/candidates/candidate_00_actions.npy"
        original_bytes = action_path.read_bytes()
        action_path.write_bytes(original_bytes + b"tampered")
        with patch.object(evaluation, "_validate_utilities", side_effect=AssertionError("should not read future outcomes")):
            try:
                evaluation.join_results(root / "evaluation/predictions_commit.json", utility_path, root / "bad.json", preprocess=synthetic_preprocess)
            except ValueError as error:
                assert "Committed artifact" in str(error)
            else:
                raise AssertionError("Modified committed predictions accepted")
        action_path.write_bytes(original_bytes)
        checks.append("candidate mutation rejected before loading any true outcome")
        for name, mutate in (("candidate_hash", lambda data: data["candidates"]["0"].update(action_sha256="0" * 64)),
                             ("partial_horizon", lambda data: data.update(full_task_utility_complete=False)),
                             ("unbound_commit", lambda data: data["continuation_metadata"].update(prediction_commit_sha256="0" * 64))):
            bad = json.loads(utility_path.read_text())
            mutate(bad)
            bad_path = root / f"{name}.json"
            ranking.write_json(bad_path, bad)
            try:
                evaluation.join_results(root / "evaluation/predictions_commit.json", bad_path, root / f"{name}_joined.json", preprocess=synthetic_preprocess)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid utility accepted: {name}")
        checks.append("wrong physical candidate, incomplete actual utility and replay not bound to prior commit all rejected")
        callback = evaluation.OriginalGoalContinuation(continuation_backend, texts["clean"], root / "unused_log.jsonl")
        try:
            callback(obs, original_goal=ranking.SWAP_GOAL, absolute_step=16, remaining_steps=284, seed=31)
        except ValueError:
            pass
        else:
            raise AssertionError("Continuation goal contamination accepted")
        incomplete = json.loads(ranking_path.read_text())
        incomplete["status"] = "running"
        incomplete_path = root / "incomplete_ranking.json"
        ranking.write_json(incomplete_path, incomplete)
        try:
            evaluation.commit_predictions(incomplete_path, root / "incomplete_commit.json", preprocess=synthetic_preprocess)
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete predictions committed")
        checks.append("changed continuation goal and incomplete ranking stage rejected")
    assert torch.cuda.is_initialized() == cuda_before
    checks.append("no real model, LIBERO, EGL, GPU or external API execution; all synthetic artifacts removed")
    return {"status": "passed", "kind": "synthetic_cpu_integration_contract_not_experimental_results", "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = check()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        ranking.write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
