#!/usr/bin/env python3
"""CPU mock tests for replay orchestration; no LIBERO/model/GPU execution."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cosmos_counterfactual_replay as replay


class MockEnvironment:
    def __init__(self, offset=0.0):
        self.position, self.steps, self.closed = offset, 0, False
        self.data = SimpleNamespace(qpos=np.array([offset]), qvel=np.zeros(1),
                                    ctrl=np.zeros(7), qacc_warmstart=np.zeros(1), ncon=0)
        self.sim = SimpleNamespace(data=self.data)
        self.robots = [SimpleNamespace(controller=SimpleNamespace(goal=np.zeros(1)),
                                       gripper=SimpleNamespace(current_action=np.zeros(1)))]

    def observation(self):
        return {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "robot0_eef_pos": np.array([self.position, 0., 0.]),
                "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                "robot0_gripper_qpos": np.zeros(2), "object_1_pos": np.array([self.position])}

    def get_sim_state(self):
        return np.array([self.steps, self.position], dtype=np.float64)

    def check_success(self):
        return self.position >= 2.0

    def step(self, action):
        self.position += action[0]
        self.steps += 1
        self.data.qpos[:] = self.position
        self.data.qvel[:] = action[0]
        self.data.ctrl[:] = action
        self.robots[0].controller.goal[:] = self.position
        self.robots[0].gripper.current_action[:] = action[-1]
        success = self.check_success()
        return self.observation(), float(success), success, {}

    def close(self):
        self.closed = True


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.environments = []
        self.source = {"episode": {"suite": "libero_goal", "task_id": 0, "init_id": 0,
                       "task_name": "mock", "goal": "original goal", "seed": 31, "environment_seed": 0,
                       "settle_steps": 10, "max_steps": 300, "bddl_sha256": "mock", "init_file_sha256": "mock",
                       "initial_state_sha256": "mock"}, "episode_path": self.root / "episode.json",
                       "episode_sha256": "mock", "trace_path": self.root / "trace.jsonl",
                       "trace_sha256": "mock", "actions": np.zeros((100, 7))}
        self.patch_source = patch.object(replay, "_source", return_value=self.source)
        self.patch_factory = patch.object(replay, "_new_prefix_env", side_effect=self.factory)
        self.patch_source.start()
        self.patch_factory.start()

    def factory(self, source, prefix_steps):
        env = MockEnvironment()
        for action in source["actions"][:prefix_steps]:
            env.step(action.tolist())
        self.environments.append(env)
        return env, env.observation(), {"mock_source_error": 0.0}

    def tearDown(self):
        self.patch_source.stop()
        self.patch_factory.stop()
        self.assertTrue(all(env.closed for env in self.environments))
        self.tmp.cleanup()

    def run_replays(self, candidates, **kwargs):
        path = replay.replay_candidates(self.root / "episode.json", 16, candidates, self.root / "out", **kwargs)
        return json.loads(path.read_text())

    def test_zero_short_success_is_unknown_full_task_utility(self):
        actions = np.zeros((16, 7), dtype=np.float64)
        actions[:, -1] = .125  # Continuous gripper must survive verbatim.
        result = self.run_replays([actions, actions.copy()])
        self.assertEqual(len(self.environments), 4)
        self.assertTrue(result["repeat_passed"])
        self.assertFalse(result["full_task_utility_complete"])
        self.assertIsNone(result["candidates"]["0"]["success"])
        self.assertEqual(result["candidates"]["0"]["action_sha256"], replay.array_sha(actions))
        stored = np.load(result["candidates"]["0"]["actions"])
        np.testing.assert_array_equal(stored, actions)
        self.assertEqual(stored.dtype, actions.dtype)
        self.assertEqual(result["candidates"]["0"]["replays"][0]["last_executed_absolute_step"], 32)

    def test_fixed_original_goal_continuation_success_and_step_count(self):
        requests = []
        def continuation(obs, **kwargs):
            requests.append(kwargs)
            actions = np.zeros((16, 7))
            actions[:, 0] = .5
            return actions
        result = self.run_replays([np.zeros((16, 7))], continuation=continuation,
                                  continuation_metadata={"mock_policy": "fixed_original_goal"})
        row = result["candidates"]["0"]
        self.assertTrue(row["success"])
        self.assertEqual(row["steps_to_success"], 36)
        self.assertEqual(row["steps_to_success_from_candidate"], 20)
        self.assertTrue(result["full_task_utility_complete"])
        self.assertTrue(all(request["original_goal"] == "original goal" and request["seed"] == 31 for request in requests))

    def test_fixed16_forecast_kept_after_first_success(self):
        actions = np.zeros((16, 7)); actions[:, 0] = 1.0
        result = self.run_replays([actions])
        row = result["candidates"]["0"]["replays"][0]
        self.assertEqual(row["steps_to_success"], 18)
        self.assertEqual(row["last_executed_absolute_step"], 32)
        self.assertTrue(row["candidate_future_after_native_success_stop"])
        self.assertTrue(row["success"])

    def test_full_horizon_failure_has_no_fabricated_success_time(self):
        def continuation(obs, **kwargs):
            return np.zeros((16, 7))
        result = self.run_replays([np.zeros((16, 7))], continuation=continuation,
                                  continuation_metadata={"mock_policy": "fixed_zero"})
        row = result["candidates"]["0"]
        self.assertIs(row["success"], False)
        self.assertIsNone(row["steps_to_success"])
        self.assertEqual(row["replays"][0]["last_executed_absolute_step"], 300)
        self.assertTrue(result["full_task_utility_complete"])

    def test_changed_future_dynamics_rejected(self):
        calls = 0
        original_factory = self.factory
        def different_future(source, prefix):
            nonlocal calls
            env, obs, checks = original_factory(source, prefix)
            calls += 1
            if calls == 2:
                original_step = env.step
                def changed_step(action):
                    altered = list(action)
                    altered[0] += .001
                    return original_step(altered)
                env.step = changed_step
            return env, obs, checks
        with patch.object(replay, "_new_prefix_env", side_effect=different_future):
            with self.assertRaisesRegex(RuntimeError, "future or continuation terminal"):
                self.run_replays([np.zeros((16, 7))])

    def test_capture_schema_and_source_state_pairing(self):
        snapshot = replay.capture_source_prefix(self.root / "episode.json", 16, self.root / "capture")
        value = json.loads(snapshot.read_text())
        self.assertEqual(value["protocol"]["original_goal"], "original goal")
        self.assertEqual(value["state_sha256"], value["state_hash"])
        with np.load(value["observation_npz"]) as obs:
            self.assertTrue(set(replay.OBS_KEYS + ("sim_state",)).issubset(obs.files))
        result = self.run_replays([np.zeros((16, 7))], source_snapshot=snapshot)
        self.assertEqual(result["source_state_hash"], value["state_sha256"])

    def test_initial_state_mismatch_rejected(self):
        calls = 0
        original_factory = self.factory
        def different(source, prefix):
            nonlocal calls
            env, obs, checks = original_factory(source, prefix)
            calls += 1
            if calls == 2:
                env.position += .125
                obs = env.observation()
            return env, obs, checks
        with patch.object(replay, "_new_prefix_env", side_effect=different):
            with self.assertRaisesRegex(RuntimeError, "Initial simulator"):
                self.run_replays([np.zeros((16, 7))])
        result = json.loads((self.root / "out/utilities.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["repeat_passed"])

    def test_action_shapes_hash_dtype_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            replay._checked_actions(np.zeros((15, 7)), fixed16=True)
        with self.assertRaises(ValueError):
            replay._checked_actions(np.full((16, 7), np.nan), fixed16=True)
        self.assertNotEqual(replay.array_sha(np.zeros((16, 7), dtype=np.float64)),
                            replay.array_sha(np.zeros((16, 7), dtype=np.float32)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
