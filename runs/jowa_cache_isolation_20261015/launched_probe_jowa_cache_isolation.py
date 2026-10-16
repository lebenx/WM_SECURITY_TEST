#!/usr/bin/env python3
"""Bounded CPU unit checks of upstream JOWA cache ownership, without weights."""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
from typing import Optional, Union

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/JOWA"
REVISION = "8cb02c5e6768a948c566a9012ac8a4fc420c6970"
OUTPUT = ROOT / "runs/jowa_cache_isolation_20261015"


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve the existing one-shot unit-test result")
    OUTPUT.mkdir(parents=True)
    report = {
        "status": "running_cpu_cache_unit_checks",
        "scope": "Upstream cache unit semantics and an unchanged AST-isolated step method with synthetic outputs; not model inference or an environment experiment",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
        "interpreter": sys.executable, "python": platform.python_version(),
        "checks": [], "model_weights_loaded": 0, "neural_model_forwards": 0,
        "task_environment_steps": 0, "network_requests": 0,
        "vendor_files_modified": 0,
    }

    def check(name, condition):
        passed = bool(condition)
        report["checks"].append({"name": name, "passed": passed})
        if not passed:
            raise AssertionError(name)

    try:
        import numpy as np
        import torch
        from torch.distributions.categorical import Categorical

        torch.set_num_threads(1)
        report.update(torch_version=torch.__version__, torch_path=torch.__file__,
                      numpy_version=np.__version__)
        check("source_revision_matches", report["source_revision"] == REVISION)
        path = VENDOR / "src/models/kv_caching.py"
        spec = importlib.util.spec_from_file_location("jowa_isolated_kv_caching", path)
        kv = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kv)
        report["cache_import"] = "Direct file import of src/models/kv_caching.py; no models package import"

        def snapshot(cache):
            return [(layer.shape, tuple(x.clone() for x in layer.get())) for layer in cache]

        def unchanged(cache, saved):
            return all(layer.shape == shape and all(torch.equal(x, y) for x, y in zip(layer.get(), tensors))
                       for layer, (shape, tensors) in zip(cache, saved))

        def append(cache, row_values, steps=1):
            values = torch.tensor(row_values, dtype=torch.float32).view(-1, 1, 1, 1)
            for index, layer in enumerate(cache):
                n, heads, _, width = layer.shape
                k = values.expand(n, heads, steps, width).clone() + index * 100
                layer.update(k, k + 1000)

        root = kv.KeysValues(1, 2, 12, 4, 2, torch.device("cpu"))
        append(root, [1], steps=2)
        root_saved = snapshot(root)
        alias_owner = deepcopy(root)
        alias_saved = snapshot(alias_owner)
        deliberate_alias = alias_owner
        append(deliberate_alias, [77])
        deliberate_alias[0].get()[0].fill_(-77)
        check("positive_control_deliberate_alias_changes_owner_prefix_and_size",
              deliberate_alias is alias_owner and alias_owner.size == 3
              and not unchanged(alias_owner, alias_saved)
              and not torch.equal(alias_owner[0].get()[0][:, :, :2], alias_saved[0][1][0]))
        report["positive_control"] = "Deliberately shared reference changes its owner; validates detector sensitivity only, not an upstream aliasing flaw"
        child = deepcopy(root)
        append(child, [2])
        child[0].get()[0].fill_(7)
        check("deepcopy_child_append_and_prefix_write_preserve_root", unchanged(root, root_saved))
        check("deepcopy_has_distinct_storage", all(
            a.data_ptr() != b.data_ptr()
            for left, right in zip(root, child) for a, b in zip(left.get(), right.get())))
        check("deepcopy_updates_child_size_only", root.size == 2 and child.size == 3)

        parent_b = deepcopy(root)
        for layer in parent_b:
            for value in layer.get():
                value.add_(5)
        parent_b_saved = snapshot(parent_b)
        joined = kv.concate_kv([root, parent_b], repeat_num=2)
        check("concate_preserves_prefix_order_and_repetition", all(
            torch.equal(value, torch.cat([a.repeat(2, 1, 1, 1), b.repeat(2, 1, 1, 1)], dim=0))
            for index, layer in enumerate(joined)
            for value, a, b in zip(layer.get(), root[index].get(), parent_b[index].get())))
        append(joined, [11, 12, 21, 22])
        check("concate_append_preserves_both_parent_prefixes_and_sizes",
              unchanged(root, root_saved) and unchanged(parent_b, parent_b_saved))
        check("concate_mutates_only_new_batch_size", joined.size == 3 and joined[0].shape[0] == 4)

        siblings = kv.split_kv(joined)
        joined_saved = snapshot(joined)
        sibling_saved = [snapshot(sibling) for sibling in siblings]
        check("split_preserves_each_row_and_size", all(
            sibling.size == joined.size and all(
                torch.equal(value, original[i:i+1])
                for index, layer in enumerate(sibling)
                for value, original in zip(layer.get(), joined[index].get()))
            for i, sibling in enumerate(siblings)))
        append(siblings[0], [99])
        siblings[0][0].get()[1].fill_(-50)
        check("split_child_write_preserves_other_siblings",
              all(unchanged(sibling, saved) for sibling, saved in zip(siblings[1:], sibling_saved[1:])))
        check("split_child_write_preserves_batch_and_root",
              unchanged(joined, joined_saved) and unchanged(root, root_saved))

        # Keep the old backing tensors to detect storage aliasing after pruning.
        old_views = [layer.get() for layer in joined]
        old_values = [tuple(value.clone() for value in pair) for pair in old_views]
        joined.prune(np.asarray([True, False, True, False]))
        check("prune_keeps_selected_rows_and_sequence_length", joined.size == 3 and all(
            torch.equal(value, original[[0, 2]])
            for layer, originals in zip(joined, old_values)
            for value, original in zip(layer.get(), originals)))
        append(joined, [31, 32])
        joined[0].get()[0].fill_(-75)
        check("pruned_write_preserves_previous_batch_storage", all(
            torch.equal(view, original) for views, originals in zip(old_views, old_values)
            for view, original in zip(views, originals)))
        check("pruned_write_preserves_roots_and_surviving_siblings",
              unchanged(root, root_saved) and unchanged(parent_b, parent_b_saved)
              and all(unchanged(sibling, saved) for sibling, saved in zip(siblings[1:], sibling_saved[1:])))

        buffer = torch.arange(8, dtype=torch.long).view(1, -1)
        buffer_saved = buffer.clone()
        obs = buffer[:, -2:]
        check("initial_observation_slice_is_a_view",
              obs.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr())
        obs = torch.cat([torch.tensor([[9]]), torch.tensor([[10]])], dim=1)
        obs.fill_(42)
        check("observation_cat_rebinding_and_later_write_preserve_real_buffer", torch.equal(buffer, buffer_saved))

        # Compile the exact upstream method AST without importing its heavy
        # tokenizer/environment dependencies or changing its body/decorators.
        step_path = VENDOR / "src/envs/world_model_env.py"
        module_ast = ast.parse(step_path.read_text(), filename=str(step_path))
        cls = next(node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == "WorldModelEnv")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
        namespace = dict(torch=torch, np=np, Optional=Optional, Union=Union, Categorical=Categorical)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(step_path), "exec"), namespace)

        class SyntheticTokenProducer:
            """Deterministic tensors and real KV updates; no learned parameters."""
            def __init__(self):
                self.config_transformer = SimpleNamespace(max_tokens=12)
                self.calls = 0

            def to(self, device):
                check("synthetic_producer_cpu_only", torch.device(device).type == "cpu")
                return self

            def __call__(self, token, env_tokens, past_keys_values):
                self.calls += 1
                append(past_keys_values, [int(token.item())])
                observations = torch.zeros(1, 1, 16)
                observations[..., 8 + self.calls] = 10
                return SimpleNamespace(output_sequence=torch.ones(1, 1, 2),
                    logits_rewards=torch.tensor([[[0., 10., 0.]]]),
                    logits_ends=torch.tensor([[[10., 0.]]]),
                    logits_observations=observations, logits_q=torch.tensor([[[0., 1.]]]))

        producer = SyntheticTokenProducer()
        holder = SimpleNamespace(device=torch.device("cpu"), jowa=producer,
            keys_values_wm=root, num_observations_tokens=2,
            env_tokens=torch.tensor([0]), obs_tokens=buffer[:, -2:], last_actions=None)
        holder.choose_action = lambda critic, **kwargs: critic[:, -1].argmax(-1)
        holder.decode_obs_tokens = lambda: holder.obs_tokens.clone()
        explicit_cache = deepcopy(root)
        observation, reward, done, info = namespace["step"](
            holder, action=torch.tensor([[3]]), should_sample=False,
            act_mode="logits", kv_cache=explicit_cache)
        check("real_step_advances_explicit_cache_by_action_plus_observation_tokens",
              explicit_cache.size == 5 and producer.calls == 3)
        check("real_step_does_not_rebind_or_advance_member_root_cache",
              holder.keys_values_wm is root and unchanged(root, root_saved))
        check("real_step_rebinds_observation_without_writing_real_buffer",
              torch.equal(buffer, buffer_saved) and holder.obs_tokens.tolist() == [[9, 10]])
        check("real_step_returns_synthetic_generated_observation",
              torch.equal(observation, holder.obs_tokens) and info["obs_tokens"] is holder.obs_tokens)
        saved_like_mcts = deepcopy(holder.keys_values_wm)
        check("mcts_member_cache_save_would_save_stale_prefix",
              saved_like_mcts.size == 2 and explicit_cache.size == 5
              and unchanged(saved_like_mcts, root_saved))
        report["step_probe"] = {
            "source": "vendor/JOWA/src/envs/world_model_env.py", "line": method.lineno,
            "body": "Unmodified upstream AST; original torch.no_grad decorator retained",
            "substitutions": "Synthetic parameter-free tensor producer, action readout and token-copy decoder; real upstream cache implementation",
            "synthetic_producer_calls": producer.calls,
            "member_cache_tokens": holder.keys_values_wm.size,
            "explicit_cache_tokens": explicit_cache.size,
            "saved_like_mcts_tokens": saved_like_mcts.size,
        }
        check("cuda_never_initialized", not torch.cuda.is_initialized())
        report.update(status="passed_cpu_cache_unit_checks", cuda_initialized=False,
            interpretation="The checked branch-copy operations isolate root/sibling valid prefixes and sizes. Explicit-cache step updates do not update the member cache; current MCTS call-site therefore saves a stale child prefix. No model predictions, task utility, malicious interface or security harm were tested.",
            limitations=[
                "Small synthetic cache tensors; no checkpoint, neural inference, Atari environment, full Agent/MCTS execution or concurrency test.",
                "Only valid prefixes and sequence sizes are compared; unused torch.empty capacity is not meaningful data.",
                "The stale-child observation is an implementation/call-site correctness issue, not rejected-branch contamination of real history.",
                "Observation ownership is checked for the source's view-then-cat-rebinding operations; arbitrary external in-place edits are not an identified native interface."])
    except Exception as error:
        report.update(status="failed_cpu_cache_unit_checks", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["check_count"] = len(report["checks"])
        (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "interpreter", "torch_version", "check_count", "step_probe", "cuda_initialized")}, indent=2))


if __name__ == "__main__":
    main()
