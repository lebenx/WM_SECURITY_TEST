#!/usr/bin/env python3
"""CPU routing probe of two upstream methods; no learned model or physics."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import traceback
from types import MethodType, SimpleNamespace
from typing import Any, Dict


REVISION = "d644309db5d46d414ac70b18bed085b030ff9e5c"
DEFAULT_SOURCE = Path("/tmp/latent_cbf_readonly_d644309/src/latent_cbf/controllers/diffusion_controller.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=Path("runs/latent_cbf_action_feedback_20261015"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {
        "status": "started",
        "kind": "original_method_routing_probe_with_recording_stubs",
        "revision": REVISION,
        "upstream_path": "src/latent_cbf/controllers/diffusion_controller.py",
        "scope": "Synthetic observation IDs and a stub nominal controller/WM/critic; no pretrained inference, latent-error estimate, safety outcome or physical utility.",
        "fixture": {"nominal": 0.5, "forced_filtered_control": -1.0, "executor_bounds": [-2.0, 2.0], "limit": "Both fixture controls are strictly within the native bounds, so final clipping would preserve them. No physical environment is executed. In general the correct feedback must come after executor clipping; missing real ac_max files prevent assuming every filtered return equals the applied control."},
        "activity": {"real_model_loads": 0, "real_model_inference_calls": 0, "physical_environment_steps": 0, "network_calls": 0, "training_updates": 0},
    }
    try:
        import numpy as np
        import torch

        source = args.source.read_bytes()
        (args.out / "upstream_diffusion_controller.py").write_bytes(source)
        shutil.copyfile(__file__, args.out / "launched_probe.py")
        summary["source_sha256"] = hashlib.sha256(source).hexdigest()
        summary["runtime"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "torch_path": torch.__file__}
        tree = ast.parse(source.decode())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FilteredDiffusionController")
        names = {"compute_action", "reset"}
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in methods} != names:
            raise RuntimeError("Required upstream methods are missing")
        summary["extracted_original_methods"] = {n.name: [n.lineno, n.end_lineno] for n in methods}
        namespace = {"np": np, "torch": torch, "Dict": Dict, "Any": Any}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(args.source), "exec"), namespace)

        class NominalController:
            ac_max = np.array([2.0])

            def __init__(self):
                self.calls = 0

            def compute_action(self, info, observation):
                self.calls += 1
                return 0.5

            def reset(self):
                self.calls = 0

        class RecordingWM:
            """Records observe inputs; returned features have no predictive meaning."""

            def __init__(self):
                self.windows = []
                self.dynamics = self

            def preprocess(self, data):
                return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

            def encoder(self, data):
                return data["image"][:, :, 0, 0, :1]

            def observe(self, embed, action, is_first):
                self.windows.append({
                    "observation_ids": embed.flatten().tolist(),
                    "incoming_actions": action.flatten().tolist(),
                    "is_first": is_first.flatten().tolist(),
                })
                return torch.zeros((1, action.shape[1], 2)), None

            def get_feat(self, state):
                return state

        scenarios = {}
        for label, reject_once, write_back in [
            ("accepted_nominal", False, False),
            ("one_filtered_action_native_history", True, False),
            ("one_filtered_action_with_feedback_writeback", True, True),
        ]:
            base, wm = NominalController(), RecordingWM()
            holder = SimpleNamespace(controller=base, wm=wm, wm_config=SimpleNamespace(filter_mode="cbf", cbf_gamma=0.5))
            holder.compute_action = MethodType(namespace["compute_action"], holder)
            holder.reset = MethodType(namespace["reset"], holder)
            critic_calls = []

            def critic(state, actions):
                # Force exactly one routing event at the first eligible filter call.
                # This is not a learned safety score or an unsafe-action assertion.
                q = np.ones(len(actions))
                if reject_once and base.calls == 7:
                    q[:] = 0.0
                    q[7] = 1.0  # Native grid action: normalized -0.5 -> request -1.
                critic_calls.append({"nominal_call": base.calls, "input_actions": actions.flatten().tolist(), "forced_nominal_rejection": bool(reject_once and base.calls == 7)})
                return q

            holder.eval_Q = critic
            holder.reset()
            trace = []
            for t in range(8):
                selected, nominal = holder.compute_action({"agent_orientation": float(t)}, np.full((1, 1, 3), t, dtype=np.uint8))
                selected = float(np.asarray(selected).item())
                nominal = float(np.asarray(nominal).item())
                trace.append({"call_t": t, "nominal": nominal, "returned_control": selected})
                if write_back:
                    # Only test this one feedback assignment; do not change the method.
                    # Both synthetic returned controls are in [-2,2]. No env is run.
                    holder.action_history[-1] = selected
            saved_history = list(holder.action_history)
            holder.reset()
            reset_empty = not holder.action_history and not holder.observation_history and not holder.state_history and base.calls == 0
            scenarios[label] = {"trace": trace, "observe_windows": wm.windows, "critic_calls": critic_calls, "history_before_reset": saved_history, "reset_empty": reset_empty}

        normal, native, repaired = (scenarios[k] for k in scenarios)
        normal_alignment = all(w["incoming_actions"] == [normal["trace"][int(o)-1]["returned_control"] for o in w["observation_ids"]] for w in normal["observe_windows"])
        nnext, rnext = native["observe_windows"][-1], repaired["observe_windows"][-1]
        native_mismatch = native["trace"][6]["returned_control"] == -1.0 and nnext["incoming_actions"][-1] == 0.5 and nnext["is_first"][-1] == 0.0
        repaired_alignment = rnext["incoming_actions"][-1] == repaired["trace"][6]["returned_control"] == -1.0 and rnext["incoming_actions"][:-1] == nnext["incoming_actions"][:-1] and rnext["observation_ids"] == nnext["observation_ids"]
        reset_empty = all(s["reset_empty"] for s in scenarios.values())
        checks = {"accepted_nominal_history_matches_returned_controls": normal_alignment, "after_one_filter_next_observe_receives_unexecuted_nominal": native_mismatch, "one_feedback_assignment_repairs_the_corresponding_noninitial_slot_only": repaired_alignment, "original_reset_clears_three_histories_and_calls_stub_base_reset": reset_empty}
        summary.update({"scenarios": scenarios, "checks": checks})
        assert all(checks.values()), checks
        assert not torch.cuda.is_initialized(), "CUDA must remain uninitialized"
        summary["activity"].update({"original_compute_action_calls_on_stubs": 24, "recording_observe_calls": 6, "stub_critic_calls": 6, "cuda_initialized": False})
        summary["status"] = "completed_cpu_route_semantics_only"
    except Exception:
        summary["status"] = "failed_preserved_no_retry"
        summary["error"] = traceback.format_exc()
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ("status", "elapsed_seconds")}), flush=True)
    return 0 if summary["status"] == "completed_cpu_route_semantics_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
