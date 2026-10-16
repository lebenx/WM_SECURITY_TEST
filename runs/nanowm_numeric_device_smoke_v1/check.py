"""CPU renderer regression check with the numeric CUDA device selector."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / ".deps/nanowm_runtime"))
from pilot_jepa_native_task_gate import configure_process
configure_process("cuda")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                  VAE_MODEL_PATH=str(ROOT / "models/nanowm_sd_vae"))
import numpy as np
import torch
import pilot_nanowm_heldout_clean as heldout
from pilot_jepa_native_task_gate import isolate_d4rl_offline_env, prepare_legacy_compiler

out = Path(__file__).resolve().parent
protocol = json.loads(heldout.PROTOCOL.read_text())
cfg, native, manifest, cache = heldout.prepare_config(SimpleNamespace(), out, protocol)
rows, _ = heldout.load_targets(protocol)
assert not torch.cuda.is_initialized()
isolate_d4rl_offline_env()
prepare_legacy_compiler()
env = native.create_planning_env("point_maze", device="cpu")
try:
    obs, state = env.prepare(rows[0]["seed"], np.asarray(rows[0]["init_state"], dtype=np.float32))
    assert np.array_equal(state, rows[0]["init_state"])
    assert np.array_equal(obs["visual"], rows[0]["_initial_image"])
    assert not torch.cuda.is_initialized()
    result = {"status": "passed", "cuda_visible_devices": "0", "cuda_initialized": False,
              "model_loaded": False, "model_rollout_calls": 0, "environment_steps": 0,
              "render_matches_frozen_initial_image": True, "state_matches_frozen_initial_state": True}
    heldout.base.write_json(out / "summary.json", result)
    print(json.dumps(result))
finally:
    env.close()
