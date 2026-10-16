#!/usr/bin/env python3
"""Fixed CPU numerical affine-boundary qualification; no environment or planner."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time
import traceback
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/jepa_affine_query_boundary_20261016.json"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main(config_path, output):
    protocol = json.loads(config_path.read_text())
    out = (output or ROOT / protocol["output"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "launched_config.json").write_bytes(config_path.read_bytes())
    (out / "launched_script.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    counts = {key: 0 for key in protocol["max_activity"]}
    completed = {key: 0 for key in ("encode_calls", "unroll_calls", "visual_encoder_forwards",
                                    "proprio_encoder_forwards", "predictor_forwards")}
    residual_arrays = {}
    report = {"status": "starting", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
              "scope": protocol["purpose"], "prior_inspection_disclosure": protocol["prior_inspection_disclosure"],
              "activity": counts, "completed_calls": completed, "checkpoint_files": [],
              "roots": [], "qualification_gate": None,
              "actual_utility": None, "attack_success": None, "native_grasp_queries": 0}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        temp = out / "summary.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(out / "summary.json")

    def check_time():
        if time.monotonic() - started > protocol["hard_wall_seconds"]:
            raise TimeoutError("Frozen 180-second CPU probe budget exceeded")

    def stop(*_):
        raise TimeoutError("Frozen 180-second CPU probe budget exceeded")

    def increment(key):
        check_time()
        counts[key] += 1
        require(counts[key] <= protocol["max_activity"][key], "Activity budget exceeded: " + key)

    def forbidden(*_, **__):
        counts["network_attempts"] += 1
        raise RuntimeError("Network access is forbidden in this local probe")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, protocol["hard_wall_seconds"])
    persist()
    try:
        require(protocol["roots"] == [1, 2, 3] and protocol["device"] == "cpu"
                and protocol["threads"] == 4 and protocol["hard_wall_seconds"] == 180,
                "Frozen roots/runtime changed")
        require(protocol["affine_constraints"]["tolerance"] == 1e-5
                and protocol["virtual_control"]["sigma"] == 0.01
                and protocol["virtual_control"]["seed"] == 20261016,
                "Frozen numerical controls changed")
        import numpy as np
        import torch
        from tensordict import TensorDict
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        random.seed(protocol["global_seed"])
        np.random.seed(protocol["global_seed"])
        torch.manual_seed(protocol["global_seed"])
        require(not torch.cuda.is_initialized(), "CUDA was initialized")
        report["runtime"] = {"python": sys.version, "executable": sys.executable,
                             "torch": torch.__version__, "numpy": np.__version__,
                             "threads": torch.get_num_threads(), "device": "cpu"}
        sys.path.insert(0, str(ROOT / "vendor/jepa-wms"))
        import pilot_dino_metaworld_clean_gate as helper
        import pilot_jepa_metaworld_reach_wall as wall
        original_load = torch.load
        allowed = {wall.MODEL.resolve(), helper.DINO_WEIGHTS.resolve()}
        loaded = set()

        def local_load(path, *args, **kwargs):
            resolved = Path(path).resolve()
            require(resolved in allowed and resolved not in loaded, "Unexpected/repeated checkpoint load")
            increment("checkpoint_loads")
            loaded.add(resolved)
            report["checkpoint_files"].append({"path": str(resolved), "bytes": resolved.stat().st_size})
            return original_load(path, *args, **kwargs)

        def forward_pre(key):
            return lambda *_: increment(key)

        def forward_post(key):
            def record(*_):
                completed[key] += 1
                check_time()
            return record

        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, "connect", side_effect=forbidden))
            guards.enter_context(patch.object(socket.socket, "connect_ex", side_effect=forbidden))
            guards.enter_context(patch.object(socket, "create_connection", side_effect=forbidden))
            guards.enter_context(patch.object(torch.hub, "download_url_to_file", side_effect=forbidden))
            guards.enter_context(patch.object(torch, "load", side_effect=local_load))
            guards.enter_context(patch.object(helper, "MODEL", wall.MODEL))
            cfg = wall.cfg_for(helper, out, "cpu")
            require(not cfg.model_kwargs.pretrain_kwargs.wm_encoding.normalize_reps,
                    "Additional representation normalization invalidates this derivation")
            report["status"] = "loading_local_models"
            persist()
            model, _, report["strict_load"] = helper.load_model(cfg, "cpu")
            require(loaded == allowed, "Expected both local checkpoint loads")
            require(not model.normalize_reps and model.proprio_mode == "predict_proprio"
                    and model.model.pred_type == "AdaLN" and model.model.proprio_encoding == "feature"
                    and model.action_dim == 20 and model.model.encoder.feature_key == "x_norm_patchtokens",
                    "Native joint-output contract differs")
            report["native_contract"] = {"predictor": model.model.pred_type, "normalize_reps": model.normalize_reps,
                                         "proprio_mode": model.proprio_mode, "grid_size": model.grid_size,
                                         "action_dim": model.action_dim, "proprio_encoding": model.model.proprio_encoding}
            for module, key in ((model.model.encoder, "visual_encoder_forwards"),
                                (model.model.proprio_encoder, "proprio_encoder_forwards"),
                                (model.model.predictor, "predictor_forwards")):
                guards.callback(module.register_forward_pre_hook(forward_pre(key)).remove)
                guards.callback(module.register_forward_hook(forward_post(key)).remove)

            original_encode, original_unroll = model.encode, model.unroll
            def encoded(*args, **kwargs):
                increment("encode_calls")
                value = original_encode(*args, **kwargs)
                completed["encode_calls"] += 1
                check_time()
                return value

            def unrolled(*args, **kwargs):
                increment("unroll_calls")
                value = original_unroll(*args, **kwargs)
                completed["unroll_calls"] += 1
                check_time()
                return value

            guards.enter_context(patch.object(model, "encode", side_effect=encoded))
            guards.enter_context(patch.object(model, "unroll", side_effect=unrolled))
            encoder_norm = model.model.encoder.base_model.norm
            predictor_norm = model.model.predictor.predictor_norm
            projection = model.model.predictor.predictor_proj
            require(isinstance(encoder_norm, torch.nn.LayerNorm) and encoder_norm.normalized_shape == (384,)
                    and isinstance(predictor_norm, torch.nn.LayerNorm) and predictor_norm.normalized_shape == (400,)
                    and tuple(projection.weight.shape) == (384, 384), "LayerNorm/projection dimensions differ")
            to64 = lambda x: x.detach().cpu().to(torch.float64)
            eg, eb = to64(encoder_norm.weight), to64(encoder_norm.bias)
            pg, pb = to64(predictor_norm.weight), to64(predictor_norm.bias)
            w, wb = to64(projection.weight), to64(projection.bias)

            def constraint(matrix, name):
                require(bool(torch.isfinite(matrix).all()), "Nonfinite constraint matrix")
                u, singular, _ = torch.linalg.svd(matrix, full_matrices=True)
                normal = u[:, -1]
                normal = normal / torch.linalg.vector_norm(normal)
                if normal[normal.abs().argmax()] < 0:
                    normal = -normal
                null_absolute = torch.linalg.vector_norm(normal @ matrix)
                null_relative = null_absolute / singular[0].clamp_min(torch.finfo(torch.float64).tiny)
                require(float(null_relative) <= protocol["affine_constraints"]["null_relative_residual_max"],
                        name + " does not have the required numerical left-null direction")
                return normal, singular, {"normal_l2": float(normal.norm()),
                                          "null_absolute": float(null_absolute), "null_relative": float(null_relative),
                                          "largest_singular_value": float(singular[0]),
                                          "smallest_singular_value": float(singular[-1])}

            center384 = torch.eye(384, dtype=torch.float64) - torch.ones(384, 384, dtype=torch.float64) / 384
            center400 = torch.eye(400, dtype=torch.float64) - torch.ones(400, 400, dtype=torch.float64) / 400
            block = torch.block_diag(w, torch.eye(16, dtype=torch.float64))
            en, es, emeta = constraint(torch.diag(eg) @ center384, "encoder")
            pn, ps, pmeta = constraint(block @ torch.diag(pg) @ center400, "predictor")
            poffset = torch.cat((w @ pb[:384] + wb, pb[384:]))
            require(all(bool(torch.isfinite(x).all()) for x in (en, pn, eb, poffset)), "Nonfinite affine parameters")
            np.savez_compressed(out / "constraints.npz", encoder_normal=en.numpy(), encoder_offset=eb.numpy(),
                                encoder_gamma=eg.numpy(), encoder_singular_values=es.numpy(),
                                predictor_normal=pn.numpy(), predictor_offset=poffset.numpy(),
                                predictor_gamma=pg.numpy(), predictor_beta=pb.numpy(),
                                predictor_visual_weight=w.numpy(), predictor_visual_bias=wb.numpy(),
                                predictor_singular_values=ps.numpy())
            report["constraints"] = {"encoder": emeta, "predictor": pmeta, "arithmetic": "float64",
                                     "residual_shape_rule": "batch,time,patch", "tolerance": 1e-5}
            virtual_rng = torch.Generator(device="cpu").manual_seed(20261016)

            def residuals(visual, proprio):
                require(visual.dtype == proprio.dtype == torch.float32 and visual.device.type == proprio.device.type == "cpu",
                        "Expected native CPU FP32 features")
                require(visual.ndim == 6 and proprio.ndim == 4 and visual.shape[-1] == 384
                        and proprio.shape[-1] == 16, "Unexpected feature shape")
                b, t = visual.shape[:2]
                v = visual.reshape(b, t, -1, 384).double()
                p = proprio.double()
                require(v.shape[:3] == p.shape[:3] and bool(torch.isfinite(v).all())
                        and bool(torch.isfinite(p).all()), "Patch alignment/nonfinite features")
                re = (v - eb) @ en
                rp = (torch.cat((v, p), dim=-1) - poffset) @ pn
                union = torch.minimum(re.abs(), rp.abs())
                require(bool(torch.isfinite(union).all()), "Nonfinite residual")
                return {"encoder": re.numpy(), "predictor": rp.numpy(), "union": union.numpy()}

            def stats(values):
                flat = values.reshape(-1)
                return {"count": int(flat.size), "max_abs": float(np.abs(flat).max()),
                        "rms": float(np.sqrt(np.mean(np.square(flat)))),
                        "above_tolerance": int((np.abs(flat) > 1e-5).sum()),
                        "fraction_above_tolerance": float((np.abs(flat) > 1e-5).mean()),
                        "at_or_above_tolerance": int((np.abs(flat) >= 1e-5).sum())}

            with torch.inference_mode():
                for seed in protocol["roots"]:
                    source = ROOT / protocol["source_pattern"].format(seed=seed)
                    record = {"seed": seed, "source": str(source.relative_to(ROOT)), "status": "encoding"}
                    report["roots"].append(record)
                    report["status"] = "running_fixed_panel"
                    persist()
                    with np.load(source, allow_pickle=False) as data:
                        images = [data[key].copy() for key in ("initial_pixels", "goal_pixels")]
                        proprios = [data[key].copy() for key in ("initial_proprio", "goal_proprio")]
                    require(all(x.shape == (1, 3, 224, 224) and x.dtype == np.uint8 for x in images),
                            "Cached image schema differs")
                    require(all(x.shape == (1, 4) and x.dtype == np.float32 and np.isfinite(x).all() for x in proprios),
                            "Cached proprio schema differs")
                    obs = TensorDict({"visual": torch.from_numpy(np.stack(images)),
                                      "proprio": torch.from_numpy(np.stack(proprios))})
                    z = model.encode(obs, act=True)
                    real = residuals(z["visual"], z["proprio"])
                    initial = TensorDict({key: z[key][0:1].clone() for key in ("visual", "proprio")})
                    actions = torch.zeros((3, 2, 20), dtype=torch.float32)
                    actions[:, 1] = 0.2
                    record["status"] = "unrolling"
                    persist()
                    forecast = model.unroll(initial, act_suffix=actions)
                    require(forecast["visual"].shape[:2] == forecast["proprio"].shape[:2] == (4, 2),
                            "Native H3/B2 output timing differs")
                    require(torch.equal(forecast["visual"][0], initial["visual"][:, 0].expand(2, -1, -1, -1, -1))
                            and torch.equal(forecast["proprio"][0], initial["proprio"][:, 0].expand(2, -1, -1)),
                            "Returned context prefix differs")
                    rollout = residuals(forecast["visual"].transpose(0, 1), forecast["proprio"].transpose(0, 1))
                    noise = torch.randn(initial["visual"].shape, generator=virtual_rng, dtype=torch.float32) * 0.01
                    virtual = residuals(initial["visual"] + noise, initial["proprio"])
                    successor_visual = forecast["visual"][1:].transpose(0, 1)
                    hybrid_proprio = initial["proprio"].expand(2, 3, -1, -1)
                    hybrid = residuals(successor_visual, hybrid_proprio)
                    record["hybrid_label"] = "native_style_hybrid_not_actual_GRASP_execution"
                    for group, values in (("real", real), ("rollout", rollout), ("virtual", virtual),
                                          ("native_style_hybrid", hybrid)):
                        record[group] = {kind: stats(array) for kind, array in values.items()}
                        for kind, array in values.items():
                            residual_arrays[f"seed{seed}_{group}_{kind}"] = array
                    residual_arrays[f"seed{seed}_normalized_actions"] = actions.numpy()
                    record["feature_shapes"] = {"real_visual": list(z["visual"].shape),
                                                "real_proprio": list(z["proprio"].shape),
                                                "rollout_visual": list(forecast["visual"].shape),
                                                "rollout_proprio": list(forecast["proprio"].shape)}
                    record["status"] = "completed"
                    np.savez_compressed(out / "residuals.npz", **residual_arrays)
                    persist()
                    print(json.dumps({"seed": seed, "real_union_max": record["real"]["union"]["max_abs"],
                                      "rollout_union_max": record["rollout"]["union"]["max_abs"],
                                      "virtual_above_tolerance": record["virtual"]["union"]["above_tolerance"],
                                      "hybrid_above_tolerance": record["native_style_hybrid"]["union"]["above_tolerance"]}), flush=True)
            require(counts == protocol["max_activity"], "Incomplete or unexpected fixed activity")
            require(all(completed[key] == counts[key] for key in completed), "A forward did not complete")
            require(not torch.cuda.is_initialized(), "CUDA was initialized")
            real_max = max(r["real"]["union"]["max_abs"] for r in report["roots"])
            rollout_max = max(r["rollout"]["union"]["max_abs"] for r in report["roots"])
            virtual_count = sum(r["virtual"]["union"]["above_tolerance"] for r in report["roots"])
            hybrid_count = sum(r["native_style_hybrid"]["union"]["above_tolerance"] for r in report["roots"])
            virtual_total = sum(r["virtual"]["union"]["count"] for r in report["roots"])
            hybrid_total = sum(r["native_style_hybrid"]["union"]["count"] for r in report["roots"])
            report["qualification_gate"] = {"real_union_max": real_max, "rollout_union_max": rollout_max,
                                            "virtual_patches_above_tolerance": virtual_count,
                                            "virtual_fraction_above_tolerance": virtual_count / virtual_total,
                                            "native_style_hybrid_patches_above_tolerance": hybrid_count,
                                            "native_style_hybrid_fraction_above_tolerance": hybrid_count / hybrid_total,
                                            "real_passed": real_max < 1e-5, "rollout_passed": rollout_max < 1e-5,
                                            "virtual_signal_present": virtual_count > 0,
                                            "native_style_hybrid_signal_present": hybrid_count > 0,
                                            "synthetic_consumer_signal_present": virtual_count + hybrid_count > 0,
                                            "passed": real_max < 1e-5 and rollout_max < 1e-5 and virtual_count + hybrid_count > 0}
            report["status"] = "completed_affine_boundary_numerical_probe"
            report["interpretation"] = protocol["qualification_gate"]["interpretation"]
            check_time()
            persist()
    except BaseException as exc:
        report["status"] = "error"
        report["exception_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["qualification_gate"] = None
        if residual_arrays:
            np.savez_compressed(out / "residuals.npz", **residual_arrays)
        persist()
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.config.resolve(), args.output)
