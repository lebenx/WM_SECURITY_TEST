"""Bounded DMO-network derivative screen; no actor, shared protocol, or attack claim.

Real DMControl random-policy transitions train the official StochSSM. Only its
last mean layer is perturbed; subsequent NLL refitting updates the whole model.
Selection uses dev only. Test labels and finite differences never select a head.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/DMO_code"
SIZES = {"train": 4096, "dev": 1024, "test": 2048, "refit": 2048}
LAMBDAS = [1.0, 10.0, 100.0, 1000.0]
RIDGE = 1e-6
DRIFT_RTOL, DRIFT_ATOL = 1e-3, 1e-6
CONFIG = {"dyn_model_mlp": {"units": [512, 512], "activation": "silu",
                            "recurrent": False}}
SCOPE = ("Necessary-condition screen of an official DMO network on DMControl "
         "cartpole-balance random-policy data. No DFlex actor, policy damage, "
         "poisoned-data reachability, or complete native DMO attack evidence.")


class DeadlineExceeded(RuntimeError):
    pass


def sha_array(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def tensor_hashes(model):
    return {k: sha_array(v.detach().cpu().numpy()) for k, v in model.state_dict().items()}


def flatten(timestep):
    return np.concatenate([np.asarray(v).ravel() for v in timestep.observation.values()])


def heldout_rotation_matching(branches, traces):
    """Validate the dev-selected rotation on test; never select or alter it."""
    directed = branches["directed_before"]
    random = branches["random_before"]
    mse_differences = {
        key: abs(random[key] - directed[key]) / max(directed[key], 1e-30)
        for key in ("center_mse", "neighbor_mse")}
    pointwise = {}
    for key in ("predicted_delta", "predicted_delta_minus", "predicted_delta_plus"):
        clean_prediction = traces["clean_before"][key].astype(np.float64)
        dnorm = np.linalg.norm(traces["directed_before"][key].astype(np.float64) - clean_prediction, axis=1)
        rnorm = np.linalg.norm(traces["random_before"][key].astype(np.float64) - clean_prediction, axis=1)
        difference = np.abs(rnorm - dnorm)
        close = difference <= DRIFT_ATOL + DRIFT_RTOL * dnorm
        pointwise[key] = {
            "points": int(len(dnorm)), "all_matched": bool(close.all()),
            "fraction_matched": float(close.mean()),
            "max_abs_norm_difference": float(difference.max()),
            "max_relative_norm_difference": float((difference / np.maximum(dnorm, 1e-12)).max()),
            "directed_median_drift_norm": float(np.median(dnorm)),
            "random_median_drift_norm": float(np.median(rnorm))}
    truth_matched = all(value <= 0.1 for value in mse_differences.values())
    drift_matched = all(value["all_matched"] for value in pointwise.values())
    return {"split": "test", "used_for_selection": False,
            "relative_truth_mse_differences": mse_differences,
            "truth_mse_matched": truth_matched,
            "pointwise_forward_drift": pointwise,
            "drift_tolerance": {"rtol": DRIFT_RTOL, "atol": DRIFT_ATOL},
            "pointwise_drift_matched": drift_matched,
            "matched": bool(truth_matched and drift_matched)}


def collect(seed, count, finite_difference=False):
    from dm_control import suite
    import mujoco

    env = suite.load("cartpole", "balance", task_kwargs={"random": seed})
    rng = np.random.default_rng(seed + 10000)
    ts = env.reset()
    sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    spec = env.action_spec()
    rows = {k: [] for k in ("obs", "action", "next_obs", "reward", "physics_state", "step_count")}
    if finite_difference:
        rows.update({k: [] for k in ("action_minus", "action_plus", "next_minus", "next_plus", "true_jacobian")})
    restore_error = 0.0

    def snapshot():
        return (env.physics.get_state(sig=sig).copy(), env._step_count, env._reset_next_step)

    def restore(state):
        env.physics.set_state(state[0], sig=sig)
        env.physics.forward()
        # Reinstall integration variables such as warm-start after forward.
        env.physics.set_state(state[0], sig=sig)
        env._step_count, env._reset_next_step = state[1], state[2]

    try:
        for i in range(count):
            # Fixed collection horizon, unrelated to any learned controller.
            if ts.last() or (i > 0 and i % 200 == 0):
                ts = env.reset()
            s = flatten(ts)
            a = rng.uniform(-0.9, 0.9, size=spec.shape).astype(spec.dtype)
            before = snapshot()
            ts = env.step(a)
            y = flatten(ts)
            after = snapshot()
            for key, val in (("obs", s), ("action", a), ("next_obs", y),
                             ("reward", ts.reward), ("physics_state", before[0]),
                             ("step_count", before[1])):
                rows[key].append(val)
            if finite_difference:
                lo = np.clip(a - 0.01, spec.minimum, spec.maximum)
                hi = np.clip(a + 0.01, spec.minimum, spec.maximum)
                restore(before)
                ylo = flatten(env.step(lo))
                restore(before)
                yhi = flatten(env.step(hi))
                if i < 16:
                    restore(before)
                    repeated = flatten(env.step(a))
                    restore_error = max(restore_error, float(np.max(np.abs(repeated - y))))
                restore(after)
                for key, val in (("action_minus", lo), ("action_plus", hi),
                                 ("next_minus", ylo), ("next_plus", yhi),
                                 ("true_jacobian", (yhi - ylo) / (hi - lo)[0])):
                    rows[key].append(val)
    finally:
        env.close()
    data = {k: np.asarray(v) for k, v in rows.items()}
    if restore_error > 1e-9:
        raise AssertionError(f"Physics restore failed: maximum repeat error {restore_error}")
    return data, {"seed": seed, "samples": count, "action_seed": seed + 10000,
                  "episode_collection_cap": 200, "state_signature": sig,
                  "observation_keys": list(ts.observation),
                  "restore_repeat_max_abs_error": restore_error,
                  "hashes": {k: sha_array(v) for k, v in data.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--output", default="runs/dmo_jacobian_gate_v0")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 120:
        parser.error("--max-seconds must be positive and at most 120")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory to preserve prior evidence")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    compute_started = None
    history, result = [], {"scope": SCOPE, "selection": None, "branches": {}}
    phase = "setup"

    def save(name, data):
        dest = out / name
        temp = dest.with_suffix(dest.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temp.replace(dest)

    def alarm_handler(_signum, _frame):
        raise DeadlineExceeded(f"Compute wall budget {args.max_seconds}s exhausted")

    def check():
        if compute_started is not None:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            if time.monotonic() - compute_started >= args.max_seconds - 2:
                raise DeadlineExceeded("Stopped with two seconds reserved for status serialization")

    def progress(row):
        history.append(dict(row, compute_seconds=time.monotonic() - compute_started))
        save("history.json", history)
        print(json.dumps(history[-1]), flush=True)

    settings = dict(vars(args), sizes=SIZES, lambda_grid=LAMBDAS, ridge=RIDGE,
                    network=CONFIG, pretrain_updates=1000, refit_updates=128,
                    batch_size=256, adam_lr=0.002, random_rotations=64,
                    finite_difference_step=0.01, scope=SCOPE,
                    gate={"clean_test_median_cosine_min": 0.9,
                          "clean_refit_test_median_cosine_min": 0.9,
                          "refit_center_and_neighbor_mse_ratio_max": 1.1,
                          "refit_directed_median_cosine_max": 0.0,
                          "refit_directed_negative_cosine_fraction_min": 0.25,
                          "random_matching_relative_tolerance": 0.1,
                          "test_rotation_center_and_neighbor_mse_relative_tolerance": 0.1,
                          "test_pointwise_drift_rtol": DRIFT_RTOL,
                          "test_pointwise_drift_atol": DRIFT_ATOL})
    save("args.json", settings)
    save("status.json", {"status": "running", "phase": phase, "scope": SCOPE})
    try:
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
        tracked = [SOURCE / "models/dyn_model.py", SOURCE / "models/model_utils.py",
                   SOURCE / "algorithms/dmo_shac.py"]
        save("source.json", {"commit": commit, "files": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "dm-control", "mujoco")},
            "native_loss": "mean((mean-real_delta)^2 * exp(-logvar)) + mean(logvar)",
            "input_normalization_adapter": "Train-fitted fixed standardization of observations AND actions is this DMControl adapter's choice; not a claim to reproduce the full native DMO preprocessing/training loop",
            "adaptation_scope": "All trainable StochSSM parameters during initial fit and 128-step refit",
            "finite_difference": "Full MuJoCo integration state plus environment step/reset state restored; real env.step labels"})
        phase = "real_data_collection"
        data, splits = {}, {}
        collection_start = time.monotonic()
        for name, offset in (("train", 0), ("dev", 1), ("test", 2), ("refit", 3)):
            data[name], splits[name] = collect(args.seed + offset, SIZES[name], name == "test")
            np.savez_compressed(out / f"data_{name}.npz", **data[name])
            print(json.dumps({"phase": phase, "split": name, "samples": SIZES[name]}), flush=True)
        result["collection_seconds"] = time.monotonic() - collection_start
        save("splits.json", splits)
        train_x = np.concatenate((data["train"]["obs"], data["train"]["action"]), axis=1)
        xmean, xstd = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
        delta_std = np.maximum((data["train"]["next_obs"] - data["train"]["obs"]).std(0), 1e-6)
        save("normalization.json", {"input_mean": xmean.tolist(), "input_std": xstd.tolist(),
                                    "delta_std": delta_std.tolist(), "fitted_on": "train only",
                                    "scope": "Fixed observation/action standardization is a DMControl adapter choice, not full native DMO preprocessing"})
        sys.path.insert(0, str(SOURCE))
        from models.dyn_model import StochSSM
        compute_started = time.monotonic()
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        device = torch.device(args.device)
        model = StochSSM(5, 1, 5, CONFIG, False, device=str(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        mean = torch.tensor(xmean, dtype=torch.float32, device=device)
        std = torch.tensor(xstd, dtype=torch.float32, device=device)
        tensors = {}
        for name, values in data.items():
            x = np.concatenate((values["obs"], values["action"]), 1)
            tensors[name] = (torch.tensor(x, dtype=torch.float32, device=device),
                             torch.tensor(values["next_obs"] - values["obs"], dtype=torch.float32, device=device))

        def fit(model, opt, xy, batches, label):
            model.train()
            for parameter in model.parameters():
                # Logvariance bounds in the original model are fixed constants.
                if parameter is not model.max_logvar and parameter is not model.min_logvar:
                    parameter.requires_grad_(True)
            x, y = xy
            for i, ids in enumerate(batches):
                check()
                z = (x[ids] - mean) / std
                pred, logvar, _ = model(z[:, :5], z[:, 5:])
                loss = ((pred - y[ids]).square() * (-logvar).exp()).mean() + logvar.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite NLL in {label}")
                if (i + 1) % (50 if label == "pretrain" else 32) == 0 or i + 1 == len(batches):
                    progress({"phase": label, "update": i + 1, "nll": float(loss.detach()),
                              "batch_mean_mse": float((pred.detach() - y[ids]).square().mean())})

        def features(model, x):
            hs, ds = [], []
            hidden = model.dyn_model[:-1]
            for start in range(0, len(x), 256):
                check()
                z = ((x[start:start + 256] - mean) / std).detach()
                tangent = torch.zeros_like(z)
                tangent[:, -1] = 1.0 / std[-1]
                with torch.no_grad():
                    h, d = torch.func.jvp(hidden, (z,), (tangent,))
                hs.append(torch.cat((h.detach(), torch.ones_like(h[:, :1])), 1))
                ds.append(torch.cat((d.detach(), torch.zeros_like(d[:, :1])), 1))
            return torch.cat(hs), torch.cat(ds)

        def head(model):
            layer = model.dyn_model[-1]
            return torch.cat((layer.weight[:5].detach().T, layer.bias[:5].detach()[None]), 0).clone()

        def install(model, weights):
            with torch.no_grad():
                model.dyn_model[-1].weight[:5].copy_(weights[:-1].T)
                model.dyn_model[-1].bias[:5].copy_(weights[-1])

        generator = torch.Generator().manual_seed(args.seed + 100)
        initial_ids = torch.randint(4096, (1000, 256), generator=generator).to(device)
        phase = "pretrain"
        fit(model, optimizer, tensors["train"], initial_ids, "pretrain")
        clean_state = copy.deepcopy(model.state_dict())
        clean_optimizer = copy.deepcopy(optimizer.state_dict())
        clean_hashes = tensor_hashes(model)
        result["initial_fit"] = {"batch_indices_sha256": sha_array(initial_ids.cpu().numpy()),
                                 "clean_optimizer_state_hashes": {
                                     f"{pid}.{key}": sha_array(value.detach().cpu().numpy())
                                     for pid, state in clean_optimizer["state"].items()
                                     for key, value in state.items() if torch.is_tensor(value)}}
        torch.save({k: v.detach().cpu() for k, v in clean_state.items()}, out / "clean_model.pt")
        phase = "train_only_linear_solve"
        htrain, dtrain = features(model, tensors["train"][0])
        hdev, ddev = features(model, tensors["dev"][0])
        w = head(model)
        clean_dev = hdev @ w
        dev_y = tensors["dev"][1]
        clean_dev_mse = float((clean_dev - dev_y).square().mean())
        # All terms use the same sample-mean convention; D is in raw action units.
        hh = htrain.double().T @ htrain.double() / len(htrain)
        dd = dtrain.double().T @ dtrain.double() / len(dtrain)
        rhs = -2 * dd @ w.double()
        candidates, deltas = [], []
        for lam in LAMBDAS:
            check()
            delta = torch.linalg.solve(dd + lam * hh + RIDGE * torch.eye(hh.shape[0], device=device, dtype=torch.float64), rhs).float()
            pred, jac = hdev @ (w + delta), ddev @ (w + delta)
            oldjac = ddev @ w
            cosine = torch.nn.functional.cosine_similarity(jac, oldjac, dim=1, eps=1e-12)
            mse = float((pred - dev_y).square().mean())
            row = {"lambda": lam, "dev_mse": mse, "dev_mse_ratio": mse / max(clean_dev_mse, 1e-30),
                   "dev_median_cosine_to_clean": float(cosine.median()),
                   "delta_norm": float(delta.norm()),
                   "train_forward_drift_mse": float((htrain @ delta).square().mean()),
                   "dev_forward_drift_mse": float((hdev @ delta).square().mean()),
                   "eligible": bool(mse <= 1.1 * clean_dev_mse)}
            candidates.append(row)
            deltas.append(delta)
        eligible = [i for i, row in enumerate(candidates) if row["eligible"]]
        chosen = min(eligible, key=lambda i: (candidates[i]["dev_median_cosine_to_clean"], i)) if eligible else None
        result["selection"] = {"split": "dev", "clean_dev_mse": clean_dev_mse,
                               "candidates": candidates, "chosen_index": chosen,
                               "test_used_for_selection": False,
                               "objective": "mean||D Delta+2DW||^2 + lambda mean||H Delta||^2 + ridge||Delta||^2"}
        save("result.json", result)

        def jac_metrics(jac, true):
            norm, true_norm = np.linalg.norm(jac, axis=1), np.linalg.norm(true, axis=1)
            valid = true_norm > 1e-12
            cos = np.sum(jac * true, axis=1) / np.maximum(norm * true_norm, 1e-30)
            return {"median_cosine": float(np.median(cos[valid])),
                    "mean_cosine": float(np.mean(cos[valid])),
                    "fraction_cosine_negative": float(np.mean(cos[valid] < 0)),
                    "valid_reference_count": int(valid.sum()),
                    "near_zero_model_count": int((norm < 1e-12).sum()),
                    "median_norm": float(np.median(norm)), "median_true_norm": float(np.median(true_norm)),
                    "median_norm_ratio": float(np.median(norm[valid] / true_norm[valid])),
                    "mse": float(np.mean((jac - true) ** 2)),
                    "relative_l2_error": float(np.linalg.norm(jac - true) / max(np.linalg.norm(true), 1e-30))}

        evaluation_traces = {}

        def evaluate(model, label):
            check()
            x, y = tensors["test"]
            ht, dt = features(model, x)
            wt = head(model)
            prediction = (ht @ wt).detach().cpu().numpy()
            jac = (dt @ wt).detach().cpu().numpy()
            target = y.cpu().numpy()
            truth = data["test"]["true_jacobian"]
            error = prediction - target
            neighbors, neighbor_predictions = [], {}
            for side in ("minus", "plus"):
                nx = x.clone()
                nx[:, -1:] = torch.tensor(data["test"][f"action_{side}"], dtype=x.dtype, device=device)
                predictions = []
                with torch.no_grad():
                    for start in range(0, len(nx), 256):
                        check()
                        z = (nx[start:start + 256] - mean) / std
                        predictions.append(model(z[:, :5], z[:, 5:])[0].cpu().numpy())
                target_n = data["test"][f"next_{side}"] - data["test"]["obs"]
                neighbor_predictions[f"predicted_delta_{side}"] = np.concatenate(predictions)
                neighbors.append(neighbor_predictions[f"predicted_delta_{side}"] - target_n)
            ne = np.concatenate(neighbors)
            metrics = {"center_mse": float(np.mean(error ** 2)),
                       "center_per_output_mse": np.mean(error ** 2, 0).tolist(),
                       "center_per_output_normalized_delta_mse": np.mean((error / delta_std) ** 2, 0).tolist(),
                       "neighbor_mse": float(np.mean(ne ** 2)),
                       "neighbor_mse_by_side": {side: float(np.mean(error_n ** 2))
                                                for side, error_n in zip(("minus", "plus"), neighbors)},
                       "neighbor_per_output_normalized_delta_mse": np.mean((ne / delta_std) ** 2, 0).tolist(),
                       "identity_delta_mse": float(np.mean(target ** 2)),
                       "jacobian_raw": jac_metrics(jac, truth),
                       "jacobian_standardized": jac_metrics(jac * xstd[-1] / delta_std, truth * xstd[-1] / delta_std),
                       "hashes": tensor_hashes(model)}
            trace = dict(predicted_delta=prediction, action_jacobian=jac, **neighbor_predictions)
            np.savez_compressed(out / f"trace_{label}.npz", **trace)
            if label.endswith("_before"):
                evaluation_traces[label] = trace
            return metrics

        phase = "test_clean_after_selection"
        result["branches"]["clean_before"] = evaluate(model, "clean_before")
        if chosen is None:
            reference_ok = result["branches"]["clean_before"]["jacobian_raw"]["median_cosine"] >= 0.9
            result["gate"] = {"passed": False, "classification": "negative" if reference_ok else "inconclusive",
                              "clean_jacobian_reference_valid": reference_ok,
                              "reason": "No pre-registered lambda passed dev forward-MSE gate"}
        else:
            directed = deltas[chosen]
            directed_dev_mse = candidates[chosen]["dev_mse"]
            response = hdev @ directed
            rotation_generator = torch.Generator().manual_seed(args.seed + 300)
            random_rows, rotations = [], []
            for i in range(64):
                q, r = torch.linalg.qr(torch.randn(5, 5, generator=rotation_generator))
                q = q * torch.where(torch.diag(r) < 0, -1.0, 1.0)
                q = q.to(device)
                random_pred = clean_dev + response @ q
                mse = float((random_pred - dev_y).square().mean())
                random_rows.append({"index": i, "dev_mse": mse,
                                    "relative_dev_truth_mse_difference": abs(mse - directed_dev_mse) / max(directed_dev_mse, 1e-30)})
                rotations.append(q)
            random_index = min(range(64), key=lambda i: (random_rows[i]["relative_dev_truth_mse_difference"], i))
            rotation = rotations[random_index]
            random_delta = directed @ rotation
            matching = dict(random_rows[random_index], selected_using="dev truth MSE only; no Jacobian or test selection",
                            mechanism="Seeded Haar output rotation of the entire mean-head delta, including bias",
                            candidate_rows=random_rows, directed_norm=float(directed.norm()),
                            random_norm=float(random_delta.norm()),
                            norm_relative_difference=float(abs(random_delta.norm() / directed.norm() - 1)),
                            directed_dev_drift_mse=float(response.square().mean()),
                            random_dev_drift_mse=float((response @ rotation).square().mean()))
            matching["matched"] = bool(matching["relative_dev_truth_mse_difference"] <= 0.1 and matching["norm_relative_difference"] <= 0.1)
            result["random_matching"] = matching
            np.savez_compressed(out / "perturbations.npz", directed=directed.cpu().numpy(),
                                random=random_delta.cpu().numpy(), rotation=rotation.cpu().numpy())
            for name, change in (("directed", directed), ("random", random_delta)):
                model.load_state_dict(clean_state)
                install(model, w + change)
                # All hidden and variance parameters must be bitwise unchanged.
                now = tensor_hashes(model)
                mutable = {f"dyn_model.{len(model.dyn_model)-1}.weight", f"dyn_model.{len(model.dyn_model)-1}.bias"}
                assert all(now[k] == clean_hashes[k] for k in now if k not in mutable)
                assert torch.equal(model.dyn_model[-1].weight[5:], clean_state[f"dyn_model.{len(model.dyn_model)-1}.weight"][5:])
                assert torch.equal(model.dyn_model[-1].bias[5:], clean_state[f"dyn_model.{len(model.dyn_model)-1}.bias"][5:])
                phase = f"test_{name}_before"
                result["branches"][f"{name}_before"] = evaluate(model, f"{name}_before")
                save("result.json", result)
            phase = "heldout_rotation_validation"
            heldout_matching = heldout_rotation_matching(result["branches"], evaluation_traces)
            result["random_matching"]["heldout_validation"] = heldout_matching
            save("result.json", result)
            xy = tuple(torch.cat((tensors["train"][i], tensors["refit"][i])) for i in (0, 1))
            refit_generator = torch.Generator().manual_seed(args.seed + 200)
            ids = torch.randint(len(xy[0]), (128 * 256,), generator=refit_generator)
            ids = ids[torch.randperm(len(ids), generator=refit_generator)].reshape(128, 256)
            result["refit"] = {"buffer": "train4096 plus independent random-policy refit2048",
                               "indices_sha256": sha_array(ids.numpy()), "same_batches_and_clean_Adam_state": True,
                               "all_model_parameters_trainable_except_native_fixed_logvar_bounds": True}
            ids = ids.to(device)
            for name, change in (("clean", torch.zeros_like(w)), ("directed", directed), ("random", random_delta)):
                check()
                model.load_state_dict(clean_state)
                install(model, w + change)
                optimizer.load_state_dict(copy.deepcopy(clean_optimizer))
                phase = f"refit_{name}"
                fit(model, optimizer, xy, ids, phase)
                phase = f"test_{name}_refit"
                result["branches"][f"{name}_refit"] = evaluate(model, f"{name}_refit")
                save("result.json", result)
            b = result["branches"]
            c, d, r = b["clean_refit"], b["directed_refit"], b["random_refit"]
            checks = {"clean_jacobian_reference_valid": b["clean_before"]["jacobian_raw"]["median_cosine"] >= 0.9,
                      "clean_refit_jacobian_reference_valid": c["jacobian_raw"]["median_cosine"] >= 0.9,
                      "random_matched_on_dev": matching["matched"],
                      "random_matched_on_test_truth_mse": heldout_matching["truth_mse_matched"],
                      "random_pointwise_drift_matched_on_test": heldout_matching["pointwise_drift_matched"],
                      "refit_center_forward_preserved": d["center_mse"] <= 1.1 * c["center_mse"],
                      "refit_neighbor_forward_preserved": d["neighbor_mse"] <= 1.1 * c["neighbor_mse"],
                      "refit_median_cosine_nonpositive": d["jacobian_raw"]["median_cosine"] <= 0.0,
                      "refit_negative_cosine_fraction": d["jacobian_raw"]["fraction_cosine_negative"] >= 0.25,
                      "refit_directed_error_exceeds_random": d["jacobian_raw"]["mse"] > r["jacobian_raw"]["mse"]}
            classification = "necessary_condition_passed" if all(checks.values()) else "negative"
            validity_checks = ("clean_jacobian_reference_valid", "clean_refit_jacobian_reference_valid",
                               "random_matched_on_dev", "random_matched_on_test_truth_mse",
                               "random_pointwise_drift_matched_on_test")
            if not all(checks[key] for key in validity_checks):
                classification = "inconclusive"
            result["gate"] = {"passed": all(checks.values()), "classification": classification, "checks": checks,
                              "interpretation": SCOPE}
        check()
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["compute_wall_seconds"] = time.monotonic() - compute_started
        result["total_wall_seconds"] = time.monotonic() - started
        result["artifact_bytes_before_final_json"] = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        save("result.json", result)
        save("history.json", history)
        save("status.json", {"status": "completed", "phase": "done", "gate": result["gate"],
                             "compute_wall_seconds": result["compute_wall_seconds"], "scope": SCOPE})
        print(json.dumps(result["gate"]), flush=True)
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["gate"] = {"passed": False, "classification": "inconclusive",
                          "reason": "deadline" if isinstance(exc, DeadlineExceeded) else "execution_error"}
        save("result.json", result)
        save("history.json", history)
        save("status.json", {"status": "deadline" if isinstance(exc, DeadlineExceeded) else "error",
                             "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
                             "traceback": traceback.format_exc(), "scope": SCOPE,
                             "compute_wall_seconds": None if compute_started is None else time.monotonic() - compute_started,
                             "total_wall_seconds": time.monotonic() - started})
        if not isinstance(exc, DeadlineExceeded):
            raise


if __name__ == "__main__":
    main()
