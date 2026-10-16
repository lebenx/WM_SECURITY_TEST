"""Short causal screen for shared optimizer-state laundering.

An online multitask learner can retain Adam moments and policy-optimizer state
across task episodes.  This pilot gives one branch legal source-task data,
then compares victim updates with the contaminated state against the *same
contaminated model weights* with fresh optimizer state.  A clean victim-only
branch is a second control.  The source data are unedited native reacher
trajectories; this is a mechanism screen, not a completed attack claim.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import torch

from wm_security.tdmpc_native import load_agent, episode, set_seed
from pilot_tdmpc_adaptation import adapt


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/tdmpc2/mt30-48M.pt"
SOURCE_DIR = ROOT / "runs/tdmpc_adaptation_screen_v0"
SOURCE_PATHS = {"mpc": SOURCE_DIR / "mpc_data.pt", "random": SOURCE_DIR / "random_data.pt"}
VICTIM_PATH = SOURCE_DIR / "clean_walker-walk.pt"


def digest_optimizer(optim):
    h = hashlib.sha256()
    entries = 0
    for param, state in sorted(optim.state.items(), key=lambda x: id(x[0])):
        h.update(str(param.numel()).encode())
        for name, value in sorted(state.items()):
            h.update(name.encode())
            if isinstance(value, torch.Tensor):
                h.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                h.update(repr(value).encode())
            entries += 1
    return h.hexdigest(), entries


def reset_optimizers(agent):
    """Reset all train-time state while retaining the current model weights."""
    cfg, model = agent.cfg, agent.model
    agent.optim = torch.optim.Adam([
        {'params': model._encoder.parameters(), 'lr': cfg.lr * cfg.enc_lr_scale},
        {'params': model._dynamics.parameters()},
        {'params': model._reward.parameters()},
        {'params': model._termination.parameters() if cfg.episodic else []},
        {'params': model._Qs.parameters()},
        {'params': model._task_emb.parameters() if cfg.multitask else []},
    ], lr=cfg.lr, capturable=True)
    agent.pi_optim = torch.optim.Adam(model._pi.parameters(), lr=cfg.lr, eps=1e-5, capturable=True)
    from common.scale import RunningScale
    agent.scale = RunningScale(cfg)


def update_on_data(agent, data, updates):
    history = adapt(agent, data, updates, 31, 'native_full')
    opt_digest, opt_entries = digest_optimizer(agent.optim)
    pi_digest, pi_entries = digest_optimizer(agent.pi_optim)
    return history, {
        'optim_sha256': opt_digest, 'optim_state_entries': opt_entries,
        'pi_optim_sha256': pi_digest, 'pi_optim_state_entries': pi_entries,
        'scale': float(agent.scale.value),
    }


def evaluate(agent, condition, seed=11, steps=500):
    agent._prev_mean.zero_()
    result, _ = episode(agent, 'walker-walk', seed, 'mpc', steps)
    result['condition'] = condition
    return result


def main():
    out = ROOT / 'runs/tdmpc_optimizer_state_laundering_v0'
    out.mkdir(parents=True, exist_ok=False)
    source_updates, victim_updates = 20, 20
    plan = {
        'checkpoint': str(CHECKPOINT.relative_to(ROOT)),
        'source_task': 'reacher-easy', 'victim_task': 'walker-walk',
        'source_arms': list(SOURCE_PATHS),
        'source_updates': source_updates, 'victim_updates': victim_updates,
        'attacker_operation': 'Native task-labeled source data updates shared model and Adam/pi optimizer state, then exits; no labels, observations, rewards, or weights are edited.',
        'causal_control': 'For each source arm, victim updates start from the exact same source-contaminated model weights. One branch retains optimizer state; another resets all optimizer and running-scale state. A fresh victim-only branch is also evaluated.',
        'scope': 'Short CUDA screen of optimizer-state persistence; no population claim, no long training, no API requests.',
        'clean_gate': 'Walker native MPC return >= 900 for every completed branch.',
        'effect_gate': 'Retain only if state-retaining victim return is at least 100 below same-model reset and clean controls.',
        'api_requests': 0,
        'gpu_training': True,
    }
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    victim = torch.load(VICTIM_PATH, map_location='cpu', weights_only=False)
    results, details = [], {}
    started = time.monotonic()
    for source_name, source_path in SOURCE_PATHS.items():
        source = torch.load(source_path, map_location='cpu', weights_only=False)
        # Contaminated branch: source updates and victim updates share state.
        contaminated, prov = load_agent(CHECKPOINT)
        (out / f'{source_name}_provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
        source_history, source_state = update_on_data(contaminated, source, source_updates)
        source_model = {k: v.detach().cpu().clone() for k, v in contaminated.model.state_dict().items()}
        source_optim_digest = source_state
        victim_history, victim_state = update_on_data(contaminated, victim, victim_updates)
        r = evaluate(contaminated, f'{source_name}_retain_state')
        results.append(r)
        details[f'{source_name}_retain_state'] = {
            'source_state': source_optim_digest, 'victim_state': victim_state,
            'source_final_loss': source_history[-1]['total_loss'],
            'victim_final_loss': victim_history[-1]['total_loss'],
        }

        # Same contaminated model weights, but no source optimizer state.
        reset_branch, _ = load_agent(CHECKPOINT)
        reset_branch.model.load_state_dict(source_model)
        reset_optimizers(reset_branch)
        reset_history, reset_state = update_on_data(reset_branch, victim, victim_updates)
        r = evaluate(reset_branch, f'{source_name}_reset_state')
        results.append(r)
        details[f'{source_name}_reset_state'] = {
            'victim_state': reset_state,
            'victim_final_loss': reset_history[-1]['total_loss'],
            'same_source_contaminated_model_weights': True,
        }

        # Fresh victim-only branch.
        clean_branch, _ = load_agent(CHECKPOINT)
        clean_history, clean_state = update_on_data(clean_branch, victim, victim_updates)
        r = evaluate(clean_branch, f'{source_name}_clean_victim')
        results.append(r)
        details[f'{source_name}_clean_victim'] = {
            'victim_state': clean_state,
            'victim_final_loss': clean_history[-1]['total_loss'],
            'source_updates': 0,
        }
        (out / f'{source_name}_details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2, default=float) + '\n')
        print(json.dumps({'source': source_name, 'latest': results[-3:]}, ensure_ascii=False), flush=True)

    by_source = {}
    for source_name in SOURCE_PATHS:
        by = {r['condition']: r for r in results if r['condition'].startswith(source_name)}
        retain = by[f'{source_name}_retain_state']['return']
        reset = by[f'{source_name}_reset_state']['return']
        clean = by[f'{source_name}_clean_victim']['return']
        by_source[source_name] = {
            'retain_state_return': retain, 'reset_state_return': reset,
            'clean_victim_return': clean,
            'retain_minus_reset': retain - reset,
            'retain_minus_clean': retain - clean,
            'state_effect_gate_passed': retain <= reset - 100 and retain <= clean - 100,
        }
    summary = {
        'status': 'completed', 'results': results, 'by_source': by_source,
        'details': details,
        'decision': 'No candidate promotion unless the state-retaining branch loses at least 100 return against both same-model reset and clean victim controls. This short screen is diagnostic and does not establish a broad claim.',
        'elapsed_seconds': time.monotonic() - started,
        'native_environment_steps': sum(r['steps'] for r in results),
        'api_requests': 0, 'gpu_training': True,
    }
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + '\n')
    print(json.dumps({'status': summary['status'], 'by_source': by_source}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
