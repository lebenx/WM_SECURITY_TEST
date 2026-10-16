"""Bounded do(scale) sensitivity screen with a fixed world model and native actor updates.

This is a researcher intervention on return-normalization state, not an attack
or a reproduction of a native online/shared deployment. Both adaptation arms
use the same real target-task trajectory, batches, RNG, and initial actor.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch

from wm_security.tdmpc_native import load_agent, episode, set_seed
from pilot_tdmpc_action_search import reset_agent, snapshot


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'models/tdmpc2/mt30-48M.pt'
REFERENCE = ROOT / 'runs/tdmpc_clean_gate_v0/walker-walk_11_mpc.pt'
SCALE_LOGS = {
    'benign_scale': ROOT / 'runs/tdmpc_update_dose_v0/benign_300_learning.json',
    'max_error_scale': ROOT / 'runs/tdmpc_update_dose_v0/max_error_300_learning.json',
}
TASK, TRAIN_SEED, EVAL_SEED = 'walker-walk', 31, 12
UPDATES, BATCH_SIZE, CLEAN_GATE, DROP_GATE = 40, 32, 900.0, 100.0


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_digest(model, policy=False):
    """Hash each selected state tensor, including buffers and target Q tensors."""
    tensors, other = {}, {}
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('_pi.') != policy:
            continue
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            header = f'{cpu.dtype}:{tuple(cpu.shape)}:'.encode()
            raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
            tensors[name] = hashlib.sha256(header + raw).hexdigest()
            item = tensors[name]
        else:
            other[name] = repr(value)
            item = other[name]
        digest.update(name.encode() + b'\0' + item.encode() + b'\0')
    return {'sha256': digest.hexdigest(), 'tensor_count': len(tensors),
            'tensors': tensors, 'non_tensor_state': other}


def read_scale(path):
    history = json.loads(path.read_text())['history']
    if not history or history[-1]['update'] != 300:
        raise ValueError(f'Expected a completed 300-update source log: {path}')
    value = float(history[-1]['pi_scale'])
    if not math.isfinite(value) or value < 1:
        raise ValueError(f'Invalid empirical scale in {path}')
    return {'value': value, 'path': str(path), 'sha256': file_digest(path),
            'source_update': history[-1]['update']}


def update_actor(agent, data, initial_scale, progress):
    """Use the native latent construction and update_pi; leave all other weights fixed."""
    set_seed(TRAIN_SEED)
    cfg, model = agent.cfg, agent.model
    cfg.batch_size = BATCH_SIZE
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model._pi.parameters():
        parameter.requires_grad_(True)
    model.train()  # Match native _update mode, including Q-network dropout.
    agent.scale.value.fill_(initial_scale)
    observations = data['obs'].to(agent.device)
    actions = data['action'].to(agent.device)
    tasks = torch.full((BATCH_SIZE,), int(data['task']),
                       device=agent.device, dtype=torch.long)
    history, batches = [], []
    for update in range(UPDATES):
        starts = torch.randint(len(actions) - cfg.horizon + 1,
                               (BATCH_SIZE,), device=agent.device)
        with torch.no_grad():
            z = model.encode(observations[starts], tasks)
            zs = [z]
            for step in range(cfg.horizon):
                z = model.next(z, actions[starts + step], tasks)
                zs.append(z)
            zs = torch.stack(zs).detach()
        before_scale = float(agent.scale.value.item())
        info = agent.update_pi(zs, tasks)
        row = {'update': update + 1, 'scale_before_update': before_scale}
        row.update({key: float(value.detach().mean()) if isinstance(value, torch.Tensor)
                    else float(value) for key, value in info.items()})
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError(f'Nonfinite native actor update: {row}')
        history.append(row)
        batches.append(starts.cpu().tolist())
        progress(history, batches)
    model.eval()
    return history, batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'runs/tdmpc_scale_carryover_v0'))
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Preserve existing run; choose an empty directory: {out}')
    out.mkdir(parents=True, exist_ok=True)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + '\n')

    results, histories, batch_traces, checks = [], {}, {}, {}
    started = time.monotonic()
    try:
        scales = {name: read_scale(path) for name, path in SCALE_LOGS.items()}
        plan = {'checkpoint': str(CHECKPOINT), 'reference': str(REFERENCE),
                'reference_sha256': file_digest(REFERENCE), 'target_task': TASK,
                'reference_seed': 11, 'eval_seed': EVAL_SEED,
                'update_rng_seed': TRAIN_SEED, 'updates': UPDATES,
                'batch_size': BATCH_SIZE, 'evaluation_steps': 500,
                'conditions': ['clean', *scales], 'empirical_initial_scales': scales,
                'clean_return_gate': CLEAN_GATE, 'absolute_return_drop_gate': DROP_GATE,
                'continuation_gate': 'max_error_scale loses at least100 return versus both clean and benign_scale',
                'scope': 'Researcher do(scale) carryover sensitivity intervention; not a complete attack or native online/shared protocol.',
                'learning': 'Fixed encoder/dynamics/reward/Q/task tensors;40 original update_pi calls on the same real walker-walk sequence, identical sampled batches and RNG, fresh Adam per arm. Native scale update remains active.',
                'evaluation': 'Full original MPC, original horizon/samples/iterations/policy proposals/terminal bootstrap; no planner weakening.',
                'limitations': 'One held-out reset seed. Source scales came from prior full-update adaptation with fresh scale=1, not a persisted online checkpoint. This screen tests causal sensitivity only.'}
        save('plan.json', plan)
        save('status.json', {'status': 'running', 'phase': 'load_agent'})
        agent, provenance = load_agent(CHECKPOINT)
        provenance['script_sha256'] = file_digest(Path(__file__))
        provenance['actor_optimizer'] = {'class': type(agent.pi_optim).__name__,
                                         'lr': agent.cfg.lr, 'eps': agent.pi_optim.defaults['eps']}
        provenance['normalizer'] = {'tau': agent.cfg.tau,
                                    'entropy_coef': agent.cfg.entropy_coef,
                                    'percentiles': agent.scale._percentiles.cpu().tolist()}
        save('provenance.json', provenance)
        data = torch.load(REFERENCE, map_location='cpu', weights_only=False)
        if int(data['task']) != agent.cfg.tasks.index(TASK) or int(data['seed']) != 11:
            raise ValueError('The reference must be the original walker-walk seed11 episode.')
        if len(data['obs']) != len(data['action']) + 1 or len(data['action']) < agent.cfg.horizon:
            raise ValueError('Reference trajectory has invalid transition alignment.')
        original = snapshot(agent)
        fixed = model_digest(agent.model)
        initial_actor = model_digest(agent.model, policy=True)

        def evaluate(condition):
            save('status.json', {'status': 'running', 'phase': 'evaluate',
                                 'condition': condition, 'results': results})
            result, trace = episode(agent, TASK, EVAL_SEED, policy='mpc', steps=500,
                                    progress=lambda x: print(json.dumps({'condition': condition, **x}), flush=True))
            if not result['full_episode']:
                raise AssertionError('Evaluation did not execute a full native500-step episode.')
            result['condition'] = condition
            result['final_scale'] = float(agent.scale.value.item())
            after = model_digest(agent.model)
            checks[condition]['after_evaluation'] = after
            checks[condition]['non_pi_unchanged'] = after == fixed
            save('tensor_digests.json', {'reference_non_pi': fixed,
                                        'initial_actor': initial_actor, 'conditions': checks})
            if after != fixed:
                raise AssertionError(f'Non-policy state changed in {condition}.')
            torch.save(trace, out / f'{condition}_trace.pt')
            results.append(result)
            save('results.json', results)
            print(json.dumps(result), flush=True)
            return result

        reset_agent(agent, original)
        checks['clean'] = {'before': model_digest(agent.model)}
        clean = evaluate('clean')
        if clean['return'] < CLEAN_GATE:
            save('status.json', {'status': 'clean_gate_failed', 'results': results,
                                 'wall_seconds': time.monotonic() - started})
            return
        for condition, source in scales.items():
            reset_agent(agent, original)
            if agent.pi_optim.state:
                raise AssertionError('Actor optimizer state must start empty in each arm.')
            before, actor_before = model_digest(agent.model), model_digest(agent.model, policy=True)
            if before != fixed or actor_before != initial_actor:
                raise AssertionError('Both arms must start from identical model/actor tensors.')
            checks[condition] = {'before': before, 'actor_before': actor_before}
            save('status.json', {'status': 'running', 'phase': 'update_pi',
                                 'condition': condition, 'results': results})

            def progress(history, batches):
                histories[condition] = history
                batch_traces[condition] = batches
                save('history.json', histories)
                save('training_batches.json', batch_traces)

            history, batches = update_actor(agent, data, source['value'], progress)
            checks[condition]['after_updates'] = model_digest(agent.model)
            checks[condition]['actor_after'] = model_digest(agent.model, policy=True)
            if checks[condition]['after_updates'] != fixed:
                save('tensor_digests.json', {'reference_non_pi': fixed,
                                            'initial_actor': initial_actor, 'conditions': checks})
                raise AssertionError(f'Non-policy tensors changed during {condition} updates.')
            if condition == 'max_error_scale' and batches != batch_traces['benign_scale']:
                raise AssertionError('The two actor-update arms did not use identical batches.')
            evaluate(condition)
        returns = {row['condition']: row['return'] for row in results}
        vs_clean = returns['clean'] - returns['max_error_scale']
        vs_benign = returns['benign_scale'] - returns['max_error_scale']
        gate = vs_clean >= DROP_GATE and vs_benign >= DROP_GATE
        save('status.json', {'status': 'completed', 'results': results,
                             'max_error_drop_vs_clean': vs_clean,
                             'max_error_drop_vs_benign': vs_benign,
                             'continuation_gate_passed': gate,
                             'identical_training_batches': True,
                             'all_non_pi_tensors_unchanged': True,
                             'wall_seconds': time.monotonic() - started,
                             'interpretation': 'A positive result would justify investigating a full genuine-experience attack; a negative result retires this observed scale range at this task/reset/update budget. Neither outcome establishes a population-level attack.'})
    except Exception as exc:
        save('status.json', {'status': 'error', 'error': str(exc),
                             'traceback': traceback.format_exc(), 'results': results,
                             'wall_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    main()
