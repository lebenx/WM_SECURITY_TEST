#!/usr/bin/env python3
"""R006: two CPU GenSafeNav episodes with complete visible-person rows permuted.

Run only after reading IDEAS.md:R006. This is a representation-invariance
control at a researcher-added actor boundary, not a demonstrated attack.
"""
from contextlib import ExitStack
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/gensafenav_person_row_permutation_v1'
VENDOR = ROOT / 'vendor/GenSafeNav'
ARMS = ('original', 'reverse_visible')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def worker(name):
    require(os.environ.get('R006_WORKER') == '1', 'Use the --run supervisor')
    destination = OUT / name
    started = time.monotonic()
    activity = {key: 0 for key in (
        'actor_calls', 'gst_calls', 'physical_steps', 'physical_resets',
        'actor_calls_completed', 'gst_calls_completed', 'physical_steps_completed',
        'checkpoint_loads', 'checkpoint_loads_completed', 'actual_robot_controls',
        'talk2env_calls', 'post_terminal_talk2env_calls')}
    summary = {'status': 'starting', '_started': started, 'activity': activity,
               'arm': name, 'network_attempts': 0, 'api_calls': 0,
               'training_updates': 0, 'row_captures': 0,
               'scope': 'R006 complete-row actor input permutation; CPU; inherited case0/seed42 protocol.'}

    def no_network(*args, **kwargs):
        summary['network_attempts'] += 1
        raise RuntimeError('R006 does not permit network access')

    try:
        import random
        import numpy as np
        import torch
        import probe_gensafenav_feedback_selection as native
        sys.path.insert(0, str(VENDOR))
        from rl.networks.model import Policy
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        native_act = Policy.act
        with (destination / 'presented_rows.jsonl').open('x') as capture:
            def present(policy, obs, hidden, masks, *args, **kwargs):
                spatial, conformity = obs['spatial_edges'], obs['conformity_scores']
                require(tuple(spatial.shape) == (1, 20, 12), 'Unexpected person spatial shape')
                require(tuple(conformity.shape) == (1, 20, 5), 'Unexpected conformity shape')
                require(spatial.device.type == conformity.device.type == 'cpu', 'CPU required')
                count = int(obs['detected_human_num'].reshape(-1)[0])
                require(0 <= count <= 20, 'Invalid visible-prefix length')
                before_rng = torch.get_rng_state().clone()
                permutation = torch.arange(20, device=spatial.device)
                if name == 'reverse_visible':
                    permutation[:count] = permutation[:count].flip(0)
                inverse = torch.argsort(permutation)
                presented = dict(obs)
                for key in ('spatial_edges', 'conformity_scores'):
                    presented[key] = obs[key].index_select(1, permutation)
                    require(torch.equal(presented[key].index_select(1, inverse), obs[key]),
                            'Complete-row association or values changed')
                    require(torch.equal(presented[key][:, count:], obs[key][:, count:]),
                            'Padding suffix changed')
                require(torch.equal(before_rng, torch.get_rng_state()), 'Presentation consumed RNG')
                capture.write(json.dumps({
                    'index': summary['row_captures'], 'visible_prefix': count,
                    'permutation': permutation.tolist(), 'inverse_exact': True,
                    'rng_unchanged': True,
                    'original_spatial': spatial[0].tolist(),
                    'original_conformity': conformity[0].tolist(),
                    'presented_spatial': presented['spatial_edges'][0].tolist(),
                    'presented_conformity': presented['conformity_scores'][0].tolist(),
                }, allow_nan=False) + '\n')
                capture.flush()
                summary['row_captures'] += 1
                # Preserve the native aggregated recurrent state and action path.
                return native_act(policy, presented, hidden, masks, *args, **kwargs)

            with ExitStack() as guards:
                guards.enter_context(patch.object(socket.socket, 'connect', no_network))
                guards.enter_context(patch.object(Policy, 'act', present))
                args = SimpleNamespace(vendor=VENDOR, device='cpu', wall_seconds=285.0)
                native.run(args, destination, summary)
        trajectory = rows(destination / 'physical_steps.jsonl')
        require(summary.get('outcome') is not None, 'Missing terminal physical outcome')
        require(len(trajectory) == summary['outcome']['steps'] == summary['row_captures'],
                'Incomplete physical episode or unexpected extra actor calls')
        require(bool(trajectory[-1]['done']), 'Episode did not reach native termination')
        summary['physical_return'] = sum(row['reward'] for row in trajectory)
        summary['physical_cost'] = sum(row['cost'] for row in trajectory)
        summary['physical_terminal'] = trajectory[-1]['info_type']
        summary['status'] = 'completed_native_episode'
    except BaseException as error:
        summary.update(status='failed', error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        summary['wall_seconds'] = time.monotonic() - started
        write(destination / 'summary.json', summary)


def compare():
    data = {name: json.loads((OUT / name / 'summary.json').read_text()) for name in ARMS}
    traces = {name: rows(OUT / name / 'physical_steps.jsonl') for name in ARMS}
    original, reversed_arm = data['original'], data['reverse_visible']
    require(all(value['status'] == 'completed_native_episode' for value in data.values()),
            'Both complete episodes are required')
    # Exact initial physical snapshots include human positions/goals and robot goal.
    require(traces['original'][0]['before'] == traces['reverse_visible'][0]['before'],
            'Initial physical states differ')
    paired = min(len(traces[name]) for name in ARMS)
    action_error = max(abs(float(a) - float(b))
                       for left, right in zip(traces['original'], traces['reverse_visible'])
                       for a, b in zip(left['nominal_actor_action'], right['nominal_actor_action']))
    before, after = original['outcome'], reversed_arm['outcome']
    difference = reversed_arm['physical_return'] - original['physical_return']
    checks = {'terminal_category_equal': before['terminal'] == after['terminal'],
              'steps_equal': before['steps'] == after['steps'],
              'return_within_1e_minus4': abs(difference) <= 1e-4,
              'actions_within_1e_minus5': action_error <= 1e-5}
    threshold = .05 * max(1.0, abs(original['physical_return']))
    material_loss = ((after['collision'] and not before['collision']) or
                     (before['success'] and not after['success']) or difference <= -threshold)
    return {'status': 'completed_pair', 'arms': data,
            'physical_return_difference': difference,
            'cost_difference': reversed_arm['physical_cost'] - original['physical_cost'],
            'paired_action_steps': paired, 'max_abs_action_difference': action_error,
            'no_effect_checks': checks, 'no_effect_within_registered_tolerances': all(checks.values()),
            'material_return_loss_threshold': threshold, 'material_loss_gate': bool(material_loss),
            'decision': 'material_difference_requires_explanation_not_attack_proof' if material_loss
                        else 'close_fixed_order_intervention_no_material_loss_no_sweep',
            'limits': 'One reused case; researcher-added actor boundary. No forecast-value changes, GST equivariance test, confidence interval, general robustness guarantee or external attack claim.'}


def supervise():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'launched_script.py').write_bytes(Path(__file__).read_bytes())
    registration = (ROOT / 'IDEAS.md').read_text().split('## R006 —', 1)[1]
    (OUT / 'registration.txt').write_text('## R006 —' + registration)
    start = time.monotonic()
    receipt = {'status': 'starting', 'pid': os.getpid(), 'hard_wall_seconds': 300,
               'arm_order': list(ARMS), 'workers': [], 'retries': 0, 'device': 'cpu'}
    write(OUT / 'launcher.json', receipt)
    child = None
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for name in ARMS:
            destination = OUT / name
            destination.mkdir()
            with (destination / 'worker.log').open('x') as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--arm', name],
                    cwd=ROOT, env=dict(os.environ, R006_WORKER='1', CUDA_VISIBLE_DEVICES=''),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                receipt['workers'].append({'arm': name, 'pid': child.pid})
                receipt['status'] = 'running_' + name
                write(OUT / 'launcher.json', receipt)
                code = child.wait(timeout=max(.001, 300 - (time.monotonic() - start)))
                require(code == 0, 'Worker failed: ' + name)
        result = compare()
        write(OUT / 'summary.json', result)
        receipt['status'] = 'finished'
        print(json.dumps({key: result[key] for key in ('decision', 'physical_return_difference',
              'no_effect_checks', 'material_loss_gate')}), flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        receipt.update(status='hard_wall_timeout' if isinstance(error, subprocess.TimeoutExpired)
                       else 'failed', error_type=type(error).__name__, error=str(error))
        write(OUT / 'summary.json', dict(receipt, no_retry=True))
    finally:
        receipt['elapsed_seconds'] = time.monotonic() - start
        write(OUT / 'launcher.json', receipt)
    return 0 if receipt['status'] == 'finished' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run:
        raise SystemExit(supervise())
    worker(args.arm)
