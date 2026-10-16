#!/usr/bin/env python3
"""R007: fixed single-track duplication at a researcher-added tracker/GST boundary.

Three CPU episodes, one per subprocess, with a shared 300-second hard budget.
Read IDEAS.md:R007 before --run. This is not a native external attack claim.
No training, API calls, retries, seed search, or duplication-count search.
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

from run_gensafenav_person_row_permutation import require, rows, write

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/GenSafeNav'
OUT = ROOT / 'runs/gensafenav_forecast_track_duplicate_v1'
ARMS = ('original', 'duplicate_gst_only', 'pre_gst_exact_dedup')


def snapshot(value):
    """Record directly inspectable actor inputs without hashes or extra inference."""
    if hasattr(value, 'detach'):
        array = value.detach().cpu().contiguous().numpy()
        return {'shape': list(array.shape), 'dtype': str(array.dtype),
                'values': array.tolist()}
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unexpected actor data type: ' + type(value).__name__)


def worker(name):
    require(os.environ.get('R007_WORKER') == '1', 'Use the --run supervisor')
    destination = OUT / name
    started = time.monotonic()
    activity = {key: 0 for key in (
        'actor_calls', 'gst_calls', 'physical_steps', 'physical_resets',
        'actor_calls_completed', 'gst_calls_completed', 'physical_steps_completed',
        'checkpoint_loads', 'checkpoint_loads_completed', 'actual_robot_controls',
        'talk2env_calls', 'post_terminal_talk2env_calls')}
    summary = {'status': 'starting', '_started': started, 'activity': activity,
               'arm': name, 'network_attempts': 0, 'api_calls': 0,
               'training_updates': 0, 'hook_calls': 0, 'actor_captures': 0,
               'scope': 'R007 tracker-to-forecast boundary; fixed single duplicate; '
                        'original entity set consumed by actor; CPU case0/seed42.',
               'invalid_output_restoration': 'Restore only consumer-used mu_xy=-999 '
                        'and mask=0. Sigma/correlation are not consumed by the '
                        'native wrapper and are not claimed restored.',
               'dedup_scope': 'Controlled exact duplicate removal restores the saved '
                        'unused slot before GST; not a generic deployed defense.'}
    records = []

    def no_network(*args, **kwargs):
        summary['network_attempts'] += 1
        raise RuntimeError('R007 does not permit network access')

    try:
        import random
        import numpy as np
        import torch
        import probe_gensafenav_feedback_selection as native
        sys.path.insert(0, str(VENDOR))
        from rl.networks.model import Policy
        from gst_updated.scripts.wrapper.crowd_nav_interface_parallel import (
            CrowdNavPredInterfaceMultiEnv)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        native_predict = CrowdNavPredInterfaceMultiEnv.forward
        native_act = Policy.act
        (destination / 'hook_arrays').mkdir()
        with ExitStack() as guards:
            hook_file = guards.enter_context((destination / 'interventions.jsonl').open('x'))
            actor_file = guards.enter_context((destination / 'actor_io.jsonl').open('x'))

            def predict(predictor, input_traj, input_binary_mask, *args, **kwargs):
                summary['hook_calls'] += 1
                index = summary['hook_calls']
                require(tuple(input_traj.shape) == (1, 20, 5, 2), 'Unexpected GST trajectory shape')
                require(tuple(input_binary_mask.shape) == (1, 20, 5, 1), 'Unexpected GST mask shape')
                require(input_traj.device.type == input_binary_mask.device.type == 'cpu',
                        'CPU required')
                require(bool(torch.isfinite(input_traj).all()), 'Nonfinite GST input')
                require(bool(((input_binary_mask == 0) | (input_binary_mask == 1)).all()),
                        'GST mask must be binary')
                before_rng = torch.get_rng_state().clone()
                original_traj, original_mask = input_traj.clone(), input_binary_mask.clone()
                valid = (original_mask[0, :, :, 0] == 1).all(dim=1)
                empty = (original_mask[0, :, :, 0] == 0).all(dim=1)
                sources = torch.nonzero(valid, as_tuple=False).flatten().tolist()
                destinations = torch.nonzero(empty, as_tuple=False).flatten().tolist()
                eligible = bool(sources and destinations)
                source = sources[0] if sources else None
                dest = destinations[0] if destinations else None
                gst_traj, gst_mask = input_traj, input_binary_mask
                injected = name != 'original' and eligible
                if injected:
                    gst_traj, gst_mask = original_traj.clone(), original_mask.clone()
                    gst_traj[:, dest] = original_traj[:, source]
                    gst_mask[:, dest] = original_mask[:, source]
                    other = torch.arange(20) != dest
                    require(torch.equal(gst_traj[:, other], original_traj[:, other]) and
                            torch.equal(gst_mask[:, other], original_mask[:, other]),
                            'An existing track was changed')
                    require(torch.equal(gst_traj[:, dest], original_traj[:, source]) and
                            torch.equal(gst_mask[:, dest], original_mask[:, source]),
                            'Duplicate is not an exact complete history')
                    if name == 'pre_gst_exact_dedup':
                        # Keep an exact copy of the padding coordinates: a masked
                        # slot need not contain -999 in the input history.
                        gst_traj[:, dest] = original_traj[:, dest]
                        gst_mask[:, dest] = original_mask[:, dest]
                input_restored = (torch.equal(gst_traj, original_traj) and
                                  torch.equal(gst_mask, original_mask))
                if name != 'duplicate_gst_only' or not eligible:
                    require(input_restored, 'Identity/dedup GST input differs from original')
                require(torch.equal(before_rng, torch.get_rng_state()), 'Transform consumed RNG')
                raw_traj, raw_mask = native_predict(predictor, gst_traj, gst_mask, *args, **kwargs)
                require(tuple(raw_traj.shape) == (1, 20, 5, 5), 'Unexpected GST output shape')
                require(tuple(raw_mask.shape) == (1, 20, 1), 'Unexpected GST output mask shape')
                require(bool(torch.isfinite(raw_traj).all()), 'Nonfinite GST output')
                expected_mask = original_mask[:, :, -2] * original_mask[:, :, -1]
                expected_raw_mask = gst_mask[:, :, -2] * gst_mask[:, :, -1]
                require(torch.equal(raw_mask, expected_raw_mask), 'Native output mask semantics changed')
                delivered_traj, delivered_mask = raw_traj, raw_mask
                if name == 'duplicate_gst_only' and eligible:
                    delivered_traj, delivered_mask = raw_traj.clone(), raw_mask.clone()
                    delivered_traj[:, dest, :, :2] = -999.0
                    delivered_mask[:, dest] = 0.0
                    other = torch.arange(20) != dest
                    require(torch.equal(delivered_traj[:, other], raw_traj[:, other]),
                            'Restoration changed another predicted track')
                    require(torch.equal(delivered_traj[:, dest, :, 2:], raw_traj[:, dest, :, 2:]),
                            'Unused uncertainty channels changed')
                require(torch.equal(delivered_mask, expected_mask), 'Actor entity validity changed')
                invalid = ~expected_mask[0, :, 0].bool()
                require(bool((delivered_traj[0, invalid, :, :2] == -999.0).all()),
                        'Invalid trajectory means are not native -999 placeholders')
                require(torch.equal(input_traj, original_traj) and
                        torch.equal(input_binary_mask, original_mask), 'Caller input mutated')
                cache = destination / 'hook_arrays' / f'call_{index:04d}.npz'
                np.savez(cache, original_input_traj=original_traj.numpy(),
                         original_input_mask=original_mask.numpy(),
                         gst_input_traj=gst_traj.numpy(), gst_input_mask=gst_mask.numpy(),
                         raw_output_traj=raw_traj.detach().numpy(),
                         raw_output_mask=raw_mask.detach().numpy(),
                         delivered_output_traj=delivered_traj.detach().numpy(),
                         delivered_output_mask=delivered_mask.detach().numpy())
                record = {'call': index, 'physical_steps_completed': activity['physical_steps_completed'],
                          'eligible': eligible, 'source': source, 'destination': dest,
                          'full_history_sources': sources, 'all_zero_history_destinations': destinations,
                          'injected_before_optional_dedup': injected,
                          'duplicate_reaches_gst': name == 'duplicate_gst_only' and eligible,
                          'gst_input_equals_original': input_restored,
                          'original_inputs_unchanged': True, 'consumer_mask_restored': True,
                          'transform_rng_unchanged': True,
                          'cache': str(cache.relative_to(destination))}
                records.append(record)
                hook_file.write(json.dumps(record, allow_nan=False) + '\n')
                hook_file.flush()
                return delivered_traj, delivered_mask

            def act(policy, obs, hidden, masks, *args, **kwargs):
                record = {'index': summary['actor_captures'], 'obs': snapshot(obs)}
                result = native_act(policy, obs, hidden, masks, *args, **kwargs)
                actor_file.write(json.dumps(record, allow_nan=False) + '\n')
                actor_file.flush()
                summary['actor_captures'] += 1
                return result

            guards.enter_context(patch.object(socket.socket, 'connect', no_network))
            guards.enter_context(patch.object(CrowdNavPredInterfaceMultiEnv, 'forward', predict))
            guards.enter_context(patch.object(Policy, 'act', act))
            native.run(SimpleNamespace(vendor=VENDOR, device='cpu', wall_seconds=285.0),
                       destination, summary)
        trajectory = rows(destination / 'physical_steps.jsonl')
        require(summary.get('outcome') is not None, 'Missing native terminal outcome')
        require(len(trajectory) == summary['outcome']['steps'] == summary['actor_captures'],
                'Incomplete physical episode or unexpected extra actor calls')
        require(bool(trajectory[-1]['done']), 'Episode did not reach native termination')
        require(summary['hook_calls'] == activity['gst_calls'] == len(trajectory) + 1,
                'Unexpected forecast schedule; cannot identify actor-consumed forecasts')
        consumed = records[:len(trajectory)]
        summary.update(physical_return=sum(row['reward'] for row in trajectory),
                       physical_cost=sum(row['cost'] for row in trajectory),
                       physical_terminal=trajectory[-1]['info_type'],
                       eligible_actor_consumed_calls=sum(row['eligible'] for row in consumed),
                       injected_actor_consumed_calls=sum(row['duplicate_reaches_gst'] for row in consumed),
                       first_eligible_actor_consumed_call=next(
                           (row['call'] for row in consumed if row['eligible']), None),
                       post_terminal_forecast_calls_excluded_from_gate=1,
                       status='completed_native_episode')
    except BaseException as error:
        summary.update(status='failed', error_type=type(error).__name__,
                       error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        summary['wall_seconds'] = time.monotonic() - started
        write(destination / 'summary.json', summary)


def exact_caches(left, right):
    import numpy as np
    left_paths = sorted((OUT / left / 'hook_arrays').glob('call_*.npz'))
    right_paths = sorted((OUT / right / 'hook_arrays').glob('call_*.npz'))
    if len(left_paths) != len(right_paths):
        return False
    for a, b in zip(left_paths, right_paths):
        with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
            if x.files != y.files or any(x[key].dtype != y[key].dtype or
                    x[key].shape != y[key].shape or x[key].tobytes() != y[key].tobytes()
                    for key in x.files):
                return False
    return True


def compare():
    import numpy as np
    data = {name: json.loads((OUT / name / 'summary.json').read_text()) for name in ARMS}
    traces = {name: rows(OUT / name / 'physical_steps.jsonl') for name in ARMS}
    require(all(value['status'] == 'completed_native_episode' for value in data.values()),
            'Three completed native episodes are required')
    require(all(traces[name][0]['before'] == traces['original'][0]['before'] for name in ARMS),
            'Initial physical states differ')
    require(all(data[name]['scientific_settings'] == data['original']['scientific_settings']
                for name in ARMS), 'Actor/GST/environment protocol differs')
    clean, attack = data['original'], data['duplicate_gst_only']
    restored = {'physical_steps_exact': traces['original'] == traces['pre_gst_exact_dedup'],
                'actor_observations_exact': rows(OUT / 'original/actor_io.jsonl') ==
                    rows(OUT / 'pre_gst_exact_dedup/actor_io.jsonl'),
                'resets_exact': rows(OUT / 'original/resets.jsonl') ==
                    rows(OUT / 'pre_gst_exact_dedup/resets.jsonl'),
                'forecast_inputs_raw_and_delivered_outputs_exact':
                    exact_caches('original', 'pre_gst_exact_dedup'),
                'outcome_exact': clean['outcome'] == data['pre_gst_exact_dedup']['outcome']}
    difference = attack['physical_return'] - clean['physical_return']
    threshold = .05 * max(1.0, abs(clean['physical_return']))
    material_loss = ((attack['outcome']['collision'] and not clean['outcome']['collision']) or
                     (clean['outcome']['success'] and not attack['outcome']['success']) or
                     difference <= -threshold)
    eligible = attack['injected_actor_consumed_calls'] > 0
    recovery = all(restored.values())
    records = {name: rows(OUT / name / 'interventions.jsonl') for name in ARMS}
    common = next((index for index in range(min(len(traces[name]) for name in ARMS))
                   if all(records[name][index]['eligible'] for name in ARMS)), None)
    diagnostic = None
    if common is not None:
        call = common + 1
        caches = {name: OUT / name / 'hook_arrays' / f'call_{call:04d}.npz' for name in ARMS}
        with np.load(caches['original'], allow_pickle=False) as a, \
                np.load(caches['duplicate_gst_only'], allow_pickle=False) as b:
            same_input = all(np.array_equal(a[key], b[key]) for key in
                             ('original_input_traj', 'original_input_mask'))
            require(same_input, 'First eligible comparison lacks a common factual input')
            clean_obs = rows(OUT / 'original/actor_io.jsonl')[common]['obs']
            changed_obs = rows(OUT / 'duplicate_gst_only/actor_io.jsonl')[common]['obs']
            current_checks = {
                key: clean_obs[key] == changed_obs[key]
                for key in clean_obs if key not in ('spatial_edges', 'conformity_scores')
            }
            current_checks['person_current_xy'] = np.array_equal(
                np.asarray(clean_obs['spatial_edges']['values'])[:, :, :2],
                np.asarray(changed_obs['spatial_edges']['values'])[:, :, :2])
            require(all(current_checks.values()), 'First eligible actor current observation changed')
            real_valid = a['delivered_output_mask'][0, :, 0].astype(bool)
            delta = np.abs(a['raw_output_traj'][0, real_valid, :, :2] -
                           b['raw_output_traj'][0, real_valid, :, :2])
            diagnostic = {'call': call, 'original_inputs_exact': same_input,
                          'actor_current_observation_checks': current_checks,
                          'selected_pairs': {name: [records[name][common]['source'],
                                                    records[name][common]['destination']] for name in ARMS},
                          'original_valid_forecast_rows': np.flatnonzero(real_valid).tolist(),
                          'real_entity_mu_max_abs_difference': float(delta.max()) if delta.size else None,
                          'cache_paths': {name: str(path.relative_to(OUT)) for name, path in caches.items()},
                          'purpose': 'Mechanism diagnostic only; no additional forward; not utility evidence.'}
    decision = ('unsupported_no_eligible_actor_consumed_call' if not eligible else
                'invalid_control_dedup_did_not_exactly_restore' if not recovery else
                'go_material_physical_loss_and_exact_dedup_recovery' if material_loss else
                'close_fixed_duplicate_no_material_loss_no_sweep')
    return {'status': 'completed_three_arms', 'arms': data, 'decision': decision,
            'physical_return_difference': difference,
            'physical_cost_difference': attack['physical_cost'] - clean['physical_cost'],
            'material_return_loss_threshold': threshold, 'material_loss_gate': bool(material_loss),
            'eligible_intervention': eligible, 'dedup_exact_checks': restored,
            'dedup_exact_recovery': recovery, 'go_gate': bool(eligible and recovery and material_loss),
            'first_common_eligible_call': diagnostic,
            'limits': 'One reused case and seed; researcher-added tracker-to-forecast boundary; '
                      'no native external attack, general robustness claim, or standalone utility '
                      'claim from prediction differences. Sigma/correlation of restored padding '
                      'are unconsumed and not restored. No extra episode or search follows closure.'}


def supervise():
    registration_text = (ROOT / 'IDEAS.md').read_text()
    require('## R007 —' in registration_text, 'Register R007 in IDEAS.md before running')
    registration = '## R007 —' + registration_text.split('## R007 —', 1)[1].split('\n## ', 1)[0]
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'launched_script.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'registration.txt').write_text(registration)
    start = time.monotonic()
    receipt = {'status': 'starting', 'pid': os.getpid(), 'hard_wall_seconds': 300,
               'arm_order': list(ARMS), 'workers': [], 'retries': 0, 'device': 'cpu',
               'maximum_physical_episodes': 3}
    write(OUT / 'launcher.json', receipt)
    child = None

    def interrupted(signum, frame):
        raise InterruptedError('Supervisor interrupted')

    def expired(signum, frame):
        raise TimeoutError('Shared 300-second wall budget exhausted; no retry')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, 300 - (time.monotonic() - start)))
        for name in ARMS:
            destination = OUT / name
            destination.mkdir()
            with (destination / 'worker.log').open('x') as log:
                require(time.monotonic() - start < 300, 'Shared 300-second wall budget exhausted')
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--arm', name],
                    cwd=ROOT, env=dict(os.environ, R007_WORKER='1', CUDA_VISIBLE_DEVICES=''),
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
              'material_loss_gate', 'dedup_exact_checks', 'go_gate')}), flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        receipt.update(status='hard_wall_timeout' if isinstance(error, (subprocess.TimeoutExpired, TimeoutError))
                       else 'failed', error_type=type(error).__name__, error=str(error))
        write(OUT / 'summary.json', dict(receipt, no_retry=True))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt['elapsed_seconds'] = time.monotonic() - start
        write(OUT / 'launcher.json', receipt)
    return 0 if receipt['status'] == 'finished' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true', help='Run registered fixed three-arm CPU gate')
    mode.add_argument('--arm', choices=ARMS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run:
        raise SystemExit(supervise())
    worker(args.arm)
