#!/usr/bin/env python3
"""R008: fixed TD-MPC2 reward/terminal-state route diagnostic, not a new attack.

Four 500-control episodes with C proposal dynamics and one shared 300-second
hard wall budget. Register IDEAS.md:R008 before --run. No training, API calls,
network, repeated checkpoint hashing, retries, or source/seed/weight searches.
"""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from run_tdmpc_proposal_scoring_factorial import DYNAMICS, require, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/tdmpc_reward_terminal_factorial_v1'
HISTORY = ROOT / 'runs/tdmpc_proposal_scoring_factorial_v1'
ARMS = ('CC', 'RR', 'CR', 'RC')
ANCHORS = {'CC': ('CC', 495.2163214981556), 'RR': ('CR', 674.8858879804611)}
sys.path.insert(0, str(ROOT / 'src'))


def supervise():
    registration = (ROOT / 'IDEAS.md').read_text()
    require('## R008 —' in registration, 'Register R008 in IDEAS.md before execution')
    registration = '## R008 —' + registration.split('## R008 —', 1)[1].split('\n## ', 1)[0]
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'registration.txt').write_text(registration)
    (OUT / 'launched_script.py').write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    receipt = {'status': 'starting', 'pid': os.getpid(), 'hard_wall_seconds': 300,
               'retries': 0, 'maximum_physical_episodes': 4, 'arm_order': list(ARMS)}
    write(OUT / 'launcher.json', receipt)
    child = None

    def interrupted(signum, frame):
        raise InterruptedError('Supervisor interrupted')

    def expired(signum, frame):
        raise TimeoutError('Shared 300-second wall budget exhausted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, max(.001, 300 - (time.monotonic() - started)))
        with (OUT / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                cwd=ROOT, env=dict(os.environ, R008_WORKER='1'), stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(status='running', worker_pid=child.pid)
            write(OUT / 'launcher.json', receipt)
            code = child.wait(timeout=max(.001, 300 - (time.monotonic() - started)))
            receipt.update(status='finished' if code == 0 else 'worker_failed', exit_code=code)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        receipt.update(status='hard_wall_timeout' if isinstance(error, (TimeoutError, subprocess.TimeoutExpired))
                       else 'failed', error_type=type(error).__name__, error=str(error))
        try:
            summary = json.loads((OUT / 'summary.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            summary = {}
        summary.update(last_worker_status=summary.get('status'), status=receipt['status'],
                       error=str(error), no_retry=True, efficacy_conclusion=None)
        write(OUT / 'summary.json', summary)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt['elapsed_seconds'] = time.monotonic() - started
        write(OUT / 'launcher.json', receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['status'] == 'finished' else 1


def worker():
    require(os.environ.get('R008_WORKER') == '1' and OUT.is_dir(), 'Use --run supervisor')
    from pilot_tdmpc_frozen_head_loss_control import evaluate, frozen_equal, gpu_metadata
    started = time.monotonic()
    torch = None
    summary = {'status': 'starting', 'registration': 'IDEAS.md:R008', 'pid': os.getpid(),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Research inference-route intervention within the existing scoring route; '
                 'not a new security mechanism, external attacker interface, or three-system proof.',
        'arm_order': list(ARMS), 'proposal_dynamics': 'C',
        'arm_definition': 'First character selects dynamics producing reward-input states; '
                          'second selects dynamics producing the terminal state fed to BOTH '
                          'the unchanged pi and Q. All reward/pi/Q weights are identical.',
        'warning': 'Mixed scores combine rewards and terminal continuation from different '
                   'dynamics rollouts of the same candidate. They are researcher-defined '
                   'path interventions, not coherent single-dynamics native values. Candidate '
                   'banks after the first iteration and physical trajectories change endogenously.',
        'task': 'walker-run', 'task_id': 2, 'seed': 31,
        'source_dynamics': {key: str(path.relative_to(ROOT)) for key, path in DYNAMICS.items()},
        'native': {'horizon': 3, 'num_samples': 512, 'iterations': 6, 'num_elites': 64,
                   'num_pi_trajs': 24, 'eval_mode': False, 'compile': False, 'controls_per_arm': 500},
        'primary_rule': 'D=J_RR-J_CC>0; J_CR>=J_CC+0.5D AND J_RC<J_CC+0.5D. '
                        'Report recovery ratios without necessity/sufficiency or additive attribution.',
        'runtime': {'hard_wall_seconds': 300, 'worker_signal_seconds': 290,
                    'allocator_cap_gib': 4, 'minimum_startup_free_mib': 4096,
                    'sampled_process_budget_mib': 6144, 'retries': 0},
        'activity': {'native_load_agent_calls': 0, 'adapted_dynamics_loads': 0,
                     'physical_evaluation_controls': 0, 'optimizer_updates': 0,
                     'source_environment_steps': 0, 'api_calls': 0, 'network_attempts': 0,
                     'checkpoint_hashes': 0, 'additional_diagnostic_model_forwards': 0},
        'arms': {}, 'sampled_process_peak_mib': 0}

    def persist():
        summary['elapsed_seconds'] = time.monotonic() - started
        summary['activity']['physical_evaluation_controls'] = sum(
            arm.get('physical_controls_recorded', 0) for arm in summary['arms'].values())
        write(OUT / 'summary.json', summary)

    def expired(*args):
        raise TimeoutError('Worker 290-second deadline under shared 300-second supervisor; no retry')

    def deadline():
        if time.monotonic() - started > 290:
            expired()

    def no_network(*args, **kwargs):
        summary['activity']['network_attempts'] += 1
        raise RuntimeError('Network forbidden')

    def memory_check():
        result = subprocess.check_output(['nvidia-smi', '-i', '0',
            '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
            text=True, timeout=5)
        used = 0
        for line in result.splitlines():
            pid, memory = [part.strip() for part in line.split(',')]
            if int(pid) == os.getpid():
                used = int(memory)
        summary['sampled_process_peak_mib'] = max(summary['sampled_process_peak_mib'], used)
        require(used <= 6144, 'Sampled process GPU memory exceeded 6 GiB')

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, 290)
    persist()
    try:
        summary['gpu_at_start'] = gpu_metadata(0)
        require(summary['gpu_at_start']['free_mib'] >= 4096, 'Insufficient startup GPU memory')
        import torch as torch_module
        from wm_security import tdmpc_native as native
        torch = torch_module
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(
            4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket, 'connect', no_network))
            summary['status'] = 'loading_original_checkpoint'
            summary['activity']['native_load_agent_calls'] += 1
            persist()
            agent, summary['provenance'] = native.load_agent(
                ROOT / 'models/tdmpc2/mt30-48M.pt', verify_checkpoint=False)
            require(summary['provenance'].get('checkpoint_hash_checked') is False,
                    'Loader must explicitly report skipped redundant checkpoint hashing')
            from common import math as native_math
            cfg, model = agent.cfg, agent.model
            require((cfg.horizon, cfg.num_samples, cfg.iterations, cfg.num_elites, cfg.num_pi_trajs)
                    == (3, 512, 6, 64, 24) and not cfg.compile and not cfg.episodic,
                    'Native planner configuration differs')
            require(cfg.tasks.index('walker-run') == 2 and cfg.action_dims[2] == cfg.action_dim == 6,
                    'Task/action mapping differs')
            original = {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor)
                        else deepcopy(value) for key, value in model.state_dict().items()}
            modules, saved_states = {}, {}
            for key, path in DYNAMICS.items():
                state = torch.load(path, map_location='cpu', weights_only=True)
                module = deepcopy(model._dynamics)
                module.load_state_dict(state, strict=True)
                module.eval().requires_grad_(False)
                require(all(torch.equal(value.detach().cpu(), state[field])
                            for field, value in module.state_dict().items()), 'Saved dynamics load differs')
                modules[key], saved_states[key] = module, state
                summary['activity']['adapted_dynamics_loads'] += 1
            model.eval().requires_grad_(False)
            native_estimate, native_next = agent._estimate_value, model.next
            native_pi, native_Q, native_reward = model.pi, model.Q, model.reward
            old_bank = torch.load(HISTORY / 'CC/first_query_candidate_bank.pt',
                                  map_location='cpu', weights_only=False)['all_actions']
            spec = {'task': 'walker-run', 'task_id': 2, 'seed': 31, 'steps': 500, 'policy': 'mpc'}
            first_rng = None
            for name in ARMS:
                deadline()
                destination = OUT / name
                destination.mkdir()
                reward_module, terminal_module = modules[name[0]], modules[name[1]]
                proposal_module = modules['C']
                model._dynamics = proposal_module
                with torch.no_grad():
                    agent._prev_mean.zero_()
                arm = {'id': name, 'status': 'evaluating', 'reward_dynamics': name[0],
                       'terminal_state_dynamics': name[1], 'proposal_dynamics': 'C',
                       'physical_controls_recorded': 0, 'route': 'proposal',
                       'calls': {key: 0 for key in ('proposal_next', 'reward_next', 'terminal_shadow_next',
                           'proposal_pi', 'terminal_pi', 'terminal_Q', 'reward', 'estimate_value')},
                       'shadow_rng_and_embedding_checks': 0}
                summary['arms'][name] = arm
                arm['frozen_before'] = frozen_equal(model, original, torch)
                frame = {}

                def observed_next(z, action, task):
                    route = arm['route']
                    require(route in ('proposal', 'reward', 'terminal_shadow'), 'Unexpected dynamics route')
                    expected = {'proposal': proposal_module, 'reward': reward_module,
                                'terminal_shadow': terminal_module}[route]
                    require(model._dynamics is expected, 'Wrong dynamics module')
                    require(z.shape[0] == (24 if route == 'proposal' else 512), 'Unexpected dynamics batch')
                    arm['calls'][route + '_next'] += 1
                    return native_next(z, action, task)

                def observed_reward(z, action, task):
                    require(arm['route'] == 'reward', 'Reward outside native scoring rollout')
                    arm['calls']['reward'] += 1
                    result = native_reward(z, action, task)
                    if frame['capture']:
                        frame['reward_logits'].append(result.detach().clone())
                    return result

                def terminal_pi(z, task):
                    if arm['route'] == 'proposal':
                        arm['calls']['proposal_pi'] += 1
                        return native_pi(z, task)
                    require(arm['route'] == 'reward' and frame['pi_calls'] == 0,
                            'Unexpected terminal pi route/count')
                    frame['pi_calls'] += 1
                    arm['calls']['terminal_pi'] += 1
                    terminal_z = z
                    if name[0] != name[1]:
                        before_cpu = torch.get_rng_state().clone()
                        before_cuda = torch.cuda.get_rng_state(agent.device).clone()
                        before_embedding = model._task_emb.weight.detach().clone()
                        arm['route'] = 'terminal_shadow'
                        model._dynamics = terminal_module
                        try:
                            terminal_z = frame['initial_z']
                            for t in range(cfg.horizon):
                                terminal_z = model.next(terminal_z, frame['actions'][t], task)
                        finally:
                            model._dynamics = reward_module
                            arm['route'] = 'reward'
                        require(torch.equal(before_cpu, torch.get_rng_state()) and
                                torch.equal(before_cuda, torch.cuda.get_rng_state(agent.device)),
                                'Shadow dynamics consumed RNG; do not mask this failure with fork_rng')
                        require(torch.equal(before_embedding, model._task_emb.weight),
                                'Shadow task-embedding lookup mutated shared state')
                        arm['shadow_rng_and_embedding_checks'] += 1
                    frame['terminal_z'] = terminal_z
                    if frame['capture']:
                        frame['reward_terminal_z'] = z.detach().cpu().clone()
                        frame['terminal_input_z'] = terminal_z.detach().cpu().clone()
                        frame['terminal_rng_before'] = (torch.get_rng_state().clone(),
                                                       torch.cuda.get_rng_state(agent.device).clone())
                    # Exactly one native Gaussian draw, using the same random
                    # stream and tensor shape as the unmodified estimator.
                    result = native_pi(terminal_z, task)
                    if frame['capture']:
                        frame['terminal_action'] = result[0].detach().cpu().clone()
                    return result

                def terminal_Q(z, action, task, *args, **kwargs):
                    require(arm['route'] == 'reward' and frame['pi_calls'] == 1 and frame['q_calls'] == 0,
                            'Unexpected terminal Q route/count')
                    require(kwargs.get('return_type') == 'avg', 'Native terminal Q reduction changed')
                    frame['q_calls'] += 1
                    arm['calls']['terminal_Q'] += 1
                    # Q receives the SAME redirected terminal state as pi.
                    # Its native two-head randperm is called exactly once.
                    result = native_Q(frame['terminal_z'], action, task, *args, **kwargs)
                    if frame['capture']:
                        frame['terminal_q'] = result.detach().clone()
                        frame['terminal_rng_after'] = (torch.get_rng_state().clone(),
                                                      torch.cuda.get_rng_state(agent.device).clone())
                    return result

                def routed_estimate(z, actions, task):
                    nonlocal first_rng
                    require(arm['route'] == 'proposal' and model._dynamics is proposal_module,
                            'Unexpected scoring nesting')
                    capture = arm['calls']['estimate_value'] == 0
                    arm['calls']['estimate_value'] += 1
                    frame.clear()
                    frame.update(initial_z=z, actions=actions, capture=capture, reward_logits=[],
                                 pi_calls=0, q_calls=0)
                    if capture:
                        bank = actions.detach().cpu().clone()
                        require(tuple(bank.shape) == (3, 512, 6) and torch.equal(bank, old_bank),
                                'First common candidate bank differs from existing C-proposal bank')
                        arm['first_candidate_bank_exact_to_R005_CC'] = True
                    model._dynamics = reward_module
                    arm['route'] = 'reward'
                    try:
                        # Preserve native estimator implementation and arithmetic.
                        result = native_estimate(z, actions, task)
                        require(bool(torch.isfinite(result).all()),
                                'Nonfinite native score; do not treat nan_to_num as an efficacy result')
                        require(frame['pi_calls'] == frame['q_calls'] == 1,
                                'Terminal stochastic calls are not exactly one each')
                        if capture:
                            before = (torch.get_rng_state().clone(),
                                      torch.cuda.get_rng_state(agent.device).clone())
                            G, discount = 0, 1
                            termination = torch.zeros(cfg.num_samples, 1, dtype=torch.float32, device=z.device)
                            require(len(frame['reward_logits']) == cfg.horizon, 'Missing native reward captures')
                            for logits in frame['reward_logits']:
                                reward = native_math.two_hot_inv(logits, cfg)
                                G = G + discount * (1 - termination) * reward
                                update = agent.discount[torch.tensor(task)] if cfg.multitask else agent.discount
                                discount = discount * update
                            tail = discount * (1 - termination) * frame['terminal_q']
                            require(torch.equal(G + tail, result), 'Diagnostic decomposition differs from native score')
                            rng = {key: frame[key] for key in ('terminal_rng_before', 'terminal_rng_after')}
                            if first_rng is None:
                                first_rng = rng
                            require(all(torch.equal(a, b) for key in rng
                                        for a, b in zip(rng[key], first_rng[key])),
                                    'First native terminal random streams differ across arms')
                            torch.save({'candidate_bank': bank, 'initial_z': z.detach().cpu(),
                                'reward_terminal_z': frame['reward_terminal_z'],
                                'terminal_input_z': frame['terminal_input_z'],
                                'terminal_action': frame['terminal_action'],
                                'discounted_reward': G.detach().cpu(), 'discount': discount.detach().cpu(),
                                'terminal_Q': frame['terminal_q'].detach().cpu(),
                                'terminal_contribution': tail.detach().cpu(), 'score': result.detach().cpu(),
                                'rng': rng, 'query_index': 0, 'iteration_index': 0,
                                'interpretation': 'Read-only first-population decomposition; no extra model calls.'},
                                destination / 'first_query_components.pt')
                            require(torch.equal(before[0], torch.get_rng_state()) and
                                    torch.equal(before[1], torch.cuda.get_rng_state(agent.device)),
                                    'Diagnostic capture consumed RNG')
                            arm['first_terminal_rng_exact_across_arms'] = True
                            arm['first_score_decomposition_exact'] = True
                        return result
                    finally:
                        model._dynamics = proposal_module
                        arm['route'] = 'proposal'
                        frame.clear()

                summary['status'] = 'evaluating_' + name
                persist()
                memory_check()
                with ExitStack() as hooks:
                    hooks.enter_context(patch.object(agent, '_estimate_value', routed_estimate))
                    hooks.enter_context(patch.object(model, 'next', observed_next))
                    hooks.enter_context(patch.object(model, 'reward', observed_reward))
                    hooks.enter_context(patch.object(model, 'pi', terminal_pi))
                    hooks.enter_context(patch.object(model, 'Q', terminal_Q))
                    try:
                        result = evaluate(agent, spec, destination, arm, native, deadline, persist, torch)
                    finally:
                        model._dynamics = proposal_module
                        arm['route'] = 'proposal'
                arm['evaluation'] = result
                persist()  # Retain every completed return even if a later invariant fails.
                expected_calls = {'proposal_next': 1000, 'reward_next': 9000,
                    'terminal_shadow_next': 9000 if name[0] != name[1] else 0,
                    'proposal_pi': 1500, 'terminal_pi': 3000, 'terminal_Q': 3000,
                    'reward': 9000, 'estimate_value': 3000}
                require(arm['calls'] == expected_calls, 'Native/shadow call budget differs')
                require(arm['shadow_rng_and_embedding_checks'] == (3000 if name[0] != name[1] else 0),
                        'Incomplete shadow RNG/shared-state checks')
                arm['frozen_after'] = frozen_equal(model, original, torch)
                require(all(torch.equal(value.detach().cpu(), saved_states[key][field])
                    for key in modules for field, value in modules[key].state_dict().items()),
                    'Saved dynamics mutated')
                if name in ANCHORS:
                    old_name, old_return = ANCHORS[name]
                    old = torch.load(HISTORY / old_name / 'evaluation.pt', map_location='cpu', weights_only=False)
                    new = torch.load(destination / 'evaluation.pt', map_location='cpu', weights_only=False)
                    checks = {key: torch.equal(new[key], old[key]) for key in ('obs', 'action', 'reward')}
                    checks.update(return_exact=result['return'] == old_return, steps_exact=result['steps'] == 500)
                    arm['historical_anchor'] = 'R005 ' + old_name
                    arm['historical_exact_checks'] = checks
                    persist()
                    require(all(checks.values()), 'Same-source historical anchor failed; no tolerance change')
                arm['status'] = 'completed'
                memory_check()
                persist()
            returns = {name: arm['evaluation']['return'] for name, arm in summary['arms'].items()}
            gap = returns['RR'] - returns['CC']
            midpoint = returns['CC'] + .5 * gap
            terminal_recovery = returns['CR'] - returns['CC']
            reward_recovery = returns['RC'] - returns['CC']
            passed = gap > 0 and returns['CR'] >= midpoint and returns['RC'] < midpoint
            summary['analysis'] = {'returns': returns, 'D_RR_minus_CC': gap,
                'midpoint': midpoint, 'terminal_restoration_gain_CR_minus_CC': terminal_recovery,
                'reward_restoration_gain_RC_minus_CC': reward_recovery,
                'terminal_restoration_fraction': terminal_recovery / gap if gap > 0 else None,
                'reward_restoration_fraction': reward_recovery / gap if gap > 0 else None,
                'terminal_half_gap_inclusive': returns['CR'] >= midpoint,
                'reward_below_half_gap_strict': returns['RC'] < midpoint,
                'interaction_RR_minus_CR_minus_RC_plus_CC': returns['RR'] - returns['CR'] - returns['RC'] + returns['CC'],
                'registered_dominant_recovery_condition': bool(passed),
                'interpretation': 'One fixed development case. Conditional restoration gains are '
                                  'not additive causal shares, necessity/sufficiency, or a new attack.'}
            summary['status'] = 'completed_reward_terminal_factorial'
            summary['decision'] = ('registered_terminal_dominant_recovery_observed_diagnostic_only'
                                   if passed else 'registered_condition_absent_close_no_sweep')
            summary['efficacy_conclusion'] = 'diagnostic_condition_observed' if passed else 'diagnostic_condition_absent'
            with (OUT / 'results.csv').open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['arm', 'proposal', 'reward_dynamics',
                                                           'terminal_dynamics', 'return', 'steps'])
                writer.writeheader()
                for name in ARMS:
                    writer.writerow({'arm': name, 'proposal': 'C', 'reward_dynamics': name[0],
                        'terminal_dynamics': name[1], 'return': returns[name], 'steps': 500})
    except BaseException as error:
        summary.update(status='resource_interrupted' if isinstance(error, TimeoutError) or
                       'out of memory' in str(error).lower() else 'technical_failure',
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                       efficacy_conclusion=None, no_retry=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if torch is not None and torch.cuda.is_initialized():
            summary['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            summary['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
        summary['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        persist()
    print(json.dumps({key: summary[key] for key in ('status', 'decision', 'analysis', 'activity')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true', help='Run the registered fixed four-arm diagnostic')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run:
        raise SystemExit(supervise())
    worker()
