#!/usr/bin/env python3
"""R004 phase2: one fresh native CEM calibration comparison; not an attack."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/native_forecast_calibration_gate_v1'
BIAS = ROOT / 'runs/forecast_residual_calibration_v1/calibrator.npz'
KEYS = ('visual', 'proprio')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    from capture_decision_leverage_native_cem import gpu_info, process_gpu_mib
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'launched_script.py').write_bytes(Path(__file__).read_bytes())
    start, env, torch = time.monotonic(), None, None
    report = {'status': 'starting', 'registration': 'IDEAS.md:R004 phase2', 'pid': os.getpid(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'physical_seed': 3001,
              'warning': 'Single-root exploratory calibration diagnostic, NOT an attack. Added fixed adapter; external provider authority and novelty unproved. Later CEM populations change endogenously; this is not a fixed-population comparison.',
              'native': {'samples': 300, 'iterations': 15, 'elites': 10, 'horizon': 6,
                         'num_act_stepped': 3, 'controls_per_macro': 5, 'local_seed': 1},
              'calibrator': str(BIAS.relative_to(ROOT)), 'coefficient': 1., 'refit': False,
              'primary_rule': 'calibrated native mean H15 terminal reward < raw native mean H15 terminal reward AND mean B2 all-six-future weighted forecast MSE decreases',
              'runtime': {'wall_limit_seconds': 300, 'allocator_cap_gib': 7, 'sampled_process_budget_mib': 8192, 'microbatch': False, 'retries': 0},
              'activity': {'native_query_calls': 0, 'cost_calls': 0, 'unroll_calls': 0, 'predictor_forwards': 0,
                           'encode_calls': 0, 'expert_setup_controls': 0, 'root_warmup_controls': 0,
                           'replay_controls': 0, 'api_calls': 0, 'network_attempts': 0, 'training_updates': 0},
              'checkpoint_loads': {}, 'queries': {}, 'branches': [], 'checks': {}, 'sampled_process_peak_mib': 0}
    def persist():
        report['elapsed_seconds'] = time.monotonic() - start
        write(OUT / 'summary.json', report)
    def expired(*_):
        raise TimeoutError('Registered300-second total limit; no retry')
    def no_network(*_, **__):
        report['activity']['network_attempts'] += 1
        raise RuntimeError('Network forbidden')
    def memory_check():
        used = process_gpu_mib()
        report['sampled_process_peak_mib'] = max(report['sampled_process_peak_mib'], used)
        require(used <= 8192, 'Sampled CUDA process exceeds8GiB')
    signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, 300); persist()
    try:
        report['gpu_at_start'] = gpu_info()
        require(report['gpu_at_start']['memory.free'] >= 8192, 'Insufficient startup GPU memory')
        import numpy as np
        import torch as torch_module
        import mujoco
        from pilot_dino_metaworld_clean_gate import cfg_for, load_model, setup_episode, trace_step, physical, MODEL, DINO_WEIGHTS
        from probe_dino_action_saturation_gate import reconstruct_and_snapshot
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        torch = torch_module
        torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory, 0)
        require(mujoco.__version__ == '3.3.0' and str(ROOT / '.deps/dino_mujoco330') in mujoco.__file__, 'MuJoCo mismatch')
        report['runtime'].update(torch=torch.__version__, mujoco=mujoco.__version__, threads=4, renderer='osmesa', matmul_tf32=False, cudnn_tf32=True)
        with np.load(BIAS, allow_pickle=False) as saved:
            bias = {k: torch.from_numpy(saved['bias_'+k].copy()).cuda() for k in KEYS}
        require(all(v.dtype == torch.float32 and v.shape[:2] == (6, 1) and torch.isfinite(v).all() for v in bias.values()), 'Frozen bias invalid')
        cfg = cfg_for(OUT, 'cuda')
        original_load = torch.load
        def counted_load(path, *args, **kwargs):
            name = str(Path(path).resolve())
            require(name in (str(MODEL.resolve()), str(DINO_WEIGHTS.resolve())) and name not in report['checkpoint_loads'], 'Unexpected/repeated load')
            report['checkpoint_loads'][name] = 1
            return original_load(path, *args, **kwargs)
        with ExitStack() as guards, torch.inference_mode():
            guards.enter_context(patch.object(socket.socket, 'connect', side_effect=no_network))
            guards.enter_context(patch.object(torch.hub, 'download_url_to_file', side_effect=no_network))
            guards.enter_context(patch.object(torch, 'load', side_effect=counted_load))
            report['status'] = 'loading_model'; persist()
            model, pre, report['model_load'] = load_model(cfg, 'cuda')
            require(not model.training and all(not p.requires_grad for p in model.parameters()), 'Model not frozen')
            original_encode, original_unroll = model.encode, model.unroll
            def counted_encode(*args, **kwargs):
                report['activity']['encode_calls'] += 1
                return original_encode(*args, **kwargs)
            def counted_unroll(*args, **kwargs):
                report['activity']['unroll_calls'] += 1
                return original_unroll(*args, **kwargs)
            def count_forward(*_):
                report['activity']['predictor_forwards'] += 1
            model.encode, model.unroll = counted_encode, counted_unroll
            hook = model.model.predictor.register_forward_pre_hook(count_forward)
            env, initial, goal, report['setup'] = setup_episode(cfg, 3001, OUT / 'setup')
            report['activity'].update(expert_setup_controls=100, root_warmup_controls=1)
            with np.load(OUT / 'setup/setup.npz', allow_pickle=False) as saved:
                root = {k: saved[k].copy() for k in saved.files}
            restore, report['restore'] = reconstruct_and_snapshot(env, root, mujoco, torch, np)
            report['restore']['source'] = 'fresh seed3001 full-integration root'
            report['restore'].pop('historical_source_prefix_verified', None)
            def correct(forecast, clone=False):
                result = forecast.clone() if clone else forecast
                for k in KEYS:
                    require(result[k].shape[0] == 7 and result[k].dtype == torch.float32, 'Forecast shape/dtype mismatch')
                    # The raw unroll result is newly allocated; no context or latent recurrent input is edited.
                    result[k][1:].add_(bias[k])
                return result
            def calibrated_unroll(*args, **kwargs):
                return correct(model.unroll(*args, **kwargs))
            captures, means, returned_actions = {}, [], []
            z_init, z_goal = None, None
            for arm in ('raw', 'calibrated'):
                restore()
                agent = GC_Agent(cfg, model, dset=None, preprocessor=pre)
                require(cfg.local_seed == 1, 'Planner seed changed')
                agent.set_goal(goal)
                if z_goal is None:
                    z_goal = agent.goal_state_enc.clone()
                else:
                    require(all(torch.equal(z_goal[k], agent.goal_state_enc[k]) for k in KEYS), 'Goal encoding changed')
                planner = agent.planner
                require((planner.num_samples, planner.iterations, planner.num_elites, planner.horizon, planner.num_act_stepped) == (300,15,10,6,3), 'CEM settings changed')
                require(planner.momentum_mean == planner.momentum_std == 0 and planner.max_norms is None, 'CEM aggregation changed')
                require(planner.objective.alpha == .1 and planner.objective.sum_all_diffs is False, 'Scorer changed')
                if arm == 'calibrated':
                    planner.unroll = calibrated_unroll
                native_cost, capture = planner.cost_function, {'calls': 0}
                def observed_cost(actions, initial_encoding):
                    nonlocal z_init
                    if capture['calls'] == 0:
                        capture['first_population'] = actions.detach().cpu().clone()
                        if z_init is None:
                            z_init = initial_encoding.clone()
                        else:
                            require(all(torch.equal(z_init[k], initial_encoding[k]) for k in KEYS), 'Initial encoding changed')
                    cost = native_cost(actions, initial_encoding)
                    capture['calls'] += 1; report['activity']['cost_calls'] += 1
                    if capture['calls'] == 15:
                        capture['final_population'] = actions.detach().clone()
                        capture['final_costs'] = cost.detach().clone()
                    memory_check(); persist()
                    return cost
                planner.cost_function = observed_cost
                report['status'] = arm+'_native_query'; report['activity']['native_query_calls'] += 1; persist()
                query_start = time.monotonic()
                returned = agent.act(initial, steps_left=max((env.steps_left()+1)*model.action_skip//cfg.frameskip,1))
                torch.cuda.synchronize()
                require(capture['calls'] == 15, 'Incomplete query')
                ids = torch.topk(-capture['final_costs'], 10, dim=0).indices
                mean = planner._prev_mean.detach().clone()
                require(torch.equal(capture['final_population'][:,ids].mean(1), mean), 'Final mean differs from elite mean')
                require(torch.equal(returned.cpu(), mean[:3].cpu()), 'Returned prefix differs')
                means.append(mean); returned_actions.append(returned.cpu().clone())
                captures[arm] = capture
                report['queries'][arm] = {'seconds': time.monotonic()-query_start, 'cost_calls': capture['calls'], 'returned_prefix_exact': True, 'full_mean_exact_to_elites': True}
                np.savez_compressed(OUT / (arm+'_query.npz'), first_population=capture['first_population'].numpy(),
                                    final_population=capture['final_population'].cpu().numpy(), final_costs=capture['final_costs'].cpu().numpy(),
                                    elite_indices=ids.cpu().numpy(), full_mean=mean.cpu().numpy(), returned_prefix=returned.cpu().numpy())
                del agent, planner
                persist()
            require(torch.equal(captures['raw']['first_population'], captures['calibrated']['first_population']), 'First populations differ')
            report['checks']['first_populations_identical'] = True
            actions = torch.stack(means, dim=1)
            requests = pre.denormalize_actions(actions.cpu().permute(1,0,2).reshape(2,30,4))
            for i in range(2):
                require(torch.equal(requests[i,:15], pre.denormalize_actions(returned_actions[i].reshape(15,4))), 'Executed prefix changed')
            require(torch.isfinite(requests).all(), 'Nonfinite controls')
            np.savez_compressed(OUT / 'action_commitment.npz', normalized=actions.cpu().numpy(), physical=requests.numpy())
            write(OUT / 'action_commitment.json', {'both_native_means_committed': True, 'replay_controls_so_far': report['activity']['replay_controls']})
            report['status'] = 'shared_B2_forecast_commitment'; persist()
            raw = model.unroll(z_init, act_suffix=actions.clone())
            calibrated = correct(raw, clone=True)
            frozen = {prefix+'_'+k: values[k].cpu().numpy().copy() for prefix,values in (('raw',raw),('calibrated',calibrated),('initial',z_init),('goal',z_goal)) for k in KEYS}
            require(all(torch.equal(raw[k][:1], calibrated[k][:1]) for k in KEYS), 'Context changed')
            np.savez_compressed(OUT / 'forecast_commitment.npz', **frozen)
            write(OUT / 'forecast_commitment.json', {'replay_controls_so_far': report['activity']['replay_controls'], 'candidate_order': ['raw_native_mean','calibrated_native_mean'], 'primary_error': 'FP64 visualMSE +0.1proprioMSE averaged over six future horizons and same two committed actions'})
            report['status'] = 'actual_replays'; persist()
            pixels, props = [], []
            for branch, name in enumerate(('raw','calibrated')):
                restore(); directory = OUT / name; directory.mkdir()
                rows, frames, states = [], [], []
                with (directory/'trace.jsonl').open('x') as handle:
                    for step in range(30):
                        visual, info, row = trace_step(env, requests[branch,step].clone(), 1, step)
                        report['activity']['replay_controls'] += 1
                        require(not row['done'] and not row['truncated'] and not env.proprio_env.unwrapped._did_see_sim_exception, 'Replay interrupted')
                        require(np.array_equal(physical(env)['target'], root['target']), 'Physical goal changed')
                        rows.append(row); handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush()
                        if (step+1)%5 == 0:
                            frames.append(visual.numpy()[0].copy()); states.append(np.asarray(info['state'][:4],dtype=np.float32).copy())
                pixels.append(np.stack(frames)); props.append(np.stack(states))
                np.savez_compressed(directory/'actual_future.npz',pixels=pixels[-1],proprio=props[-1])
                utilities = {}
                for h in (15,30):
                    selected = rows[:h]
                    utilities['h'+str(h)] = {'terminal_reward': selected[-1]['reward'], 'cumulative_reward': sum(r['reward'] for r in selected),
                        'terminal_success': bool(selected[-1]['success']), 'ever_success': any(r['success'] for r in selected), 'terminal_distance': selected[-1]['reach_distance']}
                report['branches'].append({'arm':name, **utilities}); memory_check(); persist()
            restore(); env.close(); env = None
            report['status'] = 'encoding_actual_future'; persist()
            pixels, props = np.stack(pixels), np.stack(props)
            truth = [model.encode({'visual':torch.from_numpy(pixels[:,h,None]), 'proprio':torch.from_numpy(props[:,h,None])},act=True) for h in range(6)]
            for k in KEYS:
                frozen['truth_'+k] = torch.stack([z[k][:,0] for z in truth]).cpu().numpy().copy()
                require(frozen['truth_'+k].shape[:2] == (6,2), 'Truth alignment changed')
            errors = {}
            for arm in ('raw','calibrated'):
                parts = {k: np.mean((frozen[arm+'_'+k][1:].astype(np.float64)-frozen['truth_'+k].astype(np.float64))**2, axis=tuple(range(2,frozen['truth_'+k].ndim))) for k in KEYS}
                weighted = parts['visual']+.1*parts['proprio']
                require(np.isfinite(weighted).all(), 'Nonfinite error')
                errors[arm] = {'weighted_horizon_candidate':weighted.tolist(), 'mean_all_future_mse':float(weighted.mean()), 'mean_terminal_mse':float(weighted[-1].mean())}
            np.savez_compressed(OUT/'features.npz', **frozen)
            report['forecast_errors'] = errors
            deltas = {h:{k:report['branches'][1][h][k]-report['branches'][0][h][k] for k in report['branches'][0][h]} for h in ('h15','h30')}
            de = errors['calibrated']['mean_all_future_mse']-errors['raw']['mean_all_future_mse']
            du = deltas['h15']['terminal_reward']
            report['utility_deltas_calibrated_minus_raw'] = deltas
            report['joint_gate'] = {'forecast_error_decreases':de<0,'native_H15_utility_decreases':du<0,'passed':de<0 and du<0,'delta_mean_forecast_MSE':de,'delta_H15_terminal_reward':du}
            report['checks'].update(native_returned_prefix_exact=True, context_unchanged=True, same_root_each_replay=True, action_and_forecast_committed_before_replay=True, fixed_saved_bias_no_refit=True)
            require(report['activity']['native_query_calls']==2 and report['activity']['cost_calls']==30 and report['activity']['unroll_calls']==61 and report['activity']['predictor_forwards']==366 and report['activity']['replay_controls']==60, 'Actual call counts differ')
            hook.remove()
            report['status'] = 'completed_single_root_native_calibration_diagnostic'
            report['decision'] = 'joint_condition_observed_no_security_claim' if report['joint_gate']['passed'] else 'joint_condition_absent_stop_no_retry_or_tuning'
    except BaseException as error:
        report.update(status='resource_interrupted' if isinstance(error,TimeoutError) or 'out of memory' in str(error).lower() else 'failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if env is not None:
            env.close()
        if torch is not None and torch.cuda.is_initialized():
            report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat(); persist()
    print(json.dumps({k:report.get(k) for k in ('status','joint_gate','branches','forecast_errors','activity','elapsed_seconds')}),flush=True)


if __name__ == '__main__':
    from capture_decision_leverage_native_cem import bootstrap
    bootstrap(); main()
