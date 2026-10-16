#!/usr/bin/env python3
"""R004 full episode: fixed residual update, one seed; no attack claim.

Default is help. --run launches one bounded experiment; --self-test is CPU-only.
The supervisor hard-stops the worker at600s and writes a receipt even on timeout.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/native_forecast_calibration_episode_v1'
BIAS = ROOT / 'runs/forecast_residual_calibration_v1/calibrator.npz'
REFERENCE = ROOT / 'runs/native_forecast_calibration_gate_v1'
KEYS = ('visual', 'proprio')
QUERY_STEPS = [1, 16, 31, 46, 61, 76, 91]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def apply_saved_residual(forecast, bias, horizon, context_frames):
    """Only post-unroll future outputs change; never autoregressive inputs."""
    require(1 <= horizon <= 6 and context_frames >= 1, 'Unsupported horizon/context')
    for key in KEYS:
        require(forecast[key].shape[0] == context_frames + horizon, 'Unexpected output time axis')
        require(bias[key].shape[:2] == (6, 1), 'Frozen bias layout changed')
        forecast[key][context_frames:].add_(bias[key][:horizon])
    return forecast


def utility(rows):
    require(bool(rows), 'Empty actual trajectory')
    return {'policy_controls': len(rows), 'cumulative_return': sum(r['reward'] for r in rows),
            'terminal_reward': rows[-1]['reward'], 'final_success': bool(rows[-1]['success']),
            'ever_success': any(r['success'] for r in rows), 'terminal_distance': rows[-1]['reach_distance'],
            'first_success_control': next((i+1 for i,r in enumerate(rows) if r['success']), None),
            'terminal_done': bool(rows[-1]['done']), 'terminal_truncated': bool(rows[-1]['truncated'])}


def self_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    for horizon, context in ((6,1),(2,1),(2,2)):
        bias = {k:torch.arange(6,dtype=torch.float32).reshape(6,1,1)+1 for k in KEYS}
        saved = {k:v.clone() for k,v in bias.items()}
        forecast = {k:torch.arange((context+horizon)*3,dtype=torch.float32).reshape(context+horizon,3,1) for k in KEYS}
        original = {k:v.clone() for k,v in forecast.items()}
        result = apply_saved_residual(forecast,bias,horizon,context)
        for k in KEYS:
            require(torch.equal(result[k][:context],original[k][:context]), 'Context changed')
            require(torch.equal(result[k][context:],original[k][context:]+bias[k][:horizon]), 'Horizon prefix mismatch')
            require(torch.equal(saved[k],bias[k]), 'Bias mutated')
    try:
        apply_saved_residual({}, {}, 7, 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Out-of-range horizon accepted')
    require(not torch.cuda.is_initialized(), 'Self-test used CUDA')
    print(json.dumps({'status':'cpu_self_test_passed','horizon_context_cases':3,'rejects_invalid_horizon':True,'model_loads':0,'cuda_initialized':False}))


def supervise():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'launched_script.py').write_bytes(Path(__file__).read_bytes())
    launch = {'status':'starting','started_at_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
              'wall_limit_seconds':600,'run_count':1,'retries':0}
    start = time.monotonic()
    write(OUT/'launcher.json',launch)
    child_env = dict(os.environ, NATIVE_CALIBRATION_EPISODE_WORKER='1')
    with (OUT/'worker.log').open('x') as log:
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
                                   env=child_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch['worker_pid'] = process.pid; write(OUT/'launcher.json',launch)
        try:
            result = process.wait(timeout=max(.001,600-(time.monotonic()-start)))
            launch.update(status='finished',exit_code=result)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            result = process.wait()
            launch.update(status='hard_wall_timeout',exit_code=result)
            summary_path = OUT/'summary.json'
            try:
                summary = json.loads(summary_path.read_text())
            except (FileNotFoundError,json.JSONDecodeError):
                summary = {}
            summary.update(status='hard_wall_timeout',error='Supervisor600-second deadline; worker process group killed. No retry.',
                           last_worker_status=summary.get('status'),finished_at_utc=datetime.now(timezone.utc).isoformat())
            write(summary_path,summary)
        finally:
            launch['elapsed_seconds'] = time.monotonic()-start
            launch['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            write(OUT/'launcher.json',launch)
    print(json.dumps(launch),flush=True)
    return 0 if launch['status']=='finished' and result==0 else 1


def worker():
    require(os.environ.get('NATIVE_CALIBRATION_EPISODE_WORKER')=='1' and OUT.is_dir(), 'Use supervised --run')
    from capture_decision_leverage_native_cem import gpu_info, process_gpu_mib
    start, env, torch = time.monotonic(), None, None
    report = {'status':'starting','registration':'IDEAS.md:R004 complete episode','pid':os.getpid(),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'physical_seed':3001,'arm_order':['raw','calibrated'],
        'warning':'One previously inspected seed, exploratory complete-episode diagnostic; no independent confirmation, attack or population claim. Initial-query forecast MSE is not whole-episode forecast accuracy.',
        'native':{'samples':300,'iterations':15,'elites':10,'horizon':6,'num_act_stepped':3,'controls_per_macro':5,
                  'local_seed_per_arm':1,'episode_steps':100,'warmup_controls':1,'policy_controls_per_arm':99,'done_at_succ':False,
                  'cross_query_warm_start':'Native CEM initializes zero mean/unit std on each plan; _prev_mean only records output. No added warm start.',
                  'observation_history':'Use native latest wrapped observation, including unchanged frame/proprio wrapper state; no added GC_Agent history. Agent RNG persists across its7queries.'},
        'calibrator':str(BIAS.relative_to(ROOT)),'coefficient':1.,'refit':False,
        'short_horizon_rule':'Add frozen horizon-prefix bias to future outputs only; preserve actual context length. Last query naturally uses H2.',
        'primary_rule':'complete99-control cumulative return calibrated < raw minus1e-6',
        'secondary_metrics':['final_success','ever_success','terminal_reward','terminal_distance','every15control cumulative return'],
        'first_query_repetition':'Intentional implementation validation of previous seed3001 first query; not an independent new sample. Require exact returned actions/full mean/initial population.',
        'runtime':{'wall_limit_seconds':600,'worker_signal_seconds':590,'allocator_cap_gib':7,'sampled_process_budget_mib':8192,'microbatch':False,'retries':0},
        'activity':{'native_query_calls':0,'cost_calls':0,'unroll_calls':0,'predictor_forwards':0,'encode_calls':0,
                    'expert_setup_controls':0,'root_warmup_controls':0,'policy_controls':0,'api_calls':0,'network_attempts':0,'training_updates':0},
        'checkpoint_loads':{},'branches':[],'checks':{},'sampled_process_peak_mib':0}
    def persist():
        report['elapsed_seconds'] = time.monotonic()-start
        write(OUT/'summary.json',report)
    def expired(*_):
        raise TimeoutError('Worker590s graceful limit under supervisor600s hard limit; no retry')
    def no_network(*_,**__):
        report['activity']['network_attempts']+=1
        raise RuntimeError('Network forbidden')
    def memory_check():
        used = process_gpu_mib()
        report['sampled_process_peak_mib'] = max(report['sampled_process_peak_mib'],used)
        require(used<=8192,'Sampled CUDA process memory exceeds8GiB')
    signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,expired)
    signal.setitimer(signal.ITIMER_REAL,590); persist()
    try:
        report['gpu_at_start'] = gpu_info()
        require(report['gpu_at_start']['memory.free']>=8192,'Insufficient startup GPU memory')
        import numpy as np
        import torch as torch_module
        import mujoco
        from tensordict import TensorDict
        from pilot_dino_metaworld_clean_gate import cfg_for,load_model,setup_episode,trace_step,physical,MODEL,DINO_WEIGHTS
        from probe_dino_action_saturation_gate import reconstruct_and_snapshot
        from evals.simu_env_planning.planning.gc_agent import GC_Agent
        torch = torch_module
        torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=True
        torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        require(mujoco.__version__=='3.3.0' and str(ROOT/'.deps/dino_mujoco330') in mujoco.__file__,'MuJoCo mismatch')
        report['runtime'].update(torch=torch.__version__,mujoco=mujoco.__version__,threads=4,renderer='osmesa',matmul_tf32=False,cudnn_tf32=True)
        with np.load(BIAS,allow_pickle=False) as saved:
            bias={k:torch.from_numpy(saved['bias_'+k].copy()).cuda() for k in KEYS}
        require(all(v.dtype==torch.float32 and v.shape[:2]==(6,1) and torch.isfinite(v).all() for v in bias.values()),'Frozen bias invalid')
        cfg=cfg_for(OUT,'cuda')
        require(cfg.local_seed==1 and cfg.task_specification.max_episode_steps==100 and not cfg.task_specification.done_at_succ and not cfg.planner.repeat_actskip,'Episode settings changed')
        original_load=torch.load
        def counted_load(path,*args,**kwargs):
            name=str(Path(path).resolve())
            require(name in (str(MODEL.resolve()),str(DINO_WEIGHTS.resolve())) and name not in report['checkpoint_loads'],'Unexpected/repeated load')
            report['checkpoint_loads'][name]=1
            return original_load(path,*args,**kwargs)
        with ExitStack() as guards,torch.inference_mode():
            guards.enter_context(patch.object(socket.socket,'connect',side_effect=no_network))
            guards.enter_context(patch.object(torch.hub,'download_url_to_file',side_effect=no_network))
            guards.enter_context(patch.object(torch,'load',side_effect=counted_load))
            report['status']='loading_model'; persist()
            model,pre,report['model_load']=load_model(cfg,'cuda')
            require(not model.training and all(not p.requires_grad for p in model.parameters()),'Model not frozen')
            original_encode,original_unroll=model.encode,model.unroll
            def counted_encode(*args,**kwargs):
                report['activity']['encode_calls']+=1
                return original_encode(*args,**kwargs)
            def counted_unroll(*args,**kwargs):
                report['activity']['unroll_calls']+=1
                return original_unroll(*args,**kwargs)
            def predictor_hook(*_):
                report['activity']['predictor_forwards']+=1
            model.encode,model.unroll=counted_encode,counted_unroll
            hook=model.model.predictor.register_forward_pre_hook(predictor_hook)
            env,initial,goal,report['setup']=setup_episode(cfg,3001,OUT/'setup')
            report['activity'].update(expert_setup_controls=100,root_warmup_controls=1)
            with np.load(OUT/'setup/setup.npz',allow_pickle=False) as saved:
                root={k:saved[k].copy() for k in saved.files}
            with np.load(REFERENCE/'setup/setup.npz',allow_pickle=False) as previous:
                require(set(previous.files)==set(root) and all(np.array_equal(root[k],previous[k]) for k in root),'Root/goal differs from phase2')
            report['checks']['same_root_goal_as_phase2']=True
            restore,report['restore']=reconstruct_and_snapshot(env,root,mujoco,torch,np)
            report['restore']['source']='Fresh seed3001 reconstruction checked against phase2; one full snapshot reused between arms only'
            report['restore'].pop('historical_source_prefix_verified',None)
            for arm in ('raw','calibrated'):
                restore(); obs=initial.clone()
                directory=OUT/arm; directory.mkdir()
                branch={'arm':arm,'status':'running','queries':[],'checkpoints':[],'first_query_reference_checks':{}}
                report['branches'].append(branch)
                agent=GC_Agent(cfg,model,dset=None,preprocessor=pre)
                agent.set_goal(goal)
                planner=agent.planner
                require((planner.num_samples,planner.iterations,planner.num_elites,planner.horizon,planner.num_act_stepped)==(300,15,10,6,3),'CEM geometry changed')
                require(planner.momentum_mean==planner.momentum_std==0 and planner.max_norms is None and planner.objective.alpha==.1 and planner.objective.sum_all_diffs is False,'Native scorer/aggregation changed')
                if arm=='calibrated':
                    def calibrated_unroll(z_ctxt,act_suffix=None,**kwargs):
                        output=model.unroll(z_ctxt,act_suffix=act_suffix,**kwargs)
                        return apply_saved_residual(output,bias,act_suffix.shape[0],z_ctxt['visual'].shape[1])
                    planner.unroll=calibrated_unroll
                native_cost=planner.cost_function
                first_population=None
                def observed_cost(actions,z_init):
                    nonlocal first_population
                    if first_population is None:
                        first_population=actions.detach().cpu().clone()
                    costs=native_cost(actions,z_init)
                    report['activity']['cost_calls']+=1
                    memory_check(); persist()
                    return costs
                planner.cost_function=observed_cost
                rows=[]
                with (directory/'trace.jsonl').open('x') as trace:
                    while env.steps_left()>0:
                        query_index=len(branch['queries'])+1
                        elapsed=env.elapsed_steps()
                        require(elapsed==QUERY_STEPS[query_index-1],'Unexpected native query schedule')
                        plan_left=max((env.steps_left()+1)*model.action_skip//cfg.frameskip,1)
                        report['status']=arm+'_query'+str(query_index)
                        report['activity']['native_query_calls']+=1; persist()
                        tick=time.monotonic(); before_calls=report['activity']['cost_calls']
                        normalized=agent.act(obs,steps_left=plan_left)
                        torch.cuda.synchronize()
                        require(report['activity']['cost_calls']-before_calls==15,'Incomplete query')
                        require(torch.equal(normalized.cpu(),planner._prev_mean[:3].cpu()),'Returned prefix differs from native mean')
                        actions=pre.denormalize_actions(normalized.cpu().reshape(-1,4))
                        require(torch.isfinite(actions).all(),'Nonfinite native controls')
                        if query_index==1:
                            with np.load(REFERENCE/(arm+'_query.npz'),allow_pickle=False) as old:
                                checks={'returned_prefix_exact':np.array_equal(normalized.cpu().numpy(),old['returned_prefix']),
                                        'full_mean_exact':np.array_equal(planner._prev_mean.cpu().numpy(),old['full_mean']),
                                        'first_population_exact':np.array_equal(first_population.numpy(),old['first_population'])}
                            branch['first_query_reference_checks']=checks
                            require(all(checks.values()),'First query does not reproduce phase2; stop without tuning')
                        query={'index':query_index,'elapsed_before':elapsed,'planning_steps_left':plan_left,'horizon':min(6,plan_left),
                               'normalized_macro_actions':normalized.cpu().tolist(),'physical_actions':actions.tolist(),
                               'planned_control_count':len(actions),'executed_control_count':0,'cost_calls':15,
                               'minimum_cost_per_iteration':agent._prev_losses.cpu().tolist(),'elapsed_seconds':time.monotonic()-tick}
                        branch['queries'].append(query)
                        # Commit each returned sequence before its actual environment execution.
                        write(directory/'queries.json',branch['queries']); persist()
                        for index,action in enumerate(actions):
                            visual,info,row=trace_step(env,action,query_index,index)
                            report['activity']['policy_controls']+=1; query['executed_control_count']+=1
                            require(not env.proprio_env.unwrapped._did_see_sim_exception,'Simulator exception')
                            require(np.array_equal(physical(env)['target'],root['target']),'Original target changed')
                            require(not row['done'] or env.elapsed_steps()==100,'Unexpected early termination')
                            rows.append(row); trace.write(json.dumps(row,allow_nan=False)+'\n'); trace.flush()
                            obs=TensorDict({'visual':visual,'proprio':info['proprio']})
                            if len(rows)%15==0 or row['done']:
                                branch['checkpoints'].append(utility(rows)); persist()
                            if row['done']:
                                break
                        write(directory/'queries.json',branch['queries']); memory_check(); persist()
                        print(json.dumps({'arm':arm,'query':query_index,'policy_controls':len(rows),'cumulative_return':sum(r['reward'] for r in rows),'success':rows[-1]['success']}),flush=True)
                require(len(rows)==99 and env.elapsed_steps()==100 and rows[-1]['done'] and rows[-1]['truncated'],'Native complete episode not reached')
                require([q['elapsed_before'] for q in branch['queries']]==QUERY_STEPS,'Query schedule differs')
                require([q['horizon'] for q in branch['queries']]==[6,6,6,6,6,5,2],'Native shrinking horizon differs')
                branch.update(status='completed',utility=utility(rows))
                write(directory/'result.json',branch); persist()
                del agent,planner,native_cost,observed_cost
            hook.remove(); env.close(); env=None
            raw,calibrated=[b['utility'] for b in report['branches']]
            delta=calibrated['cumulative_return']-raw['cumulative_return']
            report['primary']={'raw_episode_return':raw['cumulative_return'],'calibrated_episode_return':calibrated['cumulative_return'],
                               'delta_calibrated_minus_raw':delta,'return_loss_threshold':1e-6,'calibrated_return_lower':delta < -1e-6}
            report['secondary_deltas']={k:calibrated[k]-raw[k] for k in ('terminal_reward','terminal_distance','final_success','ever_success')}
            report['checks'].update(complete_native_episode_each_arm=True,per_arm_continuous_rng=True,no_added_history_or_warm_start=True,
                                    saved_bias_fixed_no_refit=True,context_preserved_by_adapter=True,first_query_exact_to_phase2=True)
            require(report['activity']['native_query_calls']==14 and report['activity']['cost_calls']==210 and report['activity']['unroll_calls']==420
                    and report['activity']['predictor_forwards']==2220 and report['activity']['encode_calls']==16 and report['activity']['policy_controls']==198,'Actual activity differs from registered full episodes')
            report['status']='completed_single_seed_full_native_episode_calibration'
            report['decision']='full_episode_return_loss_observed_no_attack_claim' if delta < -1e-6 else 'full_episode_return_loss_absent_close_fixed_diagnostic'
    except BaseException as error:
        report.update(status='resource_interrupted' if isinstance(error,TimeoutError) or 'out of memory' in str(error).lower() else 'failed',
                      error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            if env is not None:
                env.close()
        finally:
            if torch is not None and torch.cuda.is_initialized():
                report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
                report['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
            report['finished_at_utc']=datetime.now(timezone.utc).isoformat(); persist()
    print(json.dumps({k:report.get(k) for k in ('status','primary','secondary_deltas','activity','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run',action='store_true')
    modes.add_argument('--self-test',action='store_true')
    modes.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    elif args.run:
        raise SystemExit(supervise())
    else:
        from capture_decision_leverage_native_cem import bootstrap
        bootstrap(); worker()
