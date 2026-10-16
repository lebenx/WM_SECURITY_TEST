#!/usr/bin/env python3
"""R005: cached TD-MPC2 dynamics split by native proposal/scoring consumer.

Four complete native episodes, no training. Use --run after registration.
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

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/tdmpc_proposal_scoring_factorial_v1'
SOURCE=ROOT/'runs/tdmpc_frozen_head_loss_control_20261015'
ARMS=('RR','CC','CR','RC')
EXPECTED={'RR':612.082555025816,'CC':495.2163214981556}
DYNAMICS={'R':SOURCE/'consistency_random/dynamics.pt','C':SOURCE/'consistency_cascade/dynamics.pt'}
sys.path.insert(0,str(ROOT/'src'))


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def supervise():
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'launched_script.py').write_bytes(Path(__file__).read_bytes())
    start=time.monotonic()
    receipt={'status':'starting','pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),'hard_wall_seconds':300,'retries':0}
    write(OUT/'launcher.json',receipt)
    def interrupted(signum,_):
        raise InterruptedError('Supervisor received signal'+str(signum))
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    with (OUT/'worker.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
            env=dict(os.environ,R005_WORKER='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['worker_pid']=child.pid; write(OUT/'launcher.json',receipt)
        try:
            code=child.wait(timeout=max(.001,300-(time.monotonic()-start)))
            receipt.update(status='finished',exit_code=code)
        except (subprocess.TimeoutExpired,InterruptedError) as error:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL)
            code=child.wait()
            receipt.update(status='hard_wall_timeout' if isinstance(error,subprocess.TimeoutExpired) else 'interrupted',exit_code=code)
            try:
                data=json.loads((OUT/'summary.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):
                data={}
            data.update(last_worker_status=data.get('status'),status=receipt['status'],error=str(error),no_retry=True)
            write(OUT/'summary.json',data)
        finally:
            receipt.update(elapsed_seconds=time.monotonic()-start,finished_at_utc=datetime.now(timezone.utc).isoformat())
            write(OUT/'launcher.json',receipt)
    print(json.dumps(receipt),flush=True)
    return 0 if receipt['status']=='finished' and code==0 else 1


def worker():
    require(os.environ.get('R005_WORKER')=='1' and OUT.is_dir(),'Use --run supervisor')
    from pilot_tdmpc_frozen_head_loss_control import evaluate,frozen_equal,gpu_metadata
    start=time.monotonic(); torch=None
    summary={'status':'starting','registration':'IDEAS.md:R005','pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'question':'Does the existing conditional source-update loss pass mainly through the native policy-proposal dynamics consumer?',
        'warning':'Single reused development source/task/seed; path-intervention diagnostic, not a new external attack or population claim. R is a fixed random-source dynamics comparator, not a true-value oracle. Route effects alone do not prove that beneficial actions were absent from the finite proposal bank.',
        'source_dynamics':{k:str(v.relative_to(ROOT)) for k,v in DYNAMICS.items()},
        'task':'walker-run','task_id':2,'evaluation_seed':31,'arm_order':list(ARMS),
        'arm_definition':'First letter selects dynamics for native policy proposals; second selects dynamics inside _estimate_value. All other original weights are identical and frozen.',
        'native':{'horizon':3,'num_samples':512,'iterations':6,'num_elites':64,'num_pi_trajs':24,'eval_mode':False,'compile':False,'controls_per_arm':500},
        'protected':'Original plan/model.next bodies, reward/Q/actor/encoder/task embedding weights, task, state reset, random-seed reset, candidate algorithm, MPPI warm start and action mask. Candidate values, later populations and physical states can change endogenously.',
        'primary_rule':'Both RR/CC exactly reproduce historical trajectories and returns; G=J_RR-J_CC>0; J_RR-J_CR>=0.5G AND J_RC-J_CC>=0.5G. Otherwise close without new seeds, thresholds or training.',
        'runtime':{'hard_wall_seconds':300,'worker_signal_seconds':290,'allocator_cap_gib':4,'minimum_startup_free_mib':4096,'sampled_process_budget_mib':6144,'retries':0},
        'activity':{'native_load_agent_calls':0,'adapted_dynamics_loads':0,'physical_evaluation_controls':0,'optimizer_updates':0,'source_environment_steps':0,'api_calls':0,'network_attempts':0},
        'arms':{},'sampled_process_peak_mib':0}
    def persist():
        summary['elapsed_seconds']=time.monotonic()-start
        summary['activity']['physical_evaluation_controls']=sum(a.get('physical_controls_recorded',0) for a in summary['arms'].values())
        write(OUT/'summary.json',summary)
    def expired(*_):
        raise TimeoutError('Worker290-second deadline under300-second hard supervisor; no retry')
    def deadline():
        if time.monotonic()-start>290:
            expired()
    def no_network(*_,**__):
        summary['activity']['network_attempts']+=1
        raise RuntimeError('Network forbidden')
    def memory_check():
        text=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5)
        used=0
        for line in text.splitlines():
            pid,memory=[x.strip() for x in line.split(',')]
            if int(pid)==os.getpid(): used=int(memory)
        summary['sampled_process_peak_mib']=max(summary['sampled_process_peak_mib'],used)
        require(used<=6144,'Sampled process memory exceeded6GiB')
    signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,expired)
    signal.setitimer(signal.ITIMER_REAL,290); persist()
    try:
        summary['gpu_at_start']=gpu_metadata(0)
        require(summary['gpu_at_start']['free_mib']>=4096,'Insufficient startup GPU memory')
        import torch as torch_module
        from wm_security import tdmpc_native as native
        torch=torch_module
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(4*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
        with ExitStack() as guards:
            guards.enter_context(patch.object(socket.socket,'connect',side_effect=no_network))
            summary['status']='loading_original_checkpoint'; persist()
            summary['activity']['native_load_agent_calls']+=1
            agent,summary['provenance']=native.load_agent(ROOT/'models/tdmpc2/mt30-48M.pt')
            cfg=agent.cfg
            require((cfg.horizon,cfg.num_samples,cfg.iterations,cfg.num_elites,cfg.num_pi_trajs)==(3,512,6,64,24) and not cfg.compile and not cfg.episodic,'Native planner configuration differs')
            require(cfg.tasks.index('walker-run')==2 and cfg.action_dims[2]==cfg.action_dim==6,'Task/action mapping differs')
            original={k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else deepcopy(v) for k,v in agent.model.state_dict().items()}
            reference_dynamics=agent.model._dynamics
            modules={}; saved_states={}
            for key,path in DYNAMICS.items():
                state=torch.load(path,map_location='cpu',weights_only=True)
                module=deepcopy(reference_dynamics)
                module.load_state_dict(state,strict=True); module.eval().requires_grad_(False)
                require(all(torch.equal(v.detach().cpu(),state[k]) for k,v in module.state_dict().items()),'Saved dynamics not loaded exactly')
                modules[key]=module; saved_states[key]=state; summary['activity']['adapted_dynamics_loads']+=1
            agent.model.eval().requires_grad_(False)
            native_estimate,native_next=agent._estimate_value,agent.model.next
            spec={'task':'walker-run','task_id':2,'seed':31,'steps':500,'policy':'mpc'}
            first_query_banks={}
            for name in ARMS:
                deadline(); destination=OUT/name; destination.mkdir()
                proposal,scorer=modules[name[0]],modules[name[1]]
                agent.model._dynamics=proposal
                with torch.no_grad(): agent._prev_mean.zero_()
                arm={'id':name,'status':'evaluating','proposal_dynamics':name[0],'scoring_dynamics':name[1],
                     'physical_controls_recorded':0,'calls':{'proposal_next':0,'scoring_next':0,'estimate_value':0},'route':'proposal'}
                summary['arms'][name]=arm
                arm['frozen_before']=frozen_equal(agent.model,original,torch)
                def routed_estimate(*args,**kwargs):
                    require(arm['route']=='proposal' and agent.model._dynamics is proposal,'Unexpected scoring nesting')
                    if arm['calls']['estimate_value']==0:
                        # Observe the existing first population before scoring; no draws or model calls.
                        cpu_rng=torch.get_rng_state().clone()
                        cuda_rng=torch.cuda.get_rng_state(agent.device).clone()
                        actions=args[1] if len(args)>1 else kwargs['actions']
                        require(tuple(actions.shape)==(3,512,6),'First population shape differs')
                        bank=actions.detach().cpu().clone()
                        first_query_banks[name]=bank
                        torch.save({'all_actions':bank,'prior_first_actions':bank[0,:24].clone(),
                                    'prior_future_tail':bank[1:,:24].clone(),'gaussian_first_iteration':bank[:,24:].clone(),
                                    'query_index':0,'iteration_index':0,'task_id':2,'seed':31},
                                   destination/'first_query_candidate_bank.pt')
                        reference=first_query_banks['RR']
                        tail_delta=bank[1:,:24]-reference[1:,:24]
                        first_checks={'prior_first_actions_exact_to_RR':torch.equal(bank[0,:24],reference[0,:24]),
                                      'all488_gaussian_trajectories_exact_to_RR':torch.equal(bank[:,24:],reference[:,24:]),
                                      'capture_preserves_cpu_rng':torch.equal(cpu_rng,torch.get_rng_state()),
                                      'capture_preserves_cuda_rng':torch.equal(cuda_rng,torch.cuda.get_rng_state(agent.device))}
                        proposal_reference=first_query_banks['RR' if name[0]=='R' else 'CC']
                        first_checks['same_proposal_pair_full_bank_exact']=torch.equal(bank,proposal_reference)
                        arm['first_query_population_checks']=first_checks
                        arm['first_query_prior_tail_diagnostic']={
                            'scope':'Future steps1..2 of24 policy-prior trajectories at the shared physical first query; descriptive only.',
                            'exact_to_RR':torch.equal(bank[1:,:24],reference[1:,:24]),
                            'changed_scalar_count':int((tail_delta!=0).sum()),
                            'max_absolute_difference':float(tail_delta.abs().max()),
                            'mean_absolute_difference':float(tail_delta.abs().mean()),
                            'new_rng_draws':0,'extra_model_calls':0}
                        persist()
                        require(all(first_checks.values()),'Shared first-action/Gaussian population or observation RNG differs')
                    arm['route']='scoring'; agent.model._dynamics=scorer
                    arm['calls']['estimate_value']+=1
                    try:
                        return native_estimate(*args,**kwargs)
                    finally:
                        agent.model._dynamics=proposal; arm['route']='proposal'
                def observed_next(z,action,task):
                    route=arm['route']
                    require(z.shape[0]==(24 if route=='proposal' else 512),'Unexpected native dynamics consumer batch')
                    require(agent.model._dynamics is (proposal if route=='proposal' else scorer),'Wrong dynamics route')
                    arm['calls'][route+'_next']+=1
                    return native_next(z,action,task)
                agent._estimate_value=routed_estimate; agent.model.next=observed_next
                summary['status']='evaluating_'+name; persist(); memory_check()
                try:
                    result=evaluate(agent,spec,destination,arm,native,deadline,persist,torch)
                finally:
                    agent._estimate_value=native_estimate; agent.model.next=native_next
                    agent.model._dynamics=proposal; arm['route']='proposal'
                arm['evaluation']=result
                arm['frozen_after']=frozen_equal(agent.model,original,torch)
                require(arm['calls']=={'proposal_next':1000,'scoring_next':9000,'estimate_value':3000},'Native path call budget differs')
                require(all(torch.equal(value.detach().cpu(),saved_states[key][field]) for key in ('R','C') for field,value in modules[key].state_dict().items()),'Saved dynamics mutated during evaluation')
                if name in EXPECTED:
                    prior='consistency_random' if name=='RR' else 'consistency_cascade'
                    old=torch.load(SOURCE/prior/'evaluation.pt',map_location='cpu',weights_only=False)
                    new=torch.load(destination/'evaluation.pt',map_location='cpu',weights_only=False)
                    checks={field:torch.equal(new[field],old[field]) for field in ('obs','action','reward')}
                    checks['return']=result['return']==EXPECTED[name]
                    checks['steps']=result['steps']==500
                    arm['historical_exact_checks']=checks; persist()
                    require(all(checks.values()),'RR/CC historical reconstruction failed; stop without changed tolerance')
                arm['status']='completed'; memory_check(); persist()
            returns={k:v['evaluation']['return'] for k,v in summary['arms'].items()}
            gap=returns['RR']-returns['CC']
            proposal_loss=returns['RR']-returns['CR']
            score_fixed_recovery=returns['RC']-returns['CC']
            summary['analysis']={'returns':returns,'original_gap':gap,'proposal_only_loss_RR_minus_CR':proposal_loss,
                'proposal_restoration_gain_RC_minus_CC':score_fixed_recovery,
                'proposal_loss_fraction':proposal_loss/gap if gap>0 else None,
                'proposal_restoration_fraction':score_fixed_recovery/gap if gap>0 else None,
                'passed':gap>0 and proposal_loss>=.5*gap and score_fixed_recovery>=.5*gap,
                'statistics':'One reused source23/evaluation31 development case; descriptive, no bootstrap/ASR/population inference.'}
            # C-minus-R switch effects, retaining all interactions instead of claiming additive mediation.
            proposal_at_R=returns['CR']-returns['RR']
            proposal_at_C=returns['CC']-returns['RC']
            scorer_at_R=returns['RC']-returns['RR']
            scorer_at_C=returns['CC']-returns['CR']
            summary['analysis']['factorial_effects']={
                'sign_convention':'Switch the named route from R to C; negative means lower true episode return.',
                'proposal_switch_with_R_scorer':proposal_at_R,
                'proposal_switch_with_C_scorer':proposal_at_C,
                'scorer_switch_with_R_proposal':scorer_at_R,
                'scorer_switch_with_C_proposal':scorer_at_C,
                'average_proposal_switch_effect':.5*(proposal_at_R+proposal_at_C),
                'average_scorer_switch_effect':.5*(scorer_at_R+scorer_at_C),
                'interaction_CC_minus_CR_minus_RC_plus_RR':returns['CC']-returns['CR']-returns['RC']+returns['RR'],
                'total_switch_CC_minus_RR':returns['CC']-returns['RR'],
                'interpretation':'Descriptive2x2 consumer intervention; trajectory/population changes are endogenous. Interaction prevents treating route effects as independent causal shares.'}
            summary['status']='completed_native_consumer_factorial'
            summary['decision']='registered_route_condition_observed_not_coverage_or_attack_proof' if summary['analysis']['passed'] else 'registered_route_condition_absent_close_no_sweep'
            with (OUT/'results.csv').open('x',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=['arm','proposal','scorer','return','steps'])
                writer.writeheader()
                for key in ARMS:writer.writerow({'arm':key,'proposal':key[0],'scorer':key[1],'return':returns[key],'steps':500})
    except BaseException as error:
        summary.update(status='resource_interrupted' if isinstance(error,TimeoutError) or 'out of memory' in str(error).lower() else 'failed',
                       error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if torch is not None and torch.cuda.is_initialized():
            summary['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
            summary['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
        summary['finished_at_utc']=datetime.now(timezone.utc).isoformat(); persist()
    print(json.dumps({k:summary.get(k) for k in ('status','analysis','decision','activity','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true')
    mode.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.run:raise SystemExit(supervise())
    worker()
