"""Test a published rejected-program split in the native frontier consumer.

This is a sensitivity check, not a generated attack: group contexts come from
the explicitly counterfactual published-program replay. Physical warmup and
candidate outcomes use the native simulator. The current program is fixed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import MethodType

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/BabaInWonderland"
sys.path.insert(0, str(UPSTREAM))
from src.agents.graph_contrastive_agent import GraphContrastiveAgent
from src.data import Transition
from src.data.state_store import canonical_state_key
from src.environments.baba_wrapper import BabaWrapper
from src.program_model.evaluator import ProgramEvaluator


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@dataclass
class EnvironmentFactory:
    variant: str
    seed: int
    maps: tuple[str, ...] = ("custom_map_07", "custom_map_31")

    def __call__(self):
        aliases = yaml.safe_load((UPSTREAM / "configs/experiment_config_online.yaml").read_text())["serialization"]["word_aliases"]
        return BabaWrapper(env_name="env/baba_custom_ascii_original", max_steps=300, seed=self.seed,
            env_kwargs={"encoding_level":1, "allowed_scenario_types":list(self.maps),
                        "split_manifest_path":str(UPSTREAM / "configs/custom_map_splits_dot_test.yaml")},
            word_aliases=aliases if self.variant == "wonderland" else None)


def parameter_hash(module):
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def enable_verified_activation_checkpoint(agent):
    """Recompute each unchanged encoding bucket in backward, preserving the loss."""
    from torch.utils.checkpoint import checkpoint

    indices=list(range(min(8,len(agent.sample_store))))
    agent.contrastive_optimizer.zero_grad(set_to_none=True)
    original_keys=agent._encode_samples_as_keys_from_indices(indices)
    weights=torch.linspace(-1,1,original_keys.numel(),device=original_keys.device).reshape_as(original_keys)
    (original_keys*weights).sum().backward()
    original_grads={name:p.grad.detach().clone() for name,p in agent.dynamics.named_parameters() if p.grad is not None}
    original_keys=original_keys.detach().clone()
    agent.contrastive_optimizer.zero_grad(set_to_none=True)
    original_encode=agent.dynamics.encode_state_action

    def encode(model,tokens,mask,actions):
        if torch.is_grad_enabled():
            return checkpoint(original_encode,tokens,mask,actions,use_reentrant=False,preserve_rng_state=True)
        return original_encode(tokens,mask,actions)

    agent.dynamics.encode_state_action=MethodType(encode,agent.dynamics)
    checkpoint_keys=agent._encode_samples_as_keys_from_indices(indices)
    (checkpoint_keys*weights).sum().backward()
    checkpoint_grads={name:p.grad.detach() for name,p in agent.dynamics.named_parameters() if p.grad is not None}
    gradient_keys_match=original_grads.keys()==checkpoint_grads.keys()
    grad_max=max((float((original_grads[n]-checkpoint_grads[n]).abs().max()) for n in original_grads),default=0.0)
    result={'samples':len(indices),'outputs_equal':torch.equal(original_keys,checkpoint_keys),
        'gradient_parameter_sets_equal':gradient_keys_match,'gradient_max_abs_error':grad_max,
        'gradients_equal':gradient_keys_match and all(torch.equal(original_grads[n],checkpoint_grads[n]) for n in original_grads),
        'optimizer_updates':0}
    agent.contrastive_optimizer.zero_grad(set_to_none=True)
    if not result['outputs_equal'] or not result['gradients_equal']:
        raise RuntimeError('Activation checkpoint equivalence check failed: '+json.dumps(result))
    return result


def context(variant, stage):
    root = ROOT / "runs/baba_rejection_feedback_replay_v0" / variant
    versions = [{"version_id":p.stem,"source":p.read_text(),"index":int(p.stem[1:])}
                for p in sorted((root / "native/program_versions").glob("v*.py"))]
    return {"versions":versions, "current_version_id":versions[-1]["version_id"],
            "current_source":versions[-1]["source"],
            "group_context":json.loads((root/f"context_{stage}.json").read_text())}


def score(agent, top_k):
    agent._ensure_prototypes_current()
    sessions = list(agent._active_worlds)
    push_order = agent._frontier_push_order
    started = time.monotonic()
    work = agent._build_scored_worklists_for_collect(sessions=sessions,
        budgets={s.world_index:top_k for s in sessions}, progress_callback=None)
    agent._frontier_push_order = push_order
    rows = [{"world":world, "rank":rank, "state_id":item.state_id,
             "state_key":item.state_key, "action":item.action, "score":item.score,
             "rh":item.rh,"rz":item.rz_pred}
            for world, items in sorted(work.items()) for rank,item in enumerate(items)]
    info = {"sample_count":len(agent.sample_store),
            "class_counts":agent.sample_store.class_counts(include_zero=True),
            "active_class_ids":list(agent._group_context_snapshot.active_class_ids),
            "mature_class_ids":sorted(agent._mature_prototype_class_ids),
            "frontier_count":agent._global_frontier_size(), "score_elapsed_s":time.monotonic()-started,
            "encoder_hash":parameter_hash(agent.dynamics_state_encoder),
            "rows":rows}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["default","wonderland"], default="wonderland")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=4096)
    parser.add_argument("--top-k",type=int,default=64)
    parser.add_argument("--activation-checkpoint",action="store_true")
    parser.add_argument("--maps",nargs="+",default=["custom_map_07","custom_map_31"])
    parser.add_argument("--min-split-support",type=int,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'script.py').write_bytes(Path(__file__).read_bytes())
    start=time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    train_maps=yaml.safe_load((UPSTREAM/'configs/custom_map_splits_dot_test.yaml').read_text())['original']['all_train']
    if not set(args.maps)<=set(train_maps):
        raise ValueError('This development diagnostic accepts training maps only.')
    factory=EnvironmentFactory(args.variant,args.seed,tuple(args.maps))
    env=factory()
    config=yaml.safe_load((UPSTREAM/'src/agents/configs/graph_contrastive_agent.yaml').read_text())['agent']['params']
    config['collect_workers']=2
    evaluator=ProgramEvaluator(program_eval_workers=8)
    agent=GraphContrastiveAgent(env=env,seed=args.seed,device='cuda',dashboard_enabled=False,
                               evaluator=evaluator,collect_env_factory=factory,**config)
    try:
        run(args, factory, agent, evaluator, config, start)
    except Exception as exc:
        save(args.output/'failure.json', {'error':repr(exc),'native_collected_steps':agent.total_steps,
            'native_train_updates_completed':agent.total_updates,'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated(),
            'api_requests':0,'wall_time_s':time.monotonic()-start})
        raise
    finally:
        agent.close()
        evaluator.close()


def run(args, factory, agent, evaluator, config, start):
    before,after=context(args.variant,'before'),context(args.variant,'after')
    assert before['current_source']==after['current_source']
    save(args.output/'setup.json',{'variant':args.variant,'seed':args.seed,'config':config,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'warmup_steps':args.warmup,'top_k_per_world':args.top_k,'prototype_entropy_min_class_count':agent.prototype_entropy_min_class_count,
        'activation_checkpoint':args.activation_checkpoint,
        'maps':args.maps,'minimum_split_support':args.min_split_support,
        'source_context':'runs/baba_rejection_feedback_replay_v0/'+args.variant,
        'current_program_sha256':hashlib.sha256(before['current_source'].encode()).hexdigest(),
        'scope':'Immediate native frontier consequence of published-program rejection; no newly generated attack or task-performance claim.'})
    agent.set_program_context(before)
    last=[0.0]
    def progress(event):
        if time.monotonic()-last[0]>10:
            print(json.dumps({'phase':'warmup','steps':agent.total_steps,'elapsed_s':time.monotonic()-start}),flush=True)
            last[0]=time.monotonic()
    collected=[]
    batches=[]
    with gzip.open(args.output/'warmup.jsonl.gz','wt') as handle:
        while agent.total_steps<args.warmup:
            batch=agent.collect(max_transitions=min(256,args.warmup-agent.total_steps),progress_callback=progress)
            collected.extend(batch)
            for t in batch:
                handle.write(json.dumps({'state':t.state,'action':t.action,'next_state':t.next_state,
                    'reward':t.reward,'done':t.done,'world_index':t.world_index,'map_name':t.map_name})+'\n')
            handle.flush()
            batches.append({'batch_steps':len(batch),'native_total_steps':agent.total_steps,
                            'recorded_steps':len(collected),'elapsed_s':time.monotonic()-start})
            save(args.output/'collection_progress.json',batches)
            if not batch:
                raise RuntimeError('Native collection returned no transitions before requested warmup.')
    agent._close_collect_actor_workers()
    assert agent.total_steps==len(collected)==args.warmup
    print(json.dumps({'phase':'warmup_complete','steps':agent.total_steps,'samples':len(agent.sample_store),
                      'class_counts':agent.sample_store.class_counts(include_zero=True)}),flush=True)
    truth_env=factory()
    truth_env.reset()
    restore_checks=[]
    for world in sorted({t.world_index for t in collected}):
        t=next(t for t in reversed(collected) if t.world_index==world)
        state_id=agent.state_store.lookup_state_id(t.state)
        node=agent._all_world_sessions[world].state_nodes[state_id]
        truth_env.restore_runtime_packet(agent.state_store.runtime_state_packet(state_id),
            agent.state_store.runtime_state_vocab(),step_count=node.depth)
        next_state,reward,terminated,truncated,_=truth_env.step(agent.action_names.index(t.action))
        restore_checks.append({'world':world,'state_id':state_id,'action':t.action,
            'state_matches':canonical_state_key(next_state)==canonical_state_key(t.next_state),
            'reward_matches':reward==t.reward,'done_matches':(terminated or truncated)==t.done})
    save(args.output/'restore_checks.json',restore_checks)
    assert all(r['state_matches'] and r['reward_matches'] and r['done_matches'] for r in restore_checks)
    if agent.total_steps<agent.learning_starts:
        save(args.output/'summary.json',{'status':'below_native_warmup','collected_steps':len(collected),
            'restore_check_steps':len(restore_checks),'native_steps_total':len(collected)+len(restore_checks),
            'train_updates_completed':0,'api_requests':0,'wall_time_s':time.monotonic()-start})
        return
    if args.min_split_support:
        sources=[g['split_program_source'] for g in after['group_context']['group_hierarchy']
                 if g.get('split_program_source')]
        assert len(sources)==1, 'This support gate handles a single replayed split.'
        p_eval=evaluator.evaluate_source(before['current_source'],collected)
        q_eval=evaluator.evaluate_source(sources[0],collected)
        assert not p_eval.compile_errors and not q_eval.compile_errors
        assert not p_eval.runtime_error_count and not q_eval.runtime_error_count
        support=sum(p.is_correct and not q.is_correct for p,q in zip(p_eval.records,q_eval.records))
        gate={'new_class_support':support,'required_support':args.min_split_support,
              'passed':support>=args.min_split_support,'program_transition_evaluations':2*len(collected)}
        save(args.output/'support_gate.json',gate)
        print(json.dumps({'phase':'support_gate',**gate}),flush=True)
        if not gate['passed']:
            save(args.output/'summary.json',{'status':'split_support_gate_failed','support_gate':gate,
                'warmup_native_steps':len(collected),'restore_check_steps':len(restore_checks),
                'native_steps_total':len(collected)+len(restore_checks),'train_updates_completed':0,
                'api_requests':0,'wall_time_s':time.monotonic()-start,
                'completed_at_utc':datetime.now(timezone.utc).isoformat()})
            truth_env.close()
            return
    if args.activation_checkpoint:
        save(args.output/'activation_checkpoint_check.json',enable_verified_activation_checkpoint(agent))
    train_start=time.monotonic()
    train_result=agent.finalize_explained_transition_batch(collected)
    torch.cuda.synchronize()
    train_time=time.monotonic()-train_start
    print(json.dumps({'phase':'training_complete','result':train_result,'elapsed_s':train_time}),flush=True)
    control=score(agent,args.top_k)
    save(args.output/'control_scores.json',control)
    agent.set_program_context(after)
    rejected=score(agent,args.top_k)
    save(args.output/'rejected_scores.json',rejected)
    agent.set_program_context(before)
    restored=score(agent,args.top_k)
    save(args.output/'restored_scores.json',restored)
    key=lambda row:(row['world'],row['state_key'],row['action'])
    all_candidates={key(row):row for r in [control,rejected] for row in r['rows']}
    result_by_key={}
    diagnostic_transitions=[]
    for k,row in all_candidates.items():
        node=agent._all_world_sessions[row['world']].state_nodes[row['state_id']]
        truth_env.restore_runtime_packet(agent.state_store.runtime_state_packet(row['state_id']),
            agent.state_store.runtime_state_vocab(),step_count=node.depth)
        next_state,reward,terminated,truncated,_=truth_env.step(row['action'])
        diagnostic_transitions.append(Transition(state=agent.state_store.state_json(row['state_id']),
            action=agent.action_names[row['action']],next_state=next_state,reward=reward,done=terminated or truncated))
    evaluation=evaluator.evaluate_source(before['current_source'],diagnostic_transitions)
    with gzip.open(args.output/'truth_probes.jsonl.gz','wt') as handle:
        for k,t,record in zip(all_candidates,diagnostic_transitions,evaluation.records):
            handle.write(json.dumps({'world':k[0],'state_key':k[1],'action_id':k[2],
                'state':t.state,'action':t.action,'next_state':t.next_state,'reward':t.reward,
                'done':t.done,'program_correct':record.is_correct})+'\n')
    for k,record in zip(all_candidates,evaluation.records):
        result_by_key[k]=not record.is_correct
    comparisons={}
    for world in sorted({r['world'] for r in control['rows']}):
        c=[key(r) for r in control['rows'] if r['world']==world]
        q=[key(r) for r in rejected['rows'] if r['world']==world]
        z=[key(r) for r in restored['rows'] if r['world']==world]
        comparisons[str(world)]={'selected':len(c),'same_order':c==q,'overlap':len(set(c)&set(q)),
            'control_counterexamples':sum(result_by_key[k] for k in c),
            'rejected_counterexamples':sum(result_by_key[k] for k in q),
            'restoration_same_order':c==z}
    summary={'status':'completed','variant':args.variant,'seed':args.seed,'warmup_native_steps':len(collected),
        'diagnostic_native_steps':len(diagnostic_transitions),'restore_check_steps':len(restore_checks),
        'native_steps_total':len(collected)+len(diagnostic_transitions)+len(restore_checks),
        'training':train_result,'train_elapsed_s':train_time,'comparisons':comparisons,
        'encoder_unchanged_by_context':control['encoder_hash']==rejected['encoder_hash']==restored['encoder_hash'],
        'program_unchanged':agent._current_program_source==before['current_source'],
        'sample_counts':[control['sample_count'],rejected['sample_count'],restored['sample_count']],
        'class_counts':[control['class_counts'],rejected['class_counts'],restored['class_counts']],
        'mature_class_ids':[control['mature_class_ids'],rejected['mature_class_ids'],restored['mature_class_ids']],
        'gpu_peak_memory_bytes':torch.cuda.max_memory_allocated(),'api_requests':0,
        'wall_time_s':time.monotonic()-start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'One warmed-up native explorer, immediate group-context intervention and restoration. Counterexample yield is not autonomous task return. Candidate split is published-program replay, not attacker-generated.'}
    save(args.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    truth_env.close()


if __name__=='__main__':
    main()
