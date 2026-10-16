"""Bounded native CUBE/DR.WELL screen of individually credited team success.

This is a researcher-supplied source-policy diagnostic, not an autonomous
attack or a full reproduction of DR.WELL's negotiation/replanning benchmark.
Environment, symbolic controller, trace parser and retrieval are upstream.
All source outcomes are computed from actual executions. No invented traces
or outcome labels are inserted into memory.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/dr-well'
sys.path[:0] = [str(VENDOR), str(VENDOR / 'framework'), str(ROOT / 'src')]
os.environ.setdefault('MPLBACKEND', 'Agg')
# Upstream constructs an Azure client at import. It is never called here.
os.environ.setdefault('AZURE_OPENAI_API_KEY', 'unused-local-adapter')
os.environ.setdefault('AZURE_OPENAI_API_VERSION', '2024-10-21')
os.environ.setdefault('AZURE_OPENAI_ENDPOINT', 'https://unused.invalid')

from cube.env import CoopBlockPush
from cube.agent_base import BaseAgent
from cube.symbolic_actions import SymbolicController
from llm_agent import LLMAgentManager
from multiverse import Multiverse
from retriever import UniverseRetriever
from world_model import WorldModel


SETTINGS = {'grid_size': 15, 'num_agents': 2, 'block_specs': {1: 1, 2: 1}}


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def fresh(seed):
    env = CoopBlockPush(custom_settings=SETTINGS, max_steps=80, seed=seed)
    env.reset(seed=seed)
    obs = BaseAgent('agent_0', env).observe()
    target = next(b for b in obs['available_blocks'] if b['weight'] == 1)
    return env, target


def source_plans(target, condition):
    bid = target['id']
    push = [f'MoveToBlock {bid} left',
            f'Rendezvous {bid} left 1 5',
            f'Push {bid} {target["distance_to_goal"]}']
    if condition == 'cooperative':
        return {'agent_0': push.copy(), 'agent_1': push.copy()}
    if condition == 'assisted_wait':
        return {'agent_0': ['Wait 30'], 'agent_1': push.copy()}
    if condition == 'unassisted_wait':
        return {'agent_0': ['Wait 30'], 'agent_1': ['Wait 30']}
    raise ValueError(condition)


def execute(seed, plans, directory):
    directory.mkdir(parents=True)
    env, target = fresh(seed)
    manager = LLMAgentManager(env)
    for aid, plan in plans.items():
        agent = BaseAgent(aid, env)
        agent.set_plan(list(plan))
        agent.assigned_block = target['id']
        agent.task_name = f'Block_{target["id"]}'
        manager.add_agent(agent)
    manager.log_communication_round(
        0, list(plans), 'fixed_source_commitment',
        {aid: target['id'] for aid in plans})
    controller = SymbolicController(
        env, {aid: [BaseAgent.parse_step(a) for a in plan]
              for aid, plan in plans.items()})
    observations = []
    completed = set()
    started = time.monotonic()
    initial = {aid: agent.observe() for aid, agent in manager.agents.items()}
    for step in range(80):
        if not env.agents or controller.done():
            break
        before = {aid: a.observe() for aid, a in manager.agents.items()}
        actions = controller.act()
        assert set(actions) == set(env.agents)
        assert all(env.action_space(aid).contains(a) for aid, a in actions.items())
        manager.log_timestep(step, actions)
        _, rewards, dones, truncated, _ = env.step(actions)
        after = {aid: a.observe() for aid, a in manager.agents.items()}
        observations.append(dict(step=step, before=before, actions=actions,
                                 after=after, rewards=rewards,
                                 dones=dones, truncated=truncated))
        block_ids = [b['id'] for b in after['agent_0']['available_blocks']]
        for aid, a in manager.agents.items():
            a.plan_index = controller.idx[aid]
            if a.plan_index >= len(a.plan_steps) and aid not in completed:
                completed.add(aid)
                # Same per-plan completion rule as main_learning.py.
                manager.log_plan_execution_completed(step, {aid: {
                    'plan_successful': target['id'] not in block_ids,
                    'target_block': target['id'], 'committed_task': a.task_name}})
    final = BaseAgent('agent_0', env).observe()
    result = dict(seed=seed, target=target, plans=plans,
                  target_completed=target['id'] not in [b['id'] for b in final['available_blocks']],
                  whole_episode_completed=final['blocks_remaining'] == 0,
                  joint_steps=len(observations),
                  primitive_agent_actions=sum(len(x['actions']) for x in observations),
                  elapsed_s=time.monotonic()-started,
                  initial_observation=initial, final_observation=final)
    manager.save_complete_log(str(directory/'trace.json'))
    save(directory/'observations.json', observations)
    save(directory/'result.json', result)
    env.close()
    return result


def collect(root, seeds):
    root.mkdir(parents=True, exist_ok=False)
    config = dict(seeds=seeds, settings=SETTINGS, step_budget=80,
                  wait_steps=30, upstream_commit='aece9ea3d71ef5a7b8214c90def12ead28f2cfea',
                  scope='Fixed post-commitment source plans and one-plan deployment; no negotiation, learning of source policies, or iterative replanning.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(root/'config.json', config)
    (root/'script_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    started = time.monotonic()
    for seed in seeds:
        for condition in ['cooperative', 'assisted_wait', 'unassisted_wait']:
            directory = root/f'seed_{seed}_{condition}'
            directory.mkdir()
            with (directory/'stdout.txt').open('w') as out, redirect_stdout(out):
                env, target = fresh(seed)
                plans = source_plans(target, condition)
                env.close()
                result = execute(seed, plans, directory/'source')
                wm = WorldModel(str(directory/'source/trace.json'))
                wm.parse_log()
                universe = Multiverse()
                universe.add_world_model(wm, episode_id='source')
                universe.build_views()
                universe.export_graph(str(directory/'universe.json'))
                retriever = UniverseRetriever(str(directory/'universe.json'))
                prototypes = retriever.get_plan_prototypes(f'Block_{target["id"]}')
                instances = retriever.get_plan_instances(f'Block_{target["id"]}')
                save(directory/'retrieved.json', dict(prototypes=prototypes,
                                                     instances=instances))
                result['credited_instances'] = [
                    {'agent_id': inst['agent_id'], 'plan': inst['plan'],
                     'success': inst['success'], 'end_step': inst['end_step']}
                    for tpl in wm.task_data[f'Block_{target["id"]}']['templates'].values()
                    for inst in tpl['instances']]
                result['condition'] = condition
                result['path'] = str(directory.relative_to(ROOT))
                if condition == 'assisted_wait':
                    # Simple comparators, fixed before observing these outcomes.
                    # The native LLM sees both prototype and instance rankings;
                    # cloning the top prototype is explicitly a weak diagnostic.
                    top_proto = next(iter(prototypes['plan_prototypes']))
                    top_instance = next(iter(instances['plan_instances'].values()))['plan']
                    proto_instance = next(
                        v['plan'] for v in instances['plan_instances'].values()
                        if ' -> '.join(a.split()[0] for a in v['plan']) == top_proto)
                    result['deployment'] = {}
                    for name, chosen in [
                        ('top_prototype_cloned', {a: proto_instance for a in plans}),
                        ('top_instance_cloned', {a: top_instance for a in plans}),
                        ('joint_episode_replay', plans)]:
                        result['deployment'][name] = execute(seed, chosen, directory/name)
            rows.append(result)
            print(json.dumps(dict(seed=seed, condition=condition,
                                  target_completed=result['target_completed'],
                                  credited=result['credited_instances'],
                                  deployment={k:v['target_completed'] for k,v in result.get('deployment',{}).items()})), flush=True)
    summary = dict(rows=rows, elapsed_s=time.monotonic()-started,
                   source_episodes=len(rows),
                   deployments=sum(len(r.get('deployment',{})) for r in rows),
                   unique_worlds=len(seeds), api_requests=0, gpu_seconds=0)
    all_results = rows + [d for r in rows for d in r.get('deployment',{}).values()]
    summary['joint_steps'] = sum(r['joint_steps'] for r in all_results)
    summary['primitive_agent_actions'] = sum(r['primitive_agent_actions'] for r in all_results)
    save(root/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/drwell_success_credit_v0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    collect((ROOT/args.output).resolve(), args.seeds)
