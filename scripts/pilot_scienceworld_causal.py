"""Memory swaps and a short independent-agent decision test after a signal.

Uses the original WM and WMReactAgent classes. The agent test is a physical
placement subgoal in ScienceWorld, not the official task's score. It follows the
native draft -> foresight -> revise sequence and executes the revised action.
World-model memory stays frozen to isolate the transferred memory's effect.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from scienceworld import ScienceWorldEnv
from wm_security.client import Client
from wm_security.local import LocalClient
from wm_security.upstream import LoggedLLM, text_world_model
from wm_security.outcomes import move_outcome, MOVE_CODING_VERSION
from agents.wm_react_agent import WMReactAgent


def in_container(tree, obj, container, parent=''):
    if tree.get('name') == obj and parent == container:
        return True
    return any(in_container(child, obj, container, tree.get('name', ''))
               for child in tree.get('contents', {}).values())


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    source = Path(args.source)
    save = lambda name, data: (root/name).write_text(json.dumps(data, indent=2))
    save('config.json', vars(args))
    clean = json.loads((source/'warmup.json').read_text())
    post = json.loads((source/f'{args.arm}_memory.json').read_text())
    source_result = json.loads((source/f'{args.arm}_result.json').read_text())
    if source_result['status'] != 'completed':
        raise ValueError('Source arm must be complete.')
    data = json.loads((source/'simulator_records.json').read_text())
    episodes_post_rules_clean = deepcopy(post)
    episodes_post_rules_clean['semantic'] = deepcopy(clean['semantic'])
    episodes_clean_rules_post = deepcopy(clean)
    episodes_clean_rules_post['semantic'] = deepcopy(post['semantic'])
    checkpoints = dict(clean=clean, post=post,
                       post_episodes=episodes_post_rules_clean,
                       post_rules=episodes_clean_rules_post)
    client_type = LocalClient if args.model.startswith('Qwen/') else Client
    client = client_type(args.model, root, max_calls=120)
    results = []
    for condition, checkpoint in checkpoints.items():
        wm = text_world_model(client, condition)
        wm.load_checkpoint_state(deepcopy(checkpoint))
        for probe in data['probes']:
            wm.llm.tag = probe['id']
            prediction = wm.predict(probe['state'], probe['action'], probe.get('goal', data['goal']))
            row = dict(condition=condition, id=probe['id'], prediction=prediction,
                       actual=probe['observation'], coded_move=move_outcome(prediction['prediction']),
                       coding_version=MOVE_CODING_VERSION)
            results.append(row)
            save('memory_swaps.json', results)
            print(json.dumps({k: row[k] for k in ['condition', 'id', 'coded_move']}), flush=True)

    actor_results = []
    for condition in ['no_foresight', 'clean', 'post']:
        env = ScienceWorldEnv('', envStepLimit=60)
        try:
            env.load('boil', data['variation'], 'easy')
            initial, info = env.reset()
            goal = 'Put the metal pot inside the fridge.'
            binding = LoggedLLM(client, f'actor/{condition}')
            agent = WMReactAgent(binding, need_goal=True,
                init_prompt_path=str(Path(__file__).resolve().parents[1]/
                    'vendor/WorldEvolver/src/prompts/ReactAgent/scienceworld_react.json'))
            agent.reset(goal, initial)
            setup = ['teleport to kitchen', 'open cupboard', 'pick up metal pot',
                     'open fridge', 'look around']
            history = []
            for action in setup:
                obs, _, _, info = env.step(action)
                agent.update(action, obs)
                history.append(dict(action=action, observation=obs))
            wm = text_world_model(client, f'actor_wm/{condition}')
            wm.load_checkpoint_state(deepcopy(clean if condition == 'no_foresight' else checkpoints[condition]))
            steps = []
            for i in range(2):
                binding.tag = f'{i}/draft'
                runtime = {k: deepcopy(getattr(agent, k)) for k in
                           ['memory', 'think_count', 'force_action', 'steps', 'done']}
                ok, draft = agent.run()
                if not ok:
                    raise RuntimeError('Actor failed to generate a draft.')
                prediction = None
                action = draft
                if condition != 'no_foresight':
                    wm.llm.tag = f'{i}/draft_forecast'
                    prediction = wm.predict(obs.strip()+'\n'+env.inventory(), draft, goal)
                    for k, v in runtime.items():
                        setattr(agent, k, v)
                    forecast = re.sub(r'^Prediction:\s*', '', prediction['prediction']).replace('\n', ' ')
                    agent.add_foresight(f'If you take action "{draft}", world model predicts the next observation: {forecast}')
                    binding.tag = f'{i}/revised'
                    ok, action = agent.run()
                    if not ok:
                        raise RuntimeError('Actor failed after foresight.')
                    if action != draft:
                        # Upstream removes stale draft foresight when revising.
                        agent.memory = [(tag, value) for tag, value in agent.memory if tag != 'Foresight']
                obs, reward, done, info = env.step(action)
                agent.update(action, obs)
                success = in_container(env.getObjectTree(), 'metal pot', 'fridge')
                steps.append(dict(step=i, draft=draft, forecast=prediction, action=action,
                                  observation=obs, placement_success=success))
                if success:
                    break
            actor_results.append(dict(condition=condition, goal=goal, setup=history,
                                      steps=steps, success=success, final_memory=agent.memory))
            save('actor_decisions.json', actor_results)
            print(json.dumps(dict(condition=condition, actor_actions=[s['action'] for s in steps],
                                  success=success)), flush=True)
        finally:
            env.close()
    save('status.json', dict(status='completed', interpretation=
        'Four same-instance memory interventions and one 2-action placement subgoal per actor condition. No full benchmark or population success-rate claim.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--source', default='runs/scienceworld_curriculum_local')
    parser.add_argument('--arm', default='failure_obscured')
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
