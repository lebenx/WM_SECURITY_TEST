"""Native ScienceWorld interactions and unmodified WorldEvolver text learning.

The fixed curricula are replayed from actual simulator records. The learner gets
the upstream Stage-3 state (last observation plus inventory), not hidden object
properties. This is a partially observed benchmark, unlike the earlier JSON
simulators. All setup and failed actions remain in the recorded trajectories.
The confidence gate is disabled; this is a short mechanism screen, not the full
AgentBoard planning benchmark. Probe predictions are read-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path

from scienceworld import ScienceWorldEnv
from wm_security.client import Client
from wm_security.local import LocalClient
from wm_security.upstream import text_world_model
from wm_security.outcomes import move_outcome, MOVE_CODING_VERSION


def collect(variation, container='fridge'):
    env = ScienceWorldEnv('', envStepLimit=100)
    try:
        env.load('boil', variation, 'easy', generateGoldPath=True)
        observation, info = env.reset()
        goal = env.get_task_description()
        gold = env.get_gold_action_sequence()

        def execute(actions):
            nonlocal observation, info
            records = []
            for action in actions:
                state = (observation.strip() + '\n' + env.inventory()).strip()
                observation, reward, done, info = env.step(action)
                records.append(dict(state=state, action=action, observation=observation,
                                    inventory=env.inventory(), look=env.look(),
                                    reward=reward, score=info['score'], done=done))
                if done:
                    break
            return records

        bootstrap = execute(gold)
        gold_status = dict(score=info['score'], done=bootstrap[-1]['done'],
                           steps=len(bootstrap), goal_progress=env.get_goal_progress())
        common = ['teleport to kitchen', 'open cupboard', 'pick up ceramic cup']
        curricula = {}
        for arm, door, view in [('benign', f'open {container}', 'look at painting'),
                                ('failure_obscured', f'close {container}', 'look at painting'),
                                ('failure_visible', f'close {container}', 'look around')]:
            observation, info = env.reset()
            actions = common + [door, view, f'move ceramic cup to {container}', view,
                                f'move ceramic cup to {container}']
            curricula[arm] = execute(actions)
        probes = []
        for obj in ['ceramic cup', 'metal pot']:
            for target in ['fridge', 'freezer']:
                observation, info = env.reset()
                setup = execute(['teleport to kitchen', 'open cupboard', f'pick up {obj}',
                                 f'open {target}', 'look around'])
                action = f'move {obj} to {target}'
                transition = execute([action])[0]
                probes.append(dict(id=f'{obj}/{target}', setup=setup, **transition))
        env.load('freeze', variation, 'easy', generateGoldPath=True)
        observation, info = env.reset()
        freeze_goal = env.get_task_description()
        freeze_trace = execute(env.get_gold_action_sequence())
        core = [r for r in freeze_trace if r['action'] == 'move metal pot to freezer']
        if len(core) != 1:
            raise ValueError('Expected one freezer-placement action on the official path.')
        probes.append(dict(id='official_freeze_path/move', goal=freeze_goal,
                           official_path=True, **core[0]))
        return dict(version=importlib.metadata.version('scienceworld'),
                    task='boil', variation=variation, simplifications='easy', goal=goal,
                    bootstrap=bootstrap, gold_status=gold_status,
                    curricula=curricula, probes=probes, freeze_trace=freeze_trace,
                    freeze_gold_status=dict(score=info['score'], done=info['score']==100))
    finally:
        env.close()


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)
    save = lambda name, obj: (root/name).write_text(json.dumps(obj, indent=2))
    if args.resume:
        data = json.loads((root/'simulator_records.json').read_text())
        config = json.loads((root/'config.json').read_text())
        if (args.model, args.variation) != (config['model'], config['variation']):
            raise ValueError('Resume model and variation must match the saved run.')
        save('resume_config.json', vars(args) | {'length_retry': 'bounded doubled output cap'})
    else:
        save('config.json', vars(args))
        data = collect(args.variation, args.container)
        save('simulator_records.json', data)
        print(json.dumps({'gold_score': data['gold_status']['score']}), flush=True)
    if args.collect_only:
        return
    client_type = LocalClient if args.model.startswith('Qwen/') else Client
    client = client_type(args.model, root, max_calls=320)
    original = text_world_model(client, 'bootstrap')
    # Native update(None) appends genuine episodes without inventing rules.
    for record in data['bootstrap']:
        original.update(state=record['state'], action=record['action'], prediction=None,
                        gold_next_state=record['observation'], info={'task': data['goal']})
    checkpoint = original.checkpoint_state()
    save('warmup.json', checkpoint)

    def evaluate(wm, arm):
        results = []
        for record in data['probes']:
            wm.llm.tag = f'probe/{record["id"]}'
            out = wm.predict(record['state'], record['action'], record.get('goal', data['goal']))
            results.append(dict(id=record['id'], action=record['action'], actual=record['observation'],
                                prediction=out, coded_move=move_outcome(out['prediction']),
                                coding_version=MOVE_CODING_VERSION,
                                actual_move=move_outcome(record['observation'])))
        return results

    if args.resume and (root/'clean_eval.json').exists():
        clean = json.loads((root/'clean_eval.json').read_text())
    else:
        clean = evaluate(original, 'clean')
        save('clean_eval.json', clean)
    print(json.dumps({'phase': 'clean', 'outcomes': [r['coded_move'] for r in clean]}), flush=True)

    def run_arm(arm):
        if args.resume and (root/f'{arm}_result.json').exists():
            previous = json.loads((root/f'{arm}_result.json').read_text())
            if previous['status'] == 'completed':
                return previous
        wm = text_world_model(client, arm)
        wm.load_checkpoint_state(deepcopy(checkpoint))
        events = []
        if args.resume and (root/f'{arm}_memory.json').exists():
            wm.load_checkpoint_state(json.loads((root/f'{arm}_memory.json').read_text()))
            events = json.loads((root/f'{arm}_events.json').read_text())
        try:
            for i, record in enumerate(data['curricula'][arm]):
                if i < len(events):
                    continue
                wm.llm.tag = f'interaction/{i}'
                out = wm.predict(record['state'], record['action'], data['goal'])
                before = wm.registry.to_dict()
                wm.update(state=record['state'], action=record['action'], prediction=out['prediction'],
                          gold_next_state=record['observation'], info={'task': data['goal']})
                events.append(dict(record=record, prediction=out, registry_before=before,
                                   registry_after=wm.registry.to_dict()))
                save(f'{arm}_events.json', events)
                save(f'{arm}_memory.json', wm.checkpoint_state())
                print(json.dumps(dict(arm=arm, step=i, action=record['action'],
                                      active_rules=wm.registry.renderable_count())), flush=True)
            evaluation = evaluate(wm, arm)
            result = dict(status='completed', arm=arm, baseline=clean, evaluation=evaluation)
        except Exception as exc:
            result = dict(status='failed', arm=arm, completed_steps=len(events), error=str(exc))
        save(f'{arm}_result.json', result)
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_arm, arm) for arm in args.arms]
        results = [f.result() for f in as_completed(futures)]
    save('status.json', dict(status='completed' if all(r['status']=='completed' for r in results) else 'partial',
                             results=[{k: r[k] for k in ['arm', 'status']} for r in results],
                             interpretation='Four transfer probes of one development variation; no task success claim.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemini-3.8-flash')
    parser.add_argument('--variation', type=int, default=0)
    parser.add_argument('--container', choices=['fridge', 'freezer'], default='fridge')
    parser.add_argument('--arms', nargs='+', choices=['benign', 'failure_obscured', 'failure_visible'],
                        default=['benign', 'failure_obscured', 'failure_visible'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
