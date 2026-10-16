"""Does shared experience override information retained by the acting agent?

Use natural task order: open the destination, then pick up the object, then move
it. The acting agent's history contains the open action, while the original
WorldEvolver state contains only the pickup response and inventory. Compare that
native state with the same real state accompanied by the simulator's room view.
This is an exploratory read-only diagnostic, not a completed planning task.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from scienceworld import ScienceWorldEnv
from wm_security.client import Client
from wm_security.local import LocalClient
from wm_security.upstream import text_world_model
from wm_security.outcomes import move_outcome, MOVE_CODING_VERSION


def collect_probes():
    env = ScienceWorldEnv('', envStepLimit=60)
    try:
        env.load('freeze', 0, 'easy')
        probes = []
        for obj in ['ceramic cup', 'metal pot']:
            for target in ['fridge', 'freezer']:
                env.reset()
                history = []
                for action in ['teleport to kitchen', 'open cupboard', f'open {target}', f'pick up {obj}']:
                    obs, reward, done, info = env.step(action)
                    history.append(dict(action=action, observation=obs))
                native_state = (obs.strip() + '\n' + env.inventory()).strip()
                room_state = (env.look().strip() + '\n' + env.inventory()).strip()
                action = f'move {obj} to {target}'
                actual, reward, done, info = env.step(action)
                probes.append(dict(id=f'{obj}/{target}', history=history, action=action,
                                   goal=env.get_task_description(), native_state=native_state,
                                   room_state=room_state, actual=actual))
        return probes
    finally:
        env.close()


def main(args):
    source = Path(args.source)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    probes = collect_probes()
    (root/'probes.json').write_text(json.dumps(probes, indent=2))
    (root/'config.json').write_text(json.dumps(vars(args), indent=2))
    client_type = LocalClient if args.model.startswith('Qwen/') else Client
    client = client_type(args.model, root, max_calls=200)
    conditions = {'clean': 'warmup.json'}
    for arm in args.arms:
        path = source/f'{arm}_memory.json'
        result = source/f'{arm}_result.json'
        if not path.exists() or not result.exists() or json.loads(result.read_text())['status'] != 'completed':
            raise ValueError(f'Arm {arm} must complete before evaluating its final checkpoint.')
        conditions[arm] = path.name

    def run(condition):
        wm = text_world_model(client, condition)
        wm.load_checkpoint_state(json.loads((source/conditions[condition]).read_text()))
        results = []
        for probe in probes:
            for view in ['native_state', 'room_state']:
                wm.llm.tag = f'{probe["id"]}/{view}'
                prediction = wm.predict(probe[view], probe['action'], probe['goal'])
                row = dict(condition=condition, id=probe['id'], view=view,
                           actual=probe['actual'], prediction=prediction,
                           coding_version=MOVE_CODING_VERSION,
                           coded_move=move_outcome(prediction['prediction']))
                results.append(row)
                print(json.dumps({k: row[k] for k in ['condition', 'id', 'view', 'coded_move']}), flush=True)
                (root/f'{condition}_result.json').write_text(json.dumps(results, indent=2))
        return results

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, c) for c in conditions]
        results = [f.result() for f in as_completed(futures)]
    (root/'status.json').write_text(json.dumps(dict(status='completed', model=args.model,
        conditions=list(conditions), independent_task_variations=1,
        interpretation='Native-observation and room-view forecasts, not independent task success rates.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='qwen3.8-flash')
    parser.add_argument('--source', default='runs/scienceworld_curriculum_v0')
    parser.add_argument('--output', required=True)
    parser.add_argument('--arms', nargs='*', default=[])
    parser.add_argument('--workers', type=int, default=3)
    main(parser.parse_args())
