"""Frozen read tests of captured retrieval, with an explicit recency intervention.

The post checkpoint includes five real corrective successes. The intervention
changes only tie-breaking among equally relevant actions, without removing or
editing records. It is a researcher intervention, not an attacker capability.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path

from wm_security.client import Client
from wm_security.local import LocalClient
from wm_security.upstream import text_world_model
from wm_security.outcomes import move_outcome, MOVE_CODING_VERSION


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (root/name).write_text(json.dumps(value, indent=2))
    save('config.json', vars(args))
    source = Path(args.source)
    clean = json.loads((source/'warmup.json').read_text())
    benign = json.loads((source/'benign_memory.json').read_text())
    post = json.loads((source/'targeted_failures_memory.json').read_text())
    data = json.loads(Path(args.data).read_text())
    target = 'move metal pot to freezer'
    rows = data['freeze_trace']
    index = next(i for i, row in enumerate(rows) if row['action'] == target)
    native = rows[index]['state']
    inventory = native[native.index('In your inventory,'):]
    probes = dict(native=native, room=rows[index-1]['look'].strip()+'\n'+inventory)
    goal = next(r['goal'] for r in data['probes'] if r.get('official_path'))
    client_type = LocalClient if args.model.startswith('Qwen/') else Client
    client = client_type(args.model, root, max_calls=80)
    checkpoints = dict(clean=clean, benign=benign, post_with_corrections=post, post_recent=post)

    def query(condition, view, repeat):
        wm = text_world_model(client, f'{condition}/{view}/{repeat}')
        wm.load_checkpoint_state(deepcopy(checkpoints[condition]))
        if condition == 'post_recent':
            original_rank = wm.library._rank_by_action

            def newest_ties(action, k):
                all_matches = original_rank(action, len(wm.library.records))
                return sorted(all_matches, key=lambda item: (item[1], item[0]), reverse=True)[:k]

            wm.library._rank_by_action = newest_ties
        prediction = wm.predict(probes[view], target, goal)
        return dict(condition=condition, view=view, repeat=repeat, prediction=prediction,
                    actual=rows[index]['observation'], coded_move=move_outcome(prediction['prediction']),
                    coding_version=MOVE_CODING_VERSION)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = [executor.submit(query, condition, view, repeat)
                 for repeat in range(args.repeats) for condition in args.conditions for view in probes]
        for future in as_completed(tasks):
            result = future.result()
            results.append(result)
            save('results.json', results)
            print(json.dumps({k:result[k] for k in ['condition','view','repeat','coded_move']}), flush=True)
    save('status.json', dict(status='completed',
        interpretation='Frozen predictions on two views of one state; repetitions are not independent instances. Post memory includes five corrective successes. Recency is a researcher intervention.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='glm-5.3')
    parser.add_argument('--source', default='runs/scienceworld_capture_local')
    parser.add_argument('--data', default='runs/scienceworld_freezer_local/simulator_records.json')
    parser.add_argument('--conditions', nargs='+', default=['clean','benign','post_with_corrections','post_recent'],
                        choices=['clean','benign','post_with_corrections','post_recent'])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
