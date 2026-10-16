"""Read-only component intervention on a saved, organically learned memory.

API repeats are repeated measurements of one state, never new task instances.
No mutation is made to the original run, which may have stopped early.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from wm_security.client import Client
from wm_security.local import LocalClient
from wm_security.native import NativeMemory, TypedRegistry
from wm_security.envs import trajectory
from pilot_erasure import DiagnosticWorkbench


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(Path(args.snapshot).read_text())
    (root / 'config.json').write_text(json.dumps(vars(args), indent=2))
    client_type = LocalClient if args.model.startswith('Qwen/') else Client
    client = client_type(args.model, root, max_calls=2 * args.repeats * 4 * 2)
    memories = {}
    for condition in ['full', 'without_rules', 'rules_only', 'empty']:
        memory = NativeMemory(client, DiagnosticWorkbench())
        if condition in ['full', 'rules_only']:
            memory.registry = TypedRegistry.from_dict(deepcopy(snapshot['registry']))
        if condition in ['full', 'without_rules']:
            memory.episodic.load_records(deepcopy(snapshot['episodic']))
        memories[condition] = memory

    def probe(job):
        condition, profile, repeat = job
        env = DiagnosticWorkbench()
        trajectory(env, ['power_on', 'calibrate', 'load', f'profile_{profile}'])
        state = deepcopy(env.state)
        actual = env.step('process')
        result = dict(condition=condition, profile=profile, repeat=repeat,
                      state=state, actual=actual)
        try:
            predicted, _, retrieved = memories[condition].predict(
                state, 'process', tag=f'{condition}/{profile}/{repeat}')
            result.update(prediction=predicted, retrieved=retrieved,
                          task_error=predicted['product_done'] != actual['product_done'],
                          exact_error=predicted != actual)
        except Exception as exc:
            result['error'] = str(exc)
        return result

    jobs = [(c, p, r) for c in memories for p in [0, 1] for r in range(args.repeats)]
    random.Random(941).shuffle(jobs)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(probe, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (root / 'measurements.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({k: v for k, v in result.items()
                              if k in ['condition', 'profile', 'repeat', 'task_error', 'error']}), flush=True)
    groups = []
    for c in memories:
        for p in [0, 1]:
            rows = [r for r in results if r['condition'] == c and r['profile'] == p]
            valid = [r for r in rows if 'task_error' in r]
            groups.append(dict(condition=c, profile=p, measurements=len(rows), valid=len(valid),
                               task_errors=sum(r['task_error'] for r in valid),
                               exact_errors=sum(r['exact_error'] for r in valid)))
    request_groups = {}
    for line in (root / 'calls.jsonl').read_text().splitlines():
        call = json.loads(line)
        h = hashlib.sha256(json.dumps(call['request']['messages'], sort_keys=True).encode()).hexdigest()
        record = request_groups.setdefault(h, {'tags': [], 'visible_responses': []})
        record['tags'].append(call['tag'])
        choices = call.get('response', {}).get('choices', [])
        if choices:
            record['visible_responses'].append(choices[0]['message'].get('content'))
    (root / 'result.json').write_text(json.dumps(dict(
        status='completed', model=args.model, source_snapshot=args.snapshot,
        independent_instances=1, groups=groups, identical_prompt_groups=request_groups,
        interpretation='Read-only diagnostic. Component removal is a researcher intervention, not an attack.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default='runs/erasure_api_pilot/warmup.json')
    parser.add_argument('--model', default='qwen3.8-flash')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
