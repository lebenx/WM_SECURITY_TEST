"""Repeat exact recorded first-revision prompts; these are decision probes.

No forecast is rewritten, no environment action is executed, and an action
choice is not labeled as a task success or failure.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
from wm_security.client import Client


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save = lambda name, obj: (root/name).write_text(json.dumps(obj, indent=2))
    save('config.json', vars(args))
    calls = [json.loads(line) for line in (Path(args.source)/'calls.jsonl').read_text().splitlines()]
    prompts = {}
    for condition in args.conditions:
        matches = [r for r in calls if r.get('tag') == f'actor/{condition}/0/revised/2' and not r.get('error')]
        if len(matches) != 1:
            raise ValueError(f'Expected one successful initial revision call for {condition}; got {len(matches)}')
        request = matches[0]['request']
        messages = request['messages']
        prompts[condition] = dict(messages=messages,
            sha256=hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
            source_tag=matches[0]['tag'], source_model=request['model'])
    save('prompts.json', prompts)
    client = Client(args.model, root, max_calls=2*len(prompts)*args.repeats)

    def run(condition, repeat):
        text = client.chat(prompts[condition]['messages'], tag=f'{condition}/{repeat}')
        match = re.search(r'Action:\s*([^\n]+)', text)
        return dict(status='completed', condition=condition, repeat=repeat,
                    prompt_sha256=prompts[condition]['sha256'], response=text,
                    action=match.group(1).strip() if match else None)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(run, c, r):(c, r) for r in range(args.repeats) for c in prompts}
        for future in as_completed(jobs):
            condition, repeat = jobs[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(status='interrupted', condition=condition, repeat=repeat, error=str(exc))
            results.append(result)
            save('results.json', results)
            print(json.dumps(result), flush=True)
    save('status.json', dict(status='completed' if all(r['status']=='completed' for r in results) else 'partial',
         interpretation='Exact recorded prompt repetitions, no simulator execution. Repeats are not independent task instances. Measurement skipping is not itself task failure.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='glm-5.3')
    parser.add_argument('--source', default='runs/conductivity_read_decision_hybrid_v0')
    parser.add_argument('--conditions', nargs='+', default=['clean', 'benign', 'post_attack_only'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
