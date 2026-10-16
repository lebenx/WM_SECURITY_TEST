#!/usr/bin/env python3
"""Summarize native task predicates, preserving failures and paired episodes.

Intervals resample entire tasks; they are descriptive for the observed tasks,
not evidence that deterministic first-N initial states are a random sample.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics


def cluster_interval(values_by_task, draws=10000):
    values = list(values_by_task.values())
    if len(values) < 2:
        return None
    rng = random.Random(20261014)
    samples = sorted(
        statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws)
    )
    return [samples[int(draws * .025)], samples[int(draws * .975)]]


def task_key(row):
    return row['suite'], int(row['task_id'])


def episode_key(row):
    return (*task_key(row), int(row['init_id']), int(row['seed']))


def analyze(data, reference):
    groups = defaultdict(dict)
    for row in data['results']:
        if not isinstance(row['success'], bool):
            raise ValueError('success must be an actual Boolean task predicate')
        condition = row['condition']
        key = episode_key(row)
        if key in groups[condition]:
            raise ValueError(f'duplicate episode: {condition} {key}')
        if row.get('success_at_start', False):
            raise ValueError('task already successful at initial state')
        groups[condition][key] = row
    conditions, paired = {}, {}
    for condition, episodes in groups.items():
        by_task = defaultdict(list)
        for row in episodes.values():
            by_task[task_key(row)].append(row)
        task_means = {
            k: statistics.mean(int(r['success']) for r in rows)
            for k, rows in by_task.items()
        }
        conditions[condition] = {
            'episodes': len(episodes),
            'successes': sum(int(r['success']) for r in episodes.values()),
            'success_rate': statistics.mean(int(r['success']) for r in episodes.values()),
            'task_balanced_success_rate': statistics.mean(task_means.values()),
            'task_cluster_bootstrap_ci95_descriptive': cluster_interval(task_means),
            'environment_steps_total': sum(int(r['environment_steps']) for r in episodes.values()),
            'successful_episode_steps_mean': statistics.mean(
                r['environment_steps'] for r in episodes.values() if r['success']
            ) if any(r['success'] for r in episodes.values()) else None,
            'tasks_with_success': sum(v > 0 for v in task_means.values()),
            'tasks': [
                {'suite': k[0], 'task_id': k[1], 'episodes': len(rows),
                 'successes': sum(int(r['success']) for r in rows),
                 'success_rate': task_means[k]}
                for k, rows in sorted(by_task.items())
            ],
        }
        if condition == reference or reference not in groups:
            continue
        refs = groups[reference]
        common = sorted(set(episodes).intersection(refs))
        task_deltas = defaultdict(list)
        lost = gained = 0
        for key in common:
            row, ref = episodes[key], refs[key]
            for field in ('initial_state_sha256', 'bddl_sha256'):
                if field in row and field in ref and row[field] != ref[field]:
                    raise ValueError(f'paired physical identity differs: {key} {field}')
            delta = int(row['success']) - int(ref['success'])
            task_deltas[task_key(row)].append(delta)
            lost += delta < 0
            gained += delta > 0
        means = {k: statistics.mean(v) for k, v in task_deltas.items()}
        paired[condition] = {
            'reference': reference, 'paired_episodes': len(common),
            'unmatched_condition': len(set(episodes) - set(refs)),
            'unmatched_reference': len(set(refs) - set(episodes)),
            'success_to_failure': lost, 'failure_to_success': gained,
            'episode_success_rate_delta': (gained-lost) / len(common) if common else None,
            'task_balanced_success_rate_delta': statistics.mean(means.values()) if means else None,
            'task_cluster_bootstrap_delta_ci95_descriptive': cluster_interval(means),
        }
    return {
        'status': 'completed_task_predicate_analysis',
        'source_run_status': data.get('status'),
        'conditions': conditions, 'paired': paired,
        'scope': 'Observed native task success; first-N initial states and selected tasks limit generalization. Cluster intervals do not establish population coverage.',
        'api_requests': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--reference', default='native')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    result = analyze(data, args.reference)
    result['source'] = str(args.summary)
    result['source_sha256'] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
