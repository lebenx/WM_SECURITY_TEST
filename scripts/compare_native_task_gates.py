#!/usr/bin/env python3
"""Paired native task comparison, requiring complete physical/protocol pairing."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze_native_task_gate import analyze, episode_key


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_rows(data):
    if data.get('status') != 'completed':
        raise ValueError('Require a completed run; partial/OOM episodes are not failures')
    rows = data['results']
    if len(rows) != 30 or {(r['task_id'], r['init_id']) for r in rows} != {
            (task, initial) for task in range(10) for initial in range(3)}:
        raise ValueError('Require the complete preregistered 10-task x 3-initial-state set')
    for row in rows:
        if row['condition'] != 'native' or not isinstance(row['success'], bool):
            raise ValueError('Require native condition and actual Boolean task success')
        if row.get('success_at_start') is not False:
            raise ValueError('Require verified unsuccessful initial task state')
        trace = [json.loads(line) for line in Path(row['trace']).read_text().splitlines()]
        if len(trace) != row['environment_steps'] or not trace:
            raise ValueError('Recorded environment-step count differs from actual trace')
        if trace[-1]['task_success'] != row['success']:
            raise ValueError('Summary task outcome differs from final environment trace')
        if any(not isinstance(step['task_success'], bool) for step in trace):
            raise ValueError('Trace must contain Boolean environment success predicates')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('variant', type=Path)
    parser.add_argument('--changed-field', default='camera_size')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref, var = [json.loads(path.read_text()) for path in (args.reference, args.variant)]
    reference_rows, variant_rows = checked_rows(ref), checked_rows(var)
    for key in ('script_sha256', 'checkpoint_provenance', 'gripper_codec',
                'success_source', 'forecast_used_for_action_selection'):
        if ref.get(key) != var.get(key):
            raise ValueError(f'Uncontrolled implementation or checkpoint difference: {key}')
    differences = {key: [ref['protocol'].get(key), var['protocol'].get(key)]
                   for key in set(ref['protocol']) | set(var['protocol'])
                   if key != 'out_dir' and ref['protocol'].get(key) != var['protocol'].get(key)}
    if set(differences) != {args.changed_field}:
        raise ValueError(f'Expected only {args.changed_field} to differ, found {differences}')
    if {episode_key(row) for row in reference_rows} != {episode_key(row) for row in variant_rows}:
        raise ValueError('Episode pairing differs')
    merged = {'status': 'completed', 'results': [
        {**row, 'condition': label} for label, rows in
        (('reference', reference_rows), ('variant', variant_rows)) for row in rows]}
    result = analyze(merged, 'reference')
    result['controlled_protocol_difference'] = differences
    result['source_artifacts'] = {}
    for label, path in (('reference', args.reference), ('variant', args.variant)):
        record = {'summary': str(path), 'sha256': digest(path)}
        execution = path.parent / 'memory_execution.json'
        if execution.exists():
            meta = json.loads(execution.read_text())
            if meta['status'] != 'completed':
                raise ValueError('Storage placement execution did not complete')
            record['memory_execution'] = str(execution)
            record['memory_execution_sha256'] = digest(execution)
        result['source_artifacts'][label] = record
    result['readiness_gate'] = {
        label: {'passed': row['successes'] >= 21 and row['tasks_with_success'] >= 7,
                'minimum_successes': 21, 'minimum_tasks_with_success': 7}
        for label, row in result['conditions'].items()}
    result['scope'] += ' A configuration comparison, not attack/defense evidence. Storage-only execution changes require their separate equivalence evidence.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
