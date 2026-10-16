"""Learn and transfer a set of feedback bindings in native DiscoverPhysics.

The restart/continuation objective and binary binding grammar are supplied by
the researcher. This is a component pilot, not a novel-method or discovery-task
claim. A fixed state-and-clock continuation baseline is evaluated explicitly.
All worlds, source fixtures, calibration and validation queries are fixed before
execution; no hidden law or world description enters binding selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/DiscoverPhysics/ScienceAgent'))

import numpy as np
from scienceagent.worlds import get_world


class NativeLab:
    def __init__(self, root):
        self.root = root
        self.executors = {}
        self.counts = {}

    def run(self, world, request, phase):
        if world not in self.executors:
            self.executors[world] = get_world(world, engine='nbody', noise_std=0)['executor']
        response = self.executors[world].run([request])[0]
        for field in ['pos2', 'velocity2']:
            assert np.asarray(response[field]).shape == (len(request['measurement_times']), 2)
            assert np.isfinite(response[field]).all()
        self.counts[phase] = self.counts.get(phase, 0) + 1
        record = {'world_for_audit_only': world, 'phase': phase,
                  'request': request, 'response': response}
        with (self.root / 'native_traces.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        return response


def case(pos, velocity, p1, p2, start, prefix, suffix):
    return {'p1': p1, 'p2': p2, 'pos2': pos, 'velocity2': velocity,
            'start_time': start, 'prefix': prefix, 'suffix': suffix}


def reference_request(item):
    return {key: value for key, value in item.items() if key not in ['prefix', 'suffix']} | {
        'measurement_times': [item['prefix'], item['prefix'] + item['suffix']],
        'duration': 5.0}


def bind(item, reference, bits):
    request = {key: value for key, value in item.items() if key not in ['prefix', 'suffix']}
    for offset, field in [(0, 'pos2'), (2, 'velocity2')]:
        request[field] = [reference[field][0][j] if bits[offset + j] else item[field][j]
                          for j in range(2)]
    request['start_time'] = item['start_time'] + (item['prefix'] if bits[4] else 0)
    request['measurement_times'] = [item['suffix']]
    request['duration'] = 5.0
    return request


def residual(reference, restart):
    expected = np.array(reference['pos2'][-1] + reference['velocity2'][-1])
    observed = np.array(restart['pos2'][-1] + restart['velocity2'][-1])
    return float(np.max(np.abs(observed - expected)))


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save = lambda name, data: (root / name).write_text(json.dumps(data, indent=2) + '\n')
    started = time.monotonic()
    masks = list(product([0, 1], repeat=5))
    sources = [case([3., 1.], [-.2, .7], 2., 1., 0., .5, .4),
               case([4., -2.], [.3, -.4], 1.3, .8, .2, .35, .45)]
    calibration = case([3.5, 1.5], [.2, -.3], 1.7, 1.1, .15, .5, .4)
    validation = [case([4.5, -1.5], [-.1, .3], 1.2, 1.4, .3, .4, .5),
                  case([-3.5, 2.], [.2, .1], 2., 1.2, .6, .55, .35),
                  case([2.5, 3.5], [-.3, .2], .9, .7, .9, .3, .6)]
    worlds = ['yukawa', 'coulomb_easy', 'extra_dimensions', 'oscillator']
    save('plan.json', {
        'source_world': 'gravity', 'source_cases': sources, 'target_worlds': worlds,
        'calibration_case': calibration, 'validation_cases': validation,
        'grammar': 'Four coordinates independently use initial or observed midpoint values; clock independently retains start_time or advances by the prefix duration.',
        'objective': 'Match reference continuation after a restarted native experiment.',
        'source_acceptance_max_absolute_error': args.tolerance,
        'researcher_supplied': ['Restart/continuation template and comparison objective',
                                'Binary field-binding grammar including optional start_time',
                                'Source/target worlds and cases',
                                'Fixed full-state-and-clock propagation baseline'],
        'interface_scope': 'Optional start_time is documented and supported by the native oscillator executor; exposed equally to all arms here. This is not the unchanged standard LLM prompt protocol.',
        'native_pin': subprocess.check_output(['git', '-C', str(ROOT / 'vendor/DiscoverPhysics'),
                                               'rev-parse', 'HEAD'], text=True).strip(),
        'api_requests': 0, 'gpu_seconds': 0,
    })
    lab = NativeLab(root)
    scores = {bits: [] for bits in masks}
    for item in sources:
        reference = lab.run('gravity', reference_request(item), 'source_reference')
        for bits in masks:
            reply = lab.run('gravity', bind(item, reference, bits), 'source_binding_search')
            scores[bits].append(residual(reference, reply))
    viable = [bits for bits in masks if max(scores[bits]) <= args.tolerance]
    assert viable, 'No source-consistent feedback program found.'
    # Only source observations determine these artifacts. Preserve ties rather
    # than hiding them with floating-point argmin noise.
    representative = min(viable, key=lambda bits: (sum(bits), bits))
    source_artifact = {'viable_bindings': viable, 'frozen_representative': representative,
                       'binding_scores': [{'bits': bits, 'errors': scores[bits]} for bits in masks],
                       'source_native_calls': sum(lab.counts.values()),
                       'frozen_at_utc': datetime.now(timezone.utc).isoformat()}
    save('source_artifact.json', source_artifact)
    print(json.dumps({'phase': 'source_completed', 'viable': viable,
                      'representative': representative, 'calls': sum(lab.counts.values())}), flush=True)
    results = []
    for world in worlds:
        reference = lab.run(world, reference_request(calibration), 'target_calibration_reference')
        calibration_scores = []
        for bits in viable:
            reply = lab.run(world, bind(calibration, reference, bits), 'target_binding_calibration')
            calibration_scores.append((residual(reference, reply), bits))
        consistent = [bits for error, bits in calibration_scores if error <= args.tolerance]
        selected = min(consistent, key=lambda bits: (sum(bits), bits)) if consistent else None
        save(f'{world}_selection.json', {'scores': calibration_scores, 'selected': selected,
                                         'validation_not_yet_executed': True})
        arms = {'source_frozen': representative, 'source_set_calibrated': selected,
                'fixed_full_state_and_clock': (1, 1, 1, 1, 1)}
        validation_rows = []
        for i, item in enumerate(validation):
            reference = lab.run(world, reference_request(item), 'validation_reference')
            for arm, bits in arms.items():
                if bits is None:
                    validation_rows.append({'case': i, 'arm': arm, 'abstained': True,
                                             'passed': False, 'error': None})
                    continue
                reply = lab.run(world, bind(item, reference, bits), f'validation_{arm}')
                error = residual(reference, reply)
                validation_rows.append({'case': i, 'arm': arm, 'bits': bits,
                                         'error': error, 'passed': error <= args.tolerance})
        row = {'world': world, 'selected': selected, 'calibration_scores': calibration_scores,
               'validation': validation_rows,
               'pass_counts': {arm: sum(r['passed'] for r in validation_rows if r['arm'] == arm)
                               for arm in arms}}
        results.append(row)
        save('target_results.json', results)
        print(json.dumps({'phase': 'target_completed', 'world': world,
                          'selected': selected, 'pass_counts': row['pass_counts'],
                          'calls': sum(lab.counts.values())}), flush=True)
    summary = {
        'status': 'completed', 'source_viable_bindings': viable,
        'source_representative': representative,
        'target_summary': [{'world': r['world'], 'selected': r['selected'],
                            'pass_counts': r['pass_counts']} for r in results],
        'native_experiments': sum(lab.counts.values()), 'native_calls_by_phase': lab.counts,
        'api_requests': 0, 'gpu_seconds': 0, 'training_type': 'Finite source-program consistency induction and target version-space filtering; no neural training.',
        'elapsed_s': time.monotonic() - started,
        'accepted_novel_method': False,
        'interpretation': 'Tests transfer and calibration of source-consistent feedback bindings. Compare with the fixed full-state-and-clock baseline before drawing any method-advantage conclusion.',
        'limitations': 'A researcher-supplied classical flow-continuation template and finite grammar, deterministic noise-free fixtures, one source law, four selected target laws, three parameter cases each. No recovered law, official discovery score, reward task or independent benchmark success rate.',
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    save('summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--tolerance', type=float, default=1e-8)
    main(parser.parse_args())
