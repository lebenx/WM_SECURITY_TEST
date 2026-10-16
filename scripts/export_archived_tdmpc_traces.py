#!/usr/bin/env python3
"""Export only the four archived R005 evaluation tensors to public NumPy arrays.

This is an offline format conversion, not an experiment. The fixed whitelist
excludes model checkpoints. Existing PT files and existing exports are never
overwritten. No model/environment execution, API calls, or hashing is used.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'runs/tdmpc_proposal_scoring_factorial_v1'
ARMS = ('RR', 'CC', 'CR', 'RC')
FIELDS = ('obs', 'action', 'reward', 'task', 'seed')
SOURCES = tuple(ARCHIVE / arm / 'evaluation.pt' for arm in ARMS)
MANIFEST = ARCHIVE / 'evaluation_arrays_manifest.json'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def export():
    import numpy as np
    import torch

    outputs = tuple(ARCHIVE / arm / 'evaluation_arrays.npz' for arm in ARMS)
    require(not MANIFEST.exists() and not any(path.exists() for path in outputs),
            'An export already exists; refusing to overwrite archived artifacts')
    summary = json.loads((ARCHIVE / 'summary.json').read_text())
    require(summary['status'] == 'completed_native_consumer_factorial',
            'Expected the completed historical R005 summary')
    prepared = []
    for arm, source, output in zip(ARMS, SOURCES, outputs):
        # This is the only torch.load call, and its argument is restricted to
        # the four literal historical evaluation paths. Symlinks are rejected.
        require(source in SOURCES and source.name == 'evaluation.pt' and
                source.resolve() == source.absolute(), 'Source is outside the fixed whitelist')
        stat = source.stat()
        data = torch.load(source, map_location='cpu', weights_only=True)
        require(isinstance(data, dict) and set(data) == set(FIELDS),
                'Evaluation payload must contain only obs/action/reward/task/seed')
        arrays = {}
        for field in ('obs', 'action', 'reward'):
            value = data[field]
            require(torch.is_tensor(value) and value.device.type == 'cpu' and
                    not value.requires_grad, 'Expected an ordinary CPU evaluation tensor')
            arrays[field] = value.numpy().copy()
        for field in ('task', 'seed'):
            require(type(data[field]) is int, 'Expected integer task/seed metadata')
            arrays[field] = np.asarray(data[field], dtype=np.int64)
        require(all(not value.dtype.hasobject for value in arrays.values()), 'Object dtype forbidden')
        require(all(np.isfinite(value).all() for value in arrays.values()), 'Nonfinite archived values')
        require(arrays['obs'].shape == (501, 24) and arrays['action'].shape == (500, 6) and
                arrays['reward'].shape == (500,), 'Unexpected historical trajectory shape')
        require(all(arrays[field].dtype == np.float32 for field in ('obs', 'action', 'reward')),
                'Unexpected historical trajectory dtype')
        require(int(arrays['task']) == 2 and int(arrays['seed']) == 31, 'Historical task/seed differs')
        old = summary['arms'][arm]['evaluation']
        actual_return = sum(float(value) for value in arrays['reward'])
        require(old['steps'] == 500 and old['full_episode'] and actual_return == old['return'],
                'Archived PT reward sum does not exactly match historical summary: ' + arm)
        prepared.append((arm, source, output, stat, arrays, old))

    manifest = {
        'status': 'completed_offline_array_export',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Format conversion of existing R005 physical trajectories; no new experimental data.',
        'source_summary': str((ARCHIVE / 'summary.json').relative_to(ROOT)),
        'source_policy': 'Original evaluation.pt files are retained locally and are not part of this public export.',
        'export_policy': 'The four evaluation_arrays.npz files contain public obs/action/reward/task/seed arrays only.',
        'equivalence_rule': 'Identical per-field shape, dtype and raw bytes; exact 500-reward sum against historical summary.',
        'activity': {'evaluation_pt_reads': 4, 'checkpoint_reads': 0, 'model_forwards': 0,
                     'environment_steps': 0, 'api_calls': 0, 'training_updates': 0, 'hashes': 0},
        'arms': {},
    }
    for arm, source, output, stat, arrays, old in prepared:
        with output.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        with np.load(output, allow_pickle=False) as restored:
            require(set(restored.files) == set(FIELDS), 'Export fields differ')
            checks = {}
            for field in FIELDS:
                value = restored[field]
                checks[field] = (not value.dtype.hasobject and value.shape == arrays[field].shape and
                                 value.dtype == arrays[field].dtype and value.tobytes() == arrays[field].tobytes())
            require(all(checks.values()), 'NPZ round-trip array equivalence failed: ' + arm)
            restored_return = sum(float(value) for value in restored['reward'])
            require(restored_return == old['return'], 'NPZ reward sum differs from historical summary')
            require(len(restored['reward']) == 500, 'NPZ must contain all 500 controls')
        after = source.stat()
        require((after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns),
                'Original local PT file changed during export')
        manifest['arms'][arm] = {
            'local_source_pt': str(source.relative_to(ROOT)),
            'public_array_export': str(output.relative_to(ROOT)),
            'original_pt_retained': True, 'original_size_and_mtime_unchanged': True,
            'arrays': {field: {'shape': list(value.shape), 'dtype': str(value.dtype),
                               'round_trip_bytes_exact': checks[field]} for field, value in arrays.items()},
            'steps': 500, 'task': 2, 'seed': 31, 'return_from_npz': restored_return,
            'historical_summary_return': old['return'], 'return_exact': True,
        }
    with MANIFEST.open('x') as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': manifest['status'], 'manifest': str(MANIFEST.relative_to(ROOT)),
                      'returns': {arm: record['return_from_npz'] for arm, record in manifest['arms'].items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true', required=True,
                        help='Convert the fixed four historical PT files; refuse existing outputs')
    parser.parse_args()
    export()
