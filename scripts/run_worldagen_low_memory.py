#!/usr/bin/env python3
"""Run an unchanged WorldAgen experiment with unused visual weights on CPU.

Only storage for the never-called CLIP visual and MAE decoder is moved. All actual
policy, image, forecast and loss operations retain their original devices and
dtypes. An independent execution manifest records this shim and the helper.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('native', 'cached-metadata-ttt'), required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if '--out-dir' not in forwarded:
        parser.error('forward an explicit --out-dir after --')
    output = Path(forwarded[forwarded.index('--out-dir') + 1]).resolve()
    manifest_path = output / 'memory_execution.json'
    if manifest_path.exists():
        parser.error('memory execution manifest already exists; choose a new directory')

    import torch
    import pilot_worldagen_native_task_gate as gate
    import worldagen_unused_visual as placement

    module_name = {'native': 'pilot_worldagen_native_task_gate',
                   'cached-metadata-ttt': 'pilot_worldagen_cached_metadata_ttt'}[args.target]
    target = importlib.import_module(module_name)
    started = time.monotonic()
    manifest = {
        'status': 'running', 'target': module_name, 'arguments': forwarded,
        'source_sha256': {str(Path(p).resolve()): sha(p) for p in
                          (__file__, target.__file__, gate.__file__, gate.base.__file__, placement.__file__)},
        'scope': 'Unused CLIP visual and MAE decoder storage placement only; no arithmetic, dtype, weights, mask or sampler changes.',
        'events': [], 'api_requests': 0,
    }

    def save():
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    def move_unused(model, where):
        row = placement.offload_unused_worldagen_branches(model)
        # Return the newly unused allocation to other processes. No live
        # tensor or random-generator state is changed by empty_cache.
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        manifest['events'].append({'stage': where, **row})
        save()

    original_build = gate.base.build_model

    def build():
        model = original_build()
        move_unused(model, 'after_build')
        return model

    def wrap_initializer():
        original = gate.base.WorldAgenLiberoModelWrapper

        class StoragePlacedWrapper(original):
            def __init__(self, *positional, **keyword):
                super().__init__(*positional, **keyword)
                move_unused(self.model, 'after_wrapper_init')

        gate.base.WorldAgenLiberoModelWrapper = StoragePlacedWrapper

    gate.base.build_model = build
    wrap_initializer()
    if args.target == 'cached-metadata-ttt':
        original_install = target.install_device_wrapper

        def install(gate_module, device):
            original_install(gate_module, device)
            wrap_initializer()

        target.install_device_wrapper = install
    old_argv = sys.argv
    sys.argv = [target.__file__, *forwarded]
    try:
        target.main()
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = {'type': type(exc).__name__, 'message': str(exc),
                             'traceback': traceback.format_exc()}
        raise
    finally:
        sys.argv = old_argv
        manifest['elapsed_seconds'] = time.monotonic() - started
        if torch.cuda.is_initialized():
            manifest['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            manifest['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        save()


if __name__ == '__main__':
    main()
