"""Run the native HalfCheetah pilot with a real previous-position snapshot.

In this installed NumPy version, np.copy(data.qpos.flat) shares storage with
data.qpos. The upstream PETS environment uses that expression before stepping,
which makes its forward-velocity reward identically zero. Copy on assignment
restores the intended snapshot; physics, reward formula and planner stay intact.
The uncorrected runs are retained and excluded from locomotion evidence.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import mbrl.env.pets_halfcheetah as upstream


class SnapshotHalfCheetahEnv(upstream.HalfCheetahEnv):
    @property
    def prev_qpos(self):
        return self._position_snapshot

    @prev_qpos.setter
    def prev_qpos(self, value):
        self._position_snapshot = None if value is None else value.copy()


def main():
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = ['scripts/pilot_mbrl_halfcheetah_snapshot.py', 'scripts/pilot_mbrl_rank_general.py']
    record = dict(adapter='Copy previous qpos on assignment; preserves the upstream intended displacement reward.',
                  hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
                  excluded_previous_group='runs/mbrl_rank_halfcheetah_v0',
                  reason='Uncorrected previous-qpos snapshot aliases live state; forward reward was zero.',
                  upstream_source_modified=False)
    path=output.parent/'snapshot_adapter.json'
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        path.write_text(json.dumps(record, indent=2)+'\n')
    upstream.HalfCheetahEnv = SnapshotHalfCheetahEnv
    runpy.run_path('scripts/pilot_mbrl_rank_general.py', run_name='__main__')


if __name__ == '__main__':
    main()
