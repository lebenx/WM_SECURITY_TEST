"""Use the finite-head sampler with a bounded faster CPU coordinate search."""
import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import pilot_tdmpc_rank_coupling as runner
from wm_security.rank_coupling_fast import optimize_rank_phases_fast


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--task',default='walker-walk');p.add_argument('--steps',type=int,default=500)
    p.add_argument('--seeds',nargs='+',type=int,default=[17]);p.add_argument('--output',required=True)
    p.add_argument('--condition',choices=['native','independent','targeted','shared','mean_heads'],required=True)
    args=p.parse_args()
    root=Path(args.output).parent;root.mkdir(parents=True,exist_ok=True)
    paths=['scripts/pilot_tdmpc_rank_fast.py','src/wm_security/rank_coupling_fast.py']
    record={'hashes':{s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in paths},
            'operation':'Bounded one-sweep search over 128 candidates with the highest optimistic bank values. Same cyclic phase space and selected/elite-value objective; competitor thresholds accelerate trials, with native topk checks before acceptance and at boundary ties.',
            'numerics':'Floating reduction order can change tie-sized coordinate choices; no bitwise equivalence to the slow optimizer is claimed. Forecast-law preservation is independent of the chosen phases.'}
    path=root/'fast_search.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:path.write_text(json.dumps(record,indent=2)+'\n')
    runner.optimize_rank_phases=partial(optimize_rank_phases_fast,sweeps=1,max_coordinates=128)
    runner.main(args)
