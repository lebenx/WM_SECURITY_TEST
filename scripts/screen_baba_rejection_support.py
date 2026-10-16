"""Check whether published regression splits have support in native warmup data.

No synthesized attack, policy rollout, or new environment interaction. Candidate
pairs are selected using the earlier solution-trace criterion, independently of
their support or downstream utility in this buffer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/BabaInWonderland"))
from src.data import Transition
from src.program_model.evaluator import ProgramEvaluator


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=Path, default=ROOT/'runs/baba_frontier_effect_checkpoint_v0/warmup.jsonl.gz')
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    started=time.monotonic()
    variant='wonderland'
    with np.load(ROOT/'runs/baba_program_history_full_v0/wonderland_correctness.npz',allow_pickle=False) as archive:
        history,versions=archive['correct'],archive['versions']
    eligible=[i for i in range(1,len(versions)) if history[i-1].mean()>=.9
              and (~history[i-1]&history[i]).any() and (history[i-1]&~history[i]).any()]
    selected=sorted({i for j in eligible for i in [j-1,j]})
    with gzip.open(args.warmup,'rt') as handle:
        transitions=[Transition(**json.loads(line)) for line in handle]
    save(args.output/'setup.json',{'variant':variant,'warmup':str(args.warmup),
        'warmup_sha256':hashlib.sha256(args.warmup.read_bytes()).hexdigest(),
        'selection':'All adjacent published pairs with before accuracy >=90%, positive gains and positive losses on the previous training solution archive.',
        'pairs':[(str(versions[i-1]),str(versions[i])) for i in eligible],
        'maturity_threshold':32,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    matrices={}
    evaluation_rows=[]
    evaluator=ProgramEvaluator(program_eval_workers=8)
    try:
        for i in selected:
            path=ROOT/'vendor/BabaInWonderland_demo/programs'/variant/(str(versions[i])+'.py')
            evaluation=evaluator.evaluate_source(path.read_text(),transitions)
            assert not evaluation.compile_errors and not evaluation.runtime_error_count
            vector=np.asarray([r.is_correct for r in evaluation.records],dtype=bool)
            matrices[i]=vector
            row={'version':str(versions[i]),'correct':int(vector.sum()),'total':len(vector),
                 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'runtime_errors':0}
            evaluation_rows.append(row)
            save(args.output/'evaluations.json',evaluation_rows)
            print(json.dumps(row),flush=True)
    finally:
        evaluator.close()
    np.savez_compressed(args.output/'correctness.npz',versions=versions[selected],
                        correct=np.stack([matrices[i] for i in selected]))
    pairs=[]
    for i in eligible:
        before,after=matrices[i-1],matrices[i]
        broken=before&~after
        pairs.append({'before':str(versions[i-1]),'candidate':str(versions[i]),
            'warmup_before_correct':int(before.sum()),'warmup_gains':int((~before&after).sum()),
            'new_class_support':int(broken.sum()),'maturity_supported':int(broken.sum())>=32,
            'by_map':{m:int(sum(c and t.map_name==m for c,t in zip(broken,transitions)))
                      for m in sorted({t.map_name for t in transitions})}})
    summary={'status':'completed','pairs':pairs,'programs_evaluated':len(selected),
        'program_transition_evaluations':len(selected)*len(transitions),'reused_native_transitions':len(transitions),
        'pairs_with_mature_support':sum(r['maturity_supported'] for r in pairs),
        'wall_time_s':time.monotonic()-started,'api_requests':0,'gpu_work':False,'new_environment_interactions':0,
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Support screen only. Does not reconstruct author history, generate a rejected candidate, train an explorer or demonstrate learning/task harm.'}
    save(args.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
