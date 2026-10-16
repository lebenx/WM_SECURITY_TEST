"""Fixed nonlinear origin probe; reuses existing latents without model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(args.latents)
    data={s:np.load(root/f'{s}_latents.npz') for s in ['train','dev','test']}
    dimension=data['train']['real'].shape[-1]
    for d in data.values():
        assert d['real'].shape==d['imagined'].shape==(3,498,dimension)
    save=lambda name,d:(out/name).write_text(json.dumps(d,indent=2)+'\n')
    save('plan.json',dict(training='seed12 only',development_seeds=[13,17],
        prior_test_now_development=True,latent_dim=dimension,
        method='StandardScaler + RBF SVC(C=1,gamma=scale), no tuning',
        threshold='99.5th percentile pooled dev-real scores, higher order statistic',
        gate='Both development episodes real FPR<=1% and imagined TPR>=80% at depths1,2',
        if_passes='Evaluate frozen detector on new untouched native walker-walk seeds18,19, requiring clean returns>=900.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        latent_hashes={s:hashlib.sha256((root/f'{s}_latents.npz').read_bytes()).hexdigest() for s in data},
        no_new_world_model_calls=True,no_native_steps=True,api_requests=0))
    start=time.monotonic();tr=data['train']
    real=tr['real'].reshape(-1,dimension);imagined=tr['imagined'].reshape(-1,dimension)
    x=np.concatenate([real,imagined]);y=np.r_[np.zeros(len(real)),np.ones(len(imagined))]
    m=make_pipeline(StandardScaler(),SVC(C=1.,gamma='scale'));m.fit(x,y)
    assert list(m[1].classes_)==[0.,1.]
    scores={}
    for s in ['dev','test']:
        scores[s]={}
        for k in ['real','imagined']:
            shape=data[s][k].shape
            flat=m.decision_function(data[s][k].reshape(-1,dimension))
            assert flat.shape==(shape[0]*shape[1],)
            scores[s][k]=flat.reshape(shape[:2])
    threshold=float(np.quantile(np.concatenate([v['real'].ravel() for v in scores.values()]),.995,method='higher'))
    results={}
    for s,v in scores.items():
        rows=[]
        for j in range(3):
            a,b=v['real'][j],v['imagined'][j]
            assert len(a)==len(b)==498
            rows.append(dict(depth=j+1,real_fpr=float((a>threshold).mean()),
                imagined_tpr=float((b>threshold).mean()),real_false_positives=int((a>threshold).sum()),
                imagined_true_positives=int((b>threshold).sum()),pairs=len(a),
                auc=float(roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))],np.r_[a,b]))))
        results[s]=rows
    passed=all(z['real_fpr']<=.01 and z['imagined_tpr']>=.8 for rows in results.values() for z in rows[:2])
    np.savez_compressed(out/'probe.npz',mean=m[0].mean_,scale=m[0].scale_,
        support_vectors=m[1].support_vectors_,dual_coef=m[1].dual_coef_,
        intercept=m[1].intercept_,gamma=m[1]._gamma)
    save('summary.json',dict(status='completed',development_gate_passed=passed,
        threshold=threshold,latent_dim=dimension,development_results=results,
        fit_and_evaluation_cpu_seconds=time.monotonic()-start,
        native_environment_steps=0,world_model_training_updates=0,api_requests=0,gpu_work=False,
        valid_rows_verified=True,scope='Development-only necessary condition; not attack evidence.'))
    print((out/'summary.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--latents',default='runs/tdmpc_latent_origin_v0')
    main(p.parse_args())
