"""Post hoc diagnosis: give ordinary optimizers the identical generated source prior."""
import argparse
import copy
import datetime
import json
import time
from pathlib import Path

import cma
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from interphyre import InterphyreEnv
from screen_interphyre_continuous_search import LOW, SPAN, evaluate, save
from evaluate_interphyre_generated_policies import digest, load_reviewed_policy


def main(args):
    root=Path(args.run)
    out=root/'prior_matched_optimizers';out.mkdir(exist_ok=False)
    source_plan=json.loads((root/'plan.json').read_text())
    f=load_reviewed_policy(root/'with_experience')
    plan=dict(levels=source_plan['levels'],seeds=source_plan['target_seeds'],queries=25,
              arms=['generated_prior_gp','generated_prior_cma_es'],initial_queries=5,
              source_policy_sha256=digest(root/'with_experience'/'policy.py'),
              frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              prior='Unchanged experience-authored choose_action(scene, [], rng). Five paired charged initial queries.',
              gp='Identical target-history RBF UCB hyperparameters; 128 candidates from the generated prior and 128 Gaussian around current best.',
              cma='Ordinary CMA-ES at best of first five queries; sigma=.15 normalized, population=4, box constraints.',
              scope='Post hoc mechanism diagnosis on the same eight development tasks, not independent confirmation or a HExA/Causal-PIK reproduction.')
    save(out/'plan.json',plan);(out/'script.py').write_text(Path(__file__).read_text())
    counts=dict(placement_queries=0,valid_rollouts=0,invalid_queries=0,physics_steps=0)
    results=[];start=time.monotonic()
    with (out/'native_traces.jsonl').open('w') as log:
        for li,level in enumerate(plan['levels']):
            for seed in plan['seeds']:
                env=InterphyreEnv(level,seed=seed);scene=env.describe_scene();env.close()
                def prior(rng):
                    action=np.asarray(f(copy.deepcopy(scene),[],rng))
                    assert action.shape==(3,) and np.isfinite(action).all()
                    assert np.all(action>=LOW) and np.all(action<=LOW+SPAN)
                    return (action-LOW)/SPAN
                for arm in plan['arms']:
                    rng=np.random.default_rng(10201+seed+li*10000)
                    initial=[prior(rng) for _ in range(5)]
                    history=[];optimizer=None;pending=[];batch=[];losses=[]
                    for q in range(plan['queries']):
                        t=time.monotonic()
                        if q<5:
                            u=initial[q]
                        elif arm=='generated_prior_cma_es':
                            if optimizer is None:
                                best=max(history,key=lambda r:r['score'])['normalized']
                                optimizer=cma.CMAEvolutionStrategy(best,.15,{'bounds':[0.,1.],
                                    'popsize':4,'seed':10201+seed+li*10000,'verbose':-9,'verb_log':0})
                            if not pending:
                                pending=optimizer.ask();batch=[];losses=[]
                            u=np.asarray(pending.pop(0))
                        else:
                            x=np.array([r['normalized'] for r in history]);y=np.array([r['score'] for r in history])
                            best=x[np.argmax(y)]
                            candidates=np.vstack([[prior(rng) for _ in range(128)],
                                np.clip(best+rng.normal(0,.1,(128,3)),0,1)])
                            length=np.array([.15,.2,.25])
                            kernel=np.exp(-.5*np.sum(((x[:,None]-x[None])/length)**2,axis=-1))+np.eye(len(x))*.01
                            cross=np.exp(-.5*np.sum(((candidates[:,None]-x[None])/length)**2,axis=-1))
                            factor=cho_factor(kernel)
                            mean=y.mean()+cross@cho_solve(factor,y-y.mean())
                            var=np.maximum(1-np.sum(cross*cho_solve(factor,cross.T).T,axis=1),1e-8)
                            score=mean+.5*np.sqrt(var)
                            nearest=np.min(np.linalg.norm(candidates[:,None]-x[None],axis=-1),axis=1)
                            score[nearest<1e-6]=-np.inf
                            u=candidates[np.argmax(score)]
                        proposal_seconds=time.monotonic()-t
                        query_env=InterphyreEnv(level,seed=seed)
                        try:r=evaluate(query_env,u)
                        finally:query_env.close()
                        history.append(r);record=dict(r,level=level,seed=seed,arm=arm,query=q+1,proposal_seconds=proposal_seconds)
                        log.write(json.dumps(record)+'\n');log.flush()
                        counts['placement_queries']+=1;counts['valid_rollouts']+=int(not r['invalid'])
                        counts['invalid_queries']+=int(r['invalid']);counts['physics_steps']+=r['physics_steps']
                        if optimizer is not None:
                            batch.append(u);losses.append(-r['score'])
                            if not pending:optimizer.tell(batch,losses)
                        if r['success']:break
                    result=dict(level=level,seed=seed,arm=arm,success=history[-1]['success'],queries=len(history),
                                invalid_queries=sum(r['invalid'] for r in history),best_score=max(r['score'] for r in history))
                    results.append(result);print(json.dumps(result),flush=True)
    save(out/'task_results.json',results)
    aggregate={}
    for arm in plan['arms']:
        rows=[r for r in results if r['arm']==arm]
        aggregate[arm]=dict(successes=sum(r['success'] for r in rows),tasks=len(rows),queries=sum(r['queries'] for r in rows),
                           invalid_queries=sum(r['invalid_queries'] for r in rows))
    summary=dict(status='completed',run=str(out),results=aggregate,counts=counts,elapsed_s=time.monotonic()-start,
                 source_queries_reused=565,new_source_queries=0,new_api_requests=0,gpu_work=False,scope=plan['scope'],
                 finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args())
