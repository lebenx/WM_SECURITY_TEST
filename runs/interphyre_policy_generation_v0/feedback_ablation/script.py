"""Frozen authored policies versus ordinary target and source-data GP search."""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from interphyre import InterphyreEnv

from screen_interphyre_continuous_search import LOW, SPAN, evaluate, object_prior, bo_proposal, save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reviewed_policy(folder):
    path = folder / 'policy.py'
    review = json.loads((folder / 'review.json').read_text())
    assert review['approved_for_local_evaluation'] and review['code_sha256'] == digest(path)
    code = path.read_text()
    tree = ast.parse(code)
    forbidden = {'open', 'eval', 'exec', 'compile', '__import__', 'globals', 'locals',
                 'getattr', 'setattr', 'delattr', 'input', 'breakpoint', 'load', 'save',
                 'loadtxt', 'savetxt', 'fromfile', 'tofile', 'memmap', 'savez'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = ([x.name.split('.')[0] for x in node.names] if isinstance(node, ast.Import)
                     else [node.module.split('.')[0]])
            assert set(roots) <= {'math', 'numpy'}, roots
        if isinstance(node, ast.Name):
            assert node.id not in forbidden, node.id
        if isinstance(node, ast.Attribute):
            assert not node.attr.startswith('_') and node.attr not in forbidden, node.attr
        assert not isinstance(node, (ast.Global, ast.Nonlocal))
    namespace = {}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['choose_action']


def scene_features(scene):
    fields = ('radius', 'length', 'thickness', 'bottom_width', 'top_width', 'height')
    values = []
    for name, obj in sorted(scene['objects'].items()):
        if obj['color'] == 'red':
            continue
        values.extend([obj['x']/10, obj['y']/10, obj['angle']/np.pi])
        values.extend(obj['size'].get(f, 0)/10 for f in fields)
    return np.array(values)


class SourceGP:
    """Product RBF over action and geometry, calibrated on source likelihood only."""
    def __init__(self, worlds, all_rows):
        context = np.array([scene_features(w['scene']) for w in worlds])
        self.context_mean = context.mean(axis=0)
        self.context_std = np.maximum(context.std(axis=0), .03)
        self.world_context = (context-self.context_mean)/self.context_std
        self.keys = [(w['level'], w['source_seed']) for w in worlds]
        indexed = {k:i for i,k in enumerate(self.keys)}
        unique = {}
        for row in all_rows:
            key = (row['level'], row['seed'])
            if key in indexed:
                unique[key, tuple(row['action'])] = row
        rows = list(unique.values())
        self.x = np.array([r['normalized'] for r in rows])
        self.y = np.array([r['score'] for r in rows])
        self.c = np.array([self.world_context[indexed[r['level'], r['seed']]] for r in rows])
        self.mean = float(self.y.mean())
        best = None
        for action_scale in (.5, 1., 2.):
            for context_scale in (.5, 1., 2., 4.):
                self.action_length = np.array([.15, .2, .25])*action_scale
                self.context_length = context_scale
                k = self.kernel(self.x, self.c, self.x, self.c)+np.eye(len(rows))*.01
                factor = cho_factor(k)
                centered = self.y-self.mean
                nll = .5*centered@cho_solve(factor, centered)+np.log(np.diag(factor[0])).sum()
                if best is None or nll < best[0]:
                    best = (float(nll), action_scale, context_scale)
        self.action_length = np.array([.15, .2, .25])*best[1]
        self.context_length = best[2]
        self.calibration = dict(source_unique_actions=len(rows), action_scale=best[1],
                                context_scale=best[2], negative_log_likelihood=best[0],
                                geometry_keys=[list(k) for k in self.keys])

    def kernel(self, x, c, z, d):
        action_d = np.sum(((x[:, None]-z[None])/self.action_length)**2, axis=-1)
        context_d = np.maximum(np.sum(c*c,axis=1)[:,None]+np.sum(d*d,axis=1)[None]-2*c@d.T,0)
        context_d /= c.shape[1]*self.context_length**2
        return np.exp(-.5*(action_d+context_d))

    def propose(self, scene, history, rng):
        context = (scene_features(scene)-self.context_mean)/self.context_std
        c = np.broadcast_to(context, (len(history), len(context)))
        x = np.vstack([self.x, np.array([r['normalized'] for r in history]).reshape(-1,3)])
        y = np.r_[self.y, [r['score'] for r in history]]
        contexts = np.vstack([self.c, c])
        # All source rows train the GP; high-scoring nearby rows also seed its continuous pool.
        distance = np.mean((self.c-context)**2, axis=1)
        top = np.argsort(self.y-.1*distance)[-12:]
        source_centers = self.x[rng.choice(top, size=128)]
        pools = [object_prior(scene, rng, 128),
                 np.clip(source_centers+rng.normal(0,.07,(128,3)),0,1)]
        if history:
            best = max(history, key=lambda r:r['score'])['normalized']
            pools.append(np.clip(np.asarray(best)+rng.normal(0,.1,(128,3)),0,1))
        else:
            pools.append(object_prior(scene, rng, 128))
        candidates = np.vstack(pools)
        candidate_c = np.broadcast_to(context, (len(candidates), len(context)))
        factor = cho_factor(self.kernel(x, contexts, x, contexts)+np.eye(len(x))*.01)
        cross = self.kernel(candidates, candidate_c, x, contexts)
        mean = self.mean+cross@cho_solve(factor,y-self.mean)
        variance = np.maximum(1-np.sum(cross*cho_solve(factor,cross.T).T,axis=1),1e-8)
        acquisition = mean+.5*np.sqrt(variance)
        if history:
            old = np.array([r['normalized'] for r in history])
            acquisition[np.min(np.linalg.norm(candidates[:,None]-old[None],axis=-1),axis=1)<1e-6] = -np.inf
        return candidates[np.argmax(acquisition)]


def invalid_record(action, reason):
    return dict(action=action, normalized=None, invalid=True, success=False, score=-1.,
                physics_steps=0, min_gap=None, max_contact_duration=0.,
                first_action_contact=None, policy_error=reason)


def timeout_handler(signum, frame):
    raise TimeoutError('Policy call exceeded 0.2 seconds')


def run(args):
    root = Path(args.run)
    out = root / ('feedback_ablation' if args.ablation else 'evaluation')
    out.mkdir(exist_ok=False)
    plan = json.loads((root/'plan.json').read_text())
    packet = json.loads((root/'source_packet.json').read_text())
    assert digest(packet['source_path']) == packet['source_sha256']
    policies = {f'generated_{name}':load_reviewed_policy(root/name)
                for name in ('with_experience','without_experience')}
    policy_hashes = {name:digest(root/name.replace('generated_','')/'policy.py') for name in policies}
    if args.ablation:
        f = policies['generated_with_experience']
        def masked_policy(scene, history, rng, events_only=False):
            for row in history:
                row['first_action_contact'] = None
                if not events_only:
                    row.update(score=0., success=False, max_contact_duration=0., min_gap=None,
                               closest_error=[0.,0.])
            return f(scene, history, rng)
        policies = {'generated_prior_only':lambda s,h,r:f(s,[],r),
                    'generated_no_event_feedback':lambda s,h,r:masked_policy(s,h,r,True),
                    'generated_no_outcome_feedback':masked_policy}
        plan = dict(plan, arms=list(policies),
                    scope='Post hoc causal diagnosis on the same eight development targets, not independent confirmation.',
                    ablation='Same unmodified authored code and RNG seeds. Prior-only supplies empty history; no-event hides first-contact labels; no-outcome also hides scores/contact durations/displacements while retaining previous actions, invalid flags and attempt index.')
        save(out/'plan.json',plan)
    freeze = dict(plan_sha256=digest(root/'plan.json'), source_sha256=packet['source_sha256'],
                  policies=policy_hashes,
                  evaluator_sha256=digest(__file__),
                  frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  target_feedback_before_freeze=bool(args.ablation), policy_timeout_s=.2,
                  raw_policy_actions='Reject out-of-range, nonfinite, malformed actions; never silently clip.',
                  source_gp='Full unique source data; product action/geometry RBF; 12 source-only likelihood settings; 384 candidate UCB pool.',
                  target_gp='Five charged object-prior initial attempts then unchanged 256-candidate RBF UCB.',
                  reset_protocol='Fresh native environment per physical query; runtime policy gets only public scene and observed own history.')
    save(out/'freeze.json',freeze)
    (out/'script.py').write_text(Path(__file__).read_text())
    all_rows = [json.loads(line) for line in Path(packet['source_path']).read_text().splitlines()]
    fit_start = time.monotonic()
    models = {} if args.ablation else {
        level:SourceGP([w for w in packet['worlds'] if w['level']==level],all_rows)
        for level in plan['levels']}
    source_fit_seconds = time.monotonic()-fit_start
    save(out/'source_gp_calibration.json',{key:v.calibration for key,v in models.items()})
    signal.signal(signal.SIGALRM, timeout_handler)
    counts = dict(placement_queries=0, valid_rollouts=0, invalid_queries=0, physics_steps=0, policy_errors=0)
    results = []
    start = time.monotonic()
    with (out/'native_traces.jsonl').open('w') as log:
        for li, level in enumerate(plan['levels']):
            for seed in plan['target_seeds']:
                env = InterphyreEnv(level,seed=seed)
                scene = env.describe_scene()
                env.close()
                for arm in plan['arms']:
                    rng = np.random.default_rng(10201+seed+li*10000)
                    initial = object_prior(scene,np.random.default_rng(10101+seed+li*10000),5)
                    history = []
                    for q in range(plan['queries']):
                        t = time.monotonic()
                        record = None
                        if arm in policies:
                            try:
                                signal.setitimer(signal.ITIMER_REAL,.2)
                                action = np.asarray(policies[arm](copy.deepcopy(scene),copy.deepcopy(history),rng),dtype=float)
                                if action.shape != (3,) or not np.isfinite(action).all():
                                    raise ValueError('Policy returned a malformed/nonfinite action')
                                if np.any(action<LOW) or np.any(action>LOW+SPAN):
                                    record=invalid_record(action.tolist(),'Policy action outside the stated coordinate/radius bounds')
                                u = (action-LOW)/SPAN
                            except Exception as exc:
                                record=invalid_record(None,f'{type(exc).__name__}: {exc}')
                            finally:
                                signal.setitimer(signal.ITIMER_REAL,0)
                        elif arm == 'target_gp_bo':
                            u = initial[q] if q<5 else bo_proposal(history,scene,rng)
                        else:
                            u = models[level].propose(scene,history,rng)
                        proposal_seconds = time.monotonic()-t
                        if record is None:
                            query_env = InterphyreEnv(level,seed=seed)
                            try:
                                record = evaluate(query_env,u)
                            finally:
                                query_env.close()
                        history.append(record)
                        logged = dict(record,level=level,seed=seed,arm=arm,query=q+1,proposal_seconds=proposal_seconds)
                        log.write(json.dumps(logged)+'\n');log.flush()
                        counts['placement_queries']+=1
                        counts['valid_rollouts']+=int(not record['invalid'])
                        counts['invalid_queries']+=int(record['invalid'])
                        counts['physics_steps']+=record['physics_steps']
                        counts['policy_errors']+=int('policy_error' in record)
                        if record['success']:
                            break
                    result=dict(level=level,seed=seed,arm=arm,success=history[-1]['success'],
                                queries=len(history),best_score=max(r['score'] for r in history),
                                invalid_queries=sum(r['invalid'] for r in history),
                                policy_errors=sum('policy_error' in r for r in history))
                    results.append(result);print(json.dumps(result),flush=True)
    save(out/'task_results.json',results)
    aggregate={}
    for arm in plan['arms']:
        rows=[r for r in results if r['arm']==arm]
        aggregate[arm]=dict(successes=sum(r['success'] for r in rows),tasks=len(rows),
                           queries=sum(r['queries'] for r in rows),invalid_queries=sum(r['invalid_queries'] for r in rows),
                           policy_errors=sum(r['policy_errors'] for r in rows))
    summary=dict(status='completed',run=str(out),results=aggregate,counts=counts,
                 elapsed_s=time.monotonic()-start,source_gp_fit_s=source_fit_seconds,
                 source_queries_reused=565,new_source_queries=0,
                 evaluation_api_requests=0,authoring_api_requests=0 if args.ablation else 2,
                 authoring_requests_reused=2 if args.ablation else 0,gpu_work=False,
                 scope=plan['scope'],finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--ablation',action='store_true');run(parser.parse_args())
