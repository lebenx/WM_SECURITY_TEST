"""Can a frozen native world model's latents disclose real vs imagined origin?

Necessary-condition probe only. No modified world model or attack result.
Matched actual action sequences remove counterfactual-action coverage as an
explanation; whole episodes, rather than correlated rows, define data splits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wm_security.tdmpc_native import load_agent
from pilot_tdmpc_query_interference import model_digest

TRACES = {
    'train': 'runs/tdmpc_action_search_v0/clean_12_episode.pt',
    'dev': 'runs/tdmpc_action_search_v0/clean_13_episode.pt',
    'test': 'runs/tdmpc_rank_coupling_v0/native/episode_17.pt',
}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


@torch.no_grad()
def latents(agent, trace):
    obs, actions = trace['obs'].to(agent.device), trace['action'].to(agent.device)
    task = trace['task']
    encoded = agent.model.encode(obs, task)
    # All horizons use the same start and end-time range (no time labels).
    n = len(actions) - 3 + 1
    start = encoded[:n]
    state = start
    data = {'real': [], 'imagined': []}
    for depth in range(1, 4):
        state = agent.model.next(state, actions[depth-1:depth-1+n], task)
        data['imagined'].append(state.cpu().numpy())
        data['real'].append(encoded[depth:depth+n].cpu().numpy())
    return {key: np.stack(values) for key, values in data.items()}


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), Path('src/wm_security/tdmpc_native.py'),
             *map(Path, TRACES.values())]
    save(out / 'plan.json', dict(
        checkpoint=args.checkpoint, traces=TRACES,
        hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='Offline necessary-condition classifier screen using an unchanged strong native TD-MPC2 checkpoint and its previously collected real trajectories.',
        split='Whole train seed12, dev seed13, test seed17 episodes; no row-level random split.',
        matching='Real encoded and model-predicted states correspond to the same actual action prefixes and physical future time; no random counterfactual actions.',
        probes=['L2 logistic regression, C=1, all512 latent features, no task/depth/time labels.',
                'PCA64 residual after train-real standardization.'],
        thresholds='Set separately per probe to dev-real score 99.5th percentile (higher), frozen before test scoring.',
        gate='Held-out real FPR <=1% and imagined TPR >=80% at BOTH depth1 and depth2, for at least one fixed probe; depth3 reported separately.',
        limits='A passing origin-discrimination gate is neither a new method nor task harm. OOD-gated model scaffolding and ordinary rollout drift are prior art.',
        new_native_environment_steps=0, api_requests=0, world_model_training_updates=0))
    start = time.monotonic()
    agent, provenance = load_agent(args.checkpoint)
    save(out / 'provenance.json', provenance)
    before = model_digest(agent.model)
    data, trace_metadata = {}, {}
    torch.cuda.synchronize()
    model_start = time.monotonic()
    for split, path in TRACES.items():
        trace = torch.load(path, map_location='cpu', weights_only=True)
        assert trace['task'] == agent.cfg.tasks.index('walker-walk')
        assert trace['reward'].sum().item() >= 900
        data[split] = latents(agent, trace)
        np.savez_compressed(out / f'{split}_latents.npz', **data[split])
        trace_metadata[split] = dict(seed=trace['seed'],
            recorded_steps=len(trace['action']), recorded_return=float(trace['reward'].sum()),
            matched_starts=data[split]['real'].shape[1])
    torch.cuda.synchronize()
    model_seconds = time.monotonic() - model_start
    assert model_digest(agent.model) == before
    # Repeat real states exactly as their paired imagined time marginals.
    real = data['train']['real'].reshape(-1, agent.cfg.latent_dim)
    imagined = data['train']['imagined'].reshape(-1, agent.cfg.latent_dim)
    x = np.concatenate([real, imagined])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(imagined))])
    fit_start = time.monotonic()
    logistic = make_pipeline(StandardScaler(),
        LogisticRegression(C=1., max_iter=1000, solver='lbfgs', random_state=2026))
    logistic.fit(x, y)
    scaler = StandardScaler().fit(real)
    pca = PCA(n_components=64, svd_solver='full').fit(scaler.transform(real))

    def scores(name, x):
        flat = x.reshape(-1, x.shape[-1])
        if name == 'logistic':
            values = logistic.decision_function(flat)
        else:
            z = scaler.transform(flat)
            values = ((z - pca.inverse_transform(pca.transform(z))) ** 2).mean(axis=1)
        return values.reshape(x.shape[:-1])

    results = {}
    for name in ['logistic', 'pca64']:
        dev_real = scores(name, data['dev']['real'])
        threshold = float(np.quantile(dev_real, .995, method='higher'))
        report = {'threshold': threshold, 'threshold_source': 'dev-real only', 'splits': {}}
        for split in TRACES:
            real_score = scores(name, data[split]['real'])
            imaginary_score = scores(name, data[split]['imagined'])
            rows = []
            for depth in range(3):
                r, g = real_score[depth], imaginary_score[depth]
                labels = np.concatenate([np.zeros(len(r)), np.ones(len(g))])
                rows.append(dict(depth=depth+1, real_fpr=float((r>threshold).mean()),
                    imagined_tpr=float((g>threshold).mean()),
                    auc=float(roc_auc_score(labels,np.concatenate([r,g]))), pairs=len(r)))
            report['splits'][split] = rows
        test = report['splits']['test']
        report['gate_passed'] = all(row['real_fpr']<=.01 and row['imagined_tpr']>=.8 for row in test[:2])
        results[name] = report
    fit_seconds = time.monotonic() - fit_start
    np.savez_compressed(out / 'probes.npz',
        logistic_mean=logistic[0].mean_, logistic_scale=logistic[0].scale_,
        logistic_coef=logistic[1].coef_, logistic_intercept=logistic[1].intercept_,
        pca_scaler_mean=scaler.mean_, pca_scaler_scale=scaler.scale_,
        pca_mean=pca.mean_, pca_components=pca.components_)
    result = dict(status='completed', gate_passed=any(r['gate_passed'] for r in results.values()),
        probes=results, trace_metadata=trace_metadata, model_digest=before,
        model_hash_unchanged=True, native_environment_steps=0, api_requests=0,
        world_model_training_updates=0, encoded_observations=3*501,
        predicted_transitions=3*3*498, probe_fit_cpu_seconds=fit_seconds,
        model_inference_elapsed_seconds=model_seconds, elapsed_seconds=time.monotonic()-start,
        gpu_work='Frozen inference only; probe fitting on CPU. No GPU training.',
        attack_run=False, native_task_harm=False)
    save(out / 'summary.json', result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    main(p.parse_args())
