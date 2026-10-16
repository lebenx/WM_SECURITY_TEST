#!/usr/bin/env python3
"""Analyze the step-aligned WorldAgen ranking calibration experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(values):
    x = np.asarray(values, dtype=float)
    return {'n': int(x.size), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)) if len(x)>1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    data = json.loads(args.input.read_text())
    rows = data['rows']
    scorers = data['scorers']
    conditions = sorted({condition for row in rows for condition in row['scores']})
    result = {'status': data['status'], 'protocol': data['protocol'], 'n_tasks': len(rows),
              'conditions': conditions, 'scorers': scorers, 'per_scorer': {}}
    for scorer in scorers:
        cond_result = {}
        for condition in conditions:
            selected_utility, regret, true_rank, score_true_corr, pred_mse = [], [], [], [], []
            selected_change = []
            for row in rows:
                utility = np.asarray([x['eef_approach_progress'] for x in row['candidate_replays']], dtype=float)
                score = np.asarray(row['scores'][condition][scorer], dtype=float)
                idx = int(np.argmax(score))
                selected_change.append(int(idx != int(np.argmax(row['scores']['clean'][scorer]))))
                best = int(np.argmax(utility))
                selected_utility.append(float(utility[idx]))
                regret.append(float(utility[best] - utility[idx]))
                true_rank.append(int(np.where(np.argsort(-utility) == idx)[0][0] + 1))
                score_true_corr.append(spearman(score, utility))
                pred_mse.append(float(np.asarray(row['candidate_forecast_mse'][condition])[idx]))
            cond_result[condition] = {
                'selected_eef_progress': summarize(selected_utility),
                'regret_vs_candidate_best': summarize(regret),
                'selected_true_utility_rank': summarize(true_rank),
                'score_vs_true_utility_spearman': summarize(score_true_corr),
                'selected_forecast_mse_step3': summarize(pred_mse),
                'selected_action_changed_vs_clean': summarize(selected_change),
            }
        result['per_scorer'][scorer] = cond_result

    result['candidate_random_baseline'] = {
        'expected_regret_mean': summarize([
            np.mean([max(x['eef_approach_progress'] for x in row['candidate_replays'])
                     - x['eef_approach_progress'] for x in row['candidate_replays']])
            for row in rows
        ]),
        'interpretation': 'Uniform random candidate expected regret; descriptive calibration only.',
    }
    result['inference_limits'] = [
        'six tasks and eight recorded candidates per task; no significance claim',
        'three simulator steps are deliberately aligned with the first WorldAgen forecast chunk and are an intermediate utility, not task completion',
        'the evaluation-only real-image CLIP score is not used to select actions',
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
