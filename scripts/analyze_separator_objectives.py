"""Exact counterexample to equating separator-proof depth with sample efficiency.

This is a finite mathematical construction, not a native environment benchmark,
not a trained policy, and not an evaluation of neural theorem proving. Every
available query is a primitive binary experiment. Both strategies always choose
a one-query separator for some currently viable pair, but their identification
costs differ because the other candidates need not be split equally well.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def answer(hypothesis, query):
    kind, index = query
    return int(hypothesis == index) if kind == 'singleton' else (hypothesis >> index) & 1


def evaluate(bits, policy):
    size = 2 ** bits
    queries = [('singleton', i) for i in range(size)] + [('bit', i) for i in range(bits)]
    trajectories = []
    for truth in range(size):
        viable = list(range(size))
        steps = []
        while len(viable) > 1:
            separating = [q for q in queries if len({answer(h, q) for h in viable}) > 1]
            assert separating
            if policy == 'first_verified_separator':
                query = separating[0]
            else:
                # Exact Bayesian one-step expected remaining hypothesis count.
                def expected_remaining(q):
                    counts = Counter(answer(h, q) for h in viable)
                    return sum(n * n for n in counts.values()) / len(viable)
                query = min(separating, key=expected_remaining)
            values = {h: answer(h, query) for h in viable}
            witness = next((a, b) for a in viable for b in viable if values[a] != values[b])
            observed = answer(truth, query)
            after = [h for h in viable if values[h] == observed]
            steps.append({'before': viable, 'query': query, 'separated_pair': witness,
                          'observed': observed, 'after': after,
                          'separator_program_length': 1})
            viable = after
        assert viable == [truth]
        trajectories.append({'truth': truth, 'steps': steps, 'query_count': len(steps)})
    costs = [r['query_count'] for r in trajectories]
    return {'policy': policy, 'mean_queries': sum(costs) / size,
            'worst_queries': max(costs), 'all_queries_are_one_primitive_separators': True,
            'costs_by_truth': costs, 'trajectories': trajectories}


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for bits in [2, 3, 4, 5]:
        first = evaluate(bits, 'first_verified_separator')
        balanced = evaluate(bits, 'expected_remaining_candidates')
        n = 2 ** bits
        assert first['mean_queries'] == (n + 1) / 2 - 1 / n
        assert balanced['mean_queries'] == bits
        assert balanced['worst_queries'] == bits
        rows.append({'hypotheses': n, 'first_verified_separator': first,
                     'expected_remaining_candidates': balanced})
    (root / 'enumeration.json').write_text(json.dumps(rows, indent=2) + '\n')
    summary = {
        'status': 'completed_exact_analytic_counterexample',
        'result': [{'hypotheses': r['hypotheses'],
                    'first_separator_mean_queries': r['first_verified_separator']['mean_queries'],
                    'balanced_mean_queries': r['expected_remaining_candidates']['mean_queries'],
                    'first_separator_worst_queries': r['first_verified_separator']['worst_queries'],
                    'balanced_worst_queries': r['expected_remaining_candidates']['worst_queries']}
                   for r in rows],
        'interpretation': 'The shortest verified separating program need not minimize experiments to identify a member of a larger candidate set. Improving search for a pairwise separator alone does not establish an experiment-efficiency advantage.',
        'limitations': 'Primitive separator length is not a measured theorem-prover proof depth or runtime. No proof tactic policy was trained. This refutes an implication between objectives, not all proof-directed experimental design or possible search-computation savings.',
        'native_environment_actions': 0, 'model_api_requests': 0, 'gpu_seconds': 0,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
