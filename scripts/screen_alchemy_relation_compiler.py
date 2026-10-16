"""Native witness for source-learned relation composition, not a new-method claim.

Learn commuting two-action words from observed source traces, then compose
them into unseen three-action contrasts. Compare the resulting construction
with ordinary adjacent-rewrite closure and a same-data tabular source model.
Researcher-selected native fixtures and reset states make this a mechanism
screen, not a held-out scientific-discovery benchmark.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from itertools import combinations, permutations, product
import json
from pathlib import Path
import subprocess
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
from dm_alchemy import symbolic_alchemy
from dm_alchemy.types import graphs, stones_and_potions as sp, utils


class NativeLab:
    def __init__(self, folder):
        self.folder = folder
        self.actions = 0
        self.programs = 0
        self.by_phase = {}

    def execute(self, constraint, initial, word, phase):
        chemistry = utils.Chemistry(
            sp.all_fixed_potion_map(), sp.all_fixed_stone_map(),
            graphs.create_graph_from_constraint(constraint), np.eye(3))
        items = utils.EpisodeItems(
            stones=[[sp.Stone(0, list(initial))]],
            potions=[[sp.Potion(i, a // 2, -1 if a % 2 == 0 else 1)
                      for i, a in enumerate(word)]])
        env = symbolic_alchemy.get_symbolic_alchemy_fixed(
            items, chemistry, max_steps_per_trial=len(word) + 2)
        env.reset()

        def observed():
            stones = env.perceived_stones()
            assert len(stones) == 1
            stone = stones[0]
            return {'coords': [int(v) for v in stone.perceived_coords],
                    'reward': int(stone.reward)}

        observations = [observed()]
        for slot in range(len(word)):
            env.step_slot_based_action(utils.SlotBasedAction(
                stone_ind=0, potion_ind=slot))
            self.actions += 1
            self.by_phase[phase] = self.by_phase.get(phase, 0) + 1
            observations.append(observed())
        self.programs += 1
        record = {'phase': phase, 'initial': list(initial), 'word': list(word),
                  'observations': observations}
        with (self.folder / 'native_traces.jsonl').open('a') as out:
            out.write(json.dumps(record) + '\n')
        return record


def key(a, b):
    return tuple(sorted((a, b)))


def mine_relations(traces):
    endpoints, transition_table = {}, {}
    for record in traces:
        initial, word = tuple(record['initial']), tuple(record['word'])
        endpoints[(initial, word)] = record['observations'][-1]
        for i, action in enumerate(word):
            state = tuple(record['observations'][i]['coords'])
            after = tuple(record['observations'][i + 1]['coords'])
            table_key = state, action
            if table_key in transition_table:
                assert transition_table[table_key] == after
            transition_table[table_key] = after
    states = sorted({tuple(r['initial']) for r in traces})
    commuting = {key(a, b) for a, b in combinations(range(6), 2)
                 if all(endpoints[(s, (a, b))] == endpoints[(s, (b, a))]
                        for s in states)}
    return commuting, transition_table


def compile_contrasts(commuting):
    """Move c past a noncommuting block ab via two learned equalities."""
    return [((c, a, b), (a, b, c))
            for a, b, c in permutations(range(6), 3)
            if key(a, b) not in commuting
            and key(c, a) in commuting and key(c, b) in commuting]


def standard_rewrite_closure(word, commuting):
    """Ordinary congruence closure under adjacent commuting rewrites."""
    reached, queue = {word}, deque([word])
    while queue:
        current = queue.popleft()
        for i in range(len(current) - 1):
            if key(current[i], current[i + 1]) not in commuting:
                continue
            result = (current[:i] + (current[i + 1], current[i])
                      + current[i + 2:])
            if result not in reached:
                reached.add(result)
                queue.append(result)
    return reached


def forward(table, state, word):
    for action in word:
        state = table[(state, action)]
    return state


def evaluate_pairs(lab, constraint, initial, pairs, phase):
    results = []
    for left, right in pairs:
        a = lab.execute(constraint, initial, left, phase)
        b = lab.execute(constraint, initial, right, phase)
        results.append({'left': list(left), 'right': list(right),
                        'left_outcome': a['observations'][-1],
                        'right_outcome': b['observations'][-1],
                        'violated': a['observations'][-1] != b['observations'][-1]})
    return results


def main(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    save = lambda name, data: (folder / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    started = time.monotonic()
    source = graphs.bottleneck2_constraints()[0]
    target = graphs.bottleneck1_constraints()[0]
    initial = (-1, 1, -1)
    save('plan.json', {
        'upstream': 'https://github.com/google-deepmind/dm_alchemy',
        'pin': subprocess.check_output(
            ['git', '-C', 'vendor/dm_alchemy', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_fixture': 'native bottleneck2', 'target_fixture': 'native bottleneck1',
        'source_constraint_for_audit_only': source,
        'target_constraint_for_audit_only': target,
        'target_initial': list(initial),
        'selection': 'Researcher-selected compositional mechanism witness, specified before execution; not random held-out tasks.',
        'source_acquisition': 'All ordered distinct two-action words from all eight source states. Counts include every native action and reset program.',
        'learner_input': 'Observed source coordinates/rewards, action IDs and initial observations; no hidden graph or target data.',
        'constructor': 'If c commutes with a and b in the source, synthesize cab versus abc. Restrict ab to a source noncommuting pair to expose context dependence.',
        'comparators': ['Transferred primitive commutation tests at the same target start', 'Ordinary adjacent-rewrite closure from the same source relations', 'Exact tabular source transitions learned from the same traces'],
        'scope': 'Experiment-construction capability screen; no novel-method acceptance, downstream task-success or interaction-efficiency claim.',
        'api_requests': 0, 'gpu_seconds': 0,
    })
    lab = NativeLab(folder)
    states = list(product((-1, 1), repeat=3))
    traces = [lab.execute(source, state, word, 'source_acquisition')
              for state in states for word in permutations(range(6), 2)]
    commuting, table = mine_relations(traces)
    pairs = compile_contrasts(commuting)
    assert pairs and len(table) == 48
    save('source_artifact.json', {
        'commuting_pairs': sorted(commuting),
        'source_observation_states': len(states),
        'new_program_pairs': pairs,
        'source_primitive_transition_count': len(table),
        'source_acquisition_native_actions': lab.by_phase['source_acquisition'],
    })
    closure_reproduces = [right in standard_rewrite_closure(left, commuting)
                          for left, right in pairs]
    tabular_model_agrees = [all(forward(table, s, left) == forward(table, s, right)
                               for s in states) for left, right in pairs]
    # Freeze the generated programs before any native target execution.
    save('generated_programs.json', {
        'pairs': pairs, 'ordinary_closure_reproduces': closure_reproduces,
        'same_data_tabular_model_predicts_equality': tabular_model_agrees})
    primitive = [((a, b), (b, a)) for a, b in sorted(commuting)]
    primitive_target = evaluate_pairs(lab, target, initial, primitive, 'target_primitive')
    source_validation = evaluate_pairs(lab, source, initial, pairs, 'source_composition_validation')
    composed_target = evaluate_pairs(lab, target, initial, pairs, 'target_composed')
    save('primitive_target.json', primitive_target)
    save('composed_source_validation.json', source_validation)
    save('composed_target.json', composed_target)
    result = {
        'status': 'completed', 'learned_primitive_relations': len(commuting),
        'generated_composite_contrasts': len(pairs),
        'primitive_target_violations': sum(r['violated'] for r in primitive_target),
        'composite_source_violations': sum(r['violated'] for r in source_validation),
        'composite_target_violations': sum(r['violated'] for r in composed_target),
        'target_witnesses': [r for r in composed_target if r['violated']],
        'ordinary_rewrite_closure_reproduces_all': all(closure_reproduces),
        'same_data_tabular_model_predicts_all_source_equalities': all(tabular_model_agrees),
        'native_actions': lab.actions, 'native_program_runs': lab.programs,
        'native_actions_by_phase': lab.by_phase,
        'api_requests': 0, 'gpu_seconds': 0,
        'elapsed_s': time.monotonic() - started,
        'accepted_novel_method': False,
        'interpretation': 'Source traces generate unseen contrasts that can expose a changed contextual dependence. In this construction the generator is ordinary relation composition and every source equality is also predicted by a same-data tabular model. Native witnesses demonstrate capability, not novelty or an advantage over those strong comparators.',
        'limitations': 'One selected source-target pair, known identity perceptual mappings, researcher-controlled resets, exhaustive source-state coverage, no full discovery or reward task, and unequal costs between primitive and composed probes. Do not report a transfer success rate or sample-efficiency gain.',
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    save('summary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
