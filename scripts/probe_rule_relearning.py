"""Replay observed rule extractions and prepare narrow researcher interventions.

No agent action, observation, or original checkpoint is altered. The generated
checkpoints are explicitly researcher interventions, not attacker capabilities.
This isolates a native merge/filter behavior from an otherwise successful user
chain; it does not estimate task harm or establish a general attack.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from wm_security import native  # Register the pinned upstream import path.
from world_model.modules.typed_registry import TypedRegistry, RegistryEntry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='runs/pro_erasure_shared_users_v0')
    parser.add_argument('--output', default='runs/pro_rule_relearning_interventions_v0')
    args = parser.parse_args()
    source, root = Path(args.source), Path(args.output)
    root.mkdir(parents=True, exist_ok=False)

    def read(path):
        return json.loads(Path(path).read_text())

    def save(name, value):
        path = root / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        return str(path)

    key = 'look_at.light_bulb_state'
    calls = [json.loads(line) for line in (source/'calls.jsonl').read_text().splitlines()]
    before = read('runs/native_cross_task_erasure_pro_v0/surprise_memory.json')
    replay = TypedRegistry.from_dict(before['semantic']['registry'])
    extracts = []
    for row in calls:
        if not row['tag'].startswith('world/post/user/') or '/update/' not in row['tag']:
            continue
        response = row['response']['choices'][0]['message'].get('content', '')
        try:
            items = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or item.get('key') != key:
                continue
            entry = RegistryEntry.from_dict(item)
            assert entry.text == replay.entries[key].text
            evidence_before = replay.entries[key].evidence
            replay.register_or_merge(entry)
            extracts.append(dict(call_id=row['id'], tag=row['tag'], extraction=item,
                                 before=evidence_before, after=replay.entries[key].evidence,
                                 rendered=key in replay.compile_with_keys()[1]))
    assert len(extracts) == 3, 'Expected the three observed corrective extractions.'
    assert all(x['before'] == x['after'] < 0 and not x['rendered'] for x in extracts)

    post = read(source/'post_user_3_memory.json')
    benign = read(source/'benign_user_3_memory.json')
    memories = dict(post=str(source/'post_user_3_memory.json'),
                    benign=str(source/'benign_user_3_memory.json'))

    # Restore only the score of the exact rule actually re-extracted by users.
    single = deepcopy(post)
    single['semantic']['registry']['entries'][key]['evidence'] = 1.0
    memories['post_reactivate_reextracted_rule'] = save('post_reactivate_reextracted_rule.json', single)

    both = deepcopy(single)
    both['semantic']['registry']['entries']['look_around.light_bulb_state']['evidence'] = 1.0
    memories['post_reactivate_both_bulb_rules'] = save('post_reactivate_both_bulb_rules.json', both)

    # The benign chain learned this false rule during a genuinely successful
    # task. Removing it is a diagnostic of the additional control degradation.
    removed = deepcopy(benign)
    removed_entry = removed['semantic']['registry']['entries'].pop('move.object_connection')
    memories['benign_remove_new_movement_rule'] = save('benign_remove_new_movement_rule.json', removed)

    queries = read('configs/conductivity_calibrated_pro_queries.json')['queries']
    save('query_spec.json', dict(memories=memories, queries=queries))
    save('merge_replay.json', dict(extractions=extracts, removed_benign_entry=removed_entry,
         scope='Replay only the observed native merge events; not all online updates. '
               'Exact duplicate extraction preserves negative evidence, so the filter keeps it hidden. '
               'A differently worded extraction can create a new positive entry; this is not irreversibility.'))
    save('status.json', dict(status='completed',
         interpretation='Interventions prepared and native duplicate-merge behavior reproduced from '
                        'three actual corrective extractions. Query effects require the separate frozen probe.'))
    print(json.dumps(dict(output=str(root), actual_corrective_extractions=len(extracts),
                          conditions=list(memories))))


if __name__ == '__main__':
    main()
