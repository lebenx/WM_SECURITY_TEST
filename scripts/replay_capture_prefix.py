"""Recover the attack-only checkpoint by exact replay of recorded model calls.

No model is called, and no rule is manually reconstructed. Every prompt, tag,
and intermediate registry must exactly match the original native execution.
"""
import argparse
import json
from pathlib import Path
from wm_security.upstream import text_world_model


class RecordedClient:
    def __init__(self,records):
        self.records=records
        self.position=0

    def chat(self,messages,max_tokens,tag):
        record=self.records[self.position]
        assert messages==record['request']['messages'],f'Prompt mismatch at {tag}'
        assert tag==record['tag'],f'Tag mismatch at {tag}'
        self.position+=1
        return record['response']['choices'][0]['message']['content']


def main(args):
    source=Path(args.source)
    calls=[json.loads(line) for line in (source/'calls.jsonl').read_text().splitlines()]
    calls=[row for row in calls if row['tag'].startswith('targeted_failures/interaction/')]
    client=RecordedClient(calls)
    wm=text_world_model(client,'targeted_failures')
    wm.load_checkpoint_state(json.loads((source/'warmup.json').read_text()))
    events=json.loads((source/'targeted_failures_events.json').read_text())
    events=[e for e in events if e['tag'].startswith('interaction/')]
    data=json.loads(Path(args.data).read_text())
    goal=next(r['goal'] for r in data['probes'] if r.get('official_path'))
    for event in events:
        wm.llm.tag=event['tag']
        out=wm.predict(event['state'],event['action'],goal)
        assert out['prediction']==event['prediction']['prediction']
        assert out['retrieved_indices']==event['prediction']['retrieved_indices']
        wm.update(event['state'],event['action'],out['prediction'],event['observation'],{'task':goal})
        assert wm.registry.to_dict()==event['registry_after'],event['tag']
    assert client.position==len(calls)
    path=source/'targeted_failures_attack_only_memory.json'
    path.write_text(json.dumps(wm.checkpoint_state(),indent=2))
    result=dict(status='exact_replay_verified',model_calls_replayed=client.position,
                external_model_calls=0,actions_replayed=len(events),
                saved_checkpoint=str(path),
                interpretation='Five original real failures; no corrective interactions in this checkpoint. Every original prompt, response, retrieval, and intermediate rule registry replayed.')
    (source/'attack_only_replay.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='runs/scienceworld_capture_local')
    parser.add_argument('--data',default='runs/scienceworld_freezer_local/simulator_records.json')
    main(parser.parse_args())
