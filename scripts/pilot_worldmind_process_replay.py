"""Second native learner screen using fixed, authentic host prediction errors.

WorldMind's original plugin processes the full 17-action conductivity sessions
logged by the earlier native WorldEvolver host. No text or observations are
fabricated. Forecasts are frozen source-host outputs, not predictions generated
by a WorldMind-assisted host. This isolates transfer through native process
reflection/retrieval; it is not an end-to-end WorldMind attack evaluation.
"""
import argparse
import json
from pathlib import Path
import shutil

from wm_security.local import LocalClient
from wm_security.client import Client
from wm_security.worldmind_native import NativeWorldMind, GoalTrajectoryStep


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (root/name).write_text(json.dumps(value, indent=2))
    save('config.json', vars(args))
    kind = LocalClient if args.model.startswith('Qwen/') else Client
    client = kind(args.model, root, max_calls=90)
    data = json.loads(Path(args.data).read_text())
    source = Path(args.source)
    goal = data['goal']
    warm = NativeWorldMind(client, root/'clean', 'worldmind/clean')
    warm.tag = 'warmup'
    trajectory = [GoalTrajectoryStep(action=r['action'], observation=r['observation'],
                                    env_feedback=r['observation']) for r in data['records']]
    extracted = warm.goal.extract_experience(goal, trajectory)
    warm.check_transport()
    save('warmup_extraction.json', dict(extracted=extracted))
    if not extracted:
        save('status.json', dict(status='warmup_extraction_failed'))
        return
    save('clean_retrieval.json', warm.retrieve(goal))
    common = json.loads((source/'shared_setup_events.json').read_text())
    histories = []
    results = []
    try:
        shutil.copytree(root/'clean', root/'common')
        model = NativeWorldMind(client, root/'common', 'worldmind/common')

        def process(model, events, history, name):
            records = []
            for index, event in enumerate(events):
                if not event.get('update_completed'):
                    raise ValueError('Source host event was incomplete')
                model.tag = f'{index}/{event["action"]}'
                mismatch, experiences = model.update(goal, event, history)
                records.append(dict(source_event=event, mismatch=mismatch, experiences=experiences,
                    discrimination=model.discriminations[-1] if model.discriminations else None))
                save(f'{name}_learning.json', records)
                history.append(f'Action: {event["action"]}, Feedback: {event["observation"]}')
                print(json.dumps(dict(phase=name, step=index, mismatch=mismatch,
                                      experiences=experiences)), flush=True)
            return history

        histories = process(model, common, histories, 'common')
        save('common_retrieval.json', model.retrieve(goal))
        for arm in ['proper', 'bypass']:
            shutil.copytree(root/'common', root/arm)
            model = NativeWorldMind(client, root/arm, f'worldmind/{arm}')
            events = json.loads((source/f'{arm}_events.json').read_text())
            process(model, events, list(histories), arm)
            model.tag = 'retrieve'
            retrieved = model.retrieve(goal)
            save(f'{arm}_retrieval.json', retrieved)
            results.append(dict(arm=arm, process_experiences=model.retrieval.get_process_experience_count(),
                                selected=len(retrieved['process_experiences'])))
    except Exception as exc:
        save('status.json', dict(status='interrupted', results=results, error=str(exc)))
        raise
    save('status.json', dict(status='completed', results=results,
         interpretation='Original WorldMind plugin learning/refining the fixed native host errors on genuine 17-action sessions. Same-task successful goal experience is excluded by original retrieval. Transport replaced and broken package re-export initializer bypassed; implementation submodules unchanged. Native word-overlap fallback if embedding package absent. No WorldMind-host forecast, actor result, or task-loss claim.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--data', default='runs/native_core_records_v0/test-conductivity-of-unknown-substances.json')
    parser.add_argument('--source', default='runs/conductivity_complete_learning_local_v0')
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
