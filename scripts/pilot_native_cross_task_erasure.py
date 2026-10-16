"""Do genuine temperature-prediction errors erase useful circuit rules?

Replay fixed, already simulator-executed trajectories through the original
predict/update loop. Predictions are generated before revealing each recorded
next observation. Every setup action is learned. No memory text is edited.
This is a fixed action curriculum replay, not an adaptive online attacker.
"""
import argparse
import json
from pathlib import Path
from wm_security.client import Client
from wm_security.upstream import text_world_model
from pilot_conductivity import lamp_status


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (root/name).write_text(json.dumps(value, indent=2))
    save('config.json', vars(args))
    client = Client(args.model, root, max_calls=500)
    initial = json.loads(Path(args.checkpoint).read_text())
    queries = json.loads(Path(args.queries).read_text())['queries']
    task = json.loads(Path(args.thermal_data).read_text())['goal']
    benign = json.loads(Path(args.benign_data).read_text())['inspect_instrument']
    surprise = json.loads(Path(args.surprise_data).read_text())['liquid']
    protected = [k for k, v in initial['semantic']['registry']['entries'].items()
                 if 'light bulb' in v['text']]
    save('source.json', dict(checkpoint=args.checkpoint, protected_keys=protected,
                            benign_source=args.benign_data, surprise_source=args.surprise_data))
    summary = []
    current = {}

    def evaluate(model, condition, stage):
        result = []
        for query in queries:
            model.llm.tag = f'probe/{stage}/{query["id"]}'
            pred = model.predict(query['state'], query['action'], query['goal'])
            result.append(dict(view=query['id'], prediction=pred,
                               lamp_on=lamp_status(pred['prediction']), actual=query['actual']))
        save(f'{condition}_{stage}_probes.json', result)
        return result

    def evidence(model):
        entries = model.registry.to_dict()['entries']
        return {k: entries[k]['evidence'] if k in entries else None for k in protected}

    try:
        for condition in args.conditions:
            trajectory = dict(benign=benign, surprise=surprise)[condition]
            current = dict(condition=condition, phase='initial')
            model = text_world_model(client, condition)
            model.load_checkpoint_state(initial)
            if args.disable_negative_credit:
                # Explicit researcher control: retain original positive credit,
                # prediction, extraction, and storage, but disable global blame.
                model.registry.credit_mismatch = lambda displayed_keys: None
            baseline = evaluate(model, condition, 'before')
            if any(r['lamp_on'] is not False for r in baseline):
                save('status.json', dict(status='initial_gate_failed', **current, results=summary))
                return
            learning = []
            phases = [('curriculum', trajectory['events'][:args.steps])]
            if not args.start_after_setup:
                phases.insert(0, ('setup', trajectory['setup']))
            for phase, records in phases:
                for index, record in enumerate(records):
                    current = dict(condition=condition, phase=phase, step=index)
                    if 'No known action matches' in record['observation']:
                        raise ValueError('Physical probe contained an invalid action')
                    before = evidence(model)
                    model.llm.tag = f'{phase}/{index}/forecast'
                    pred = model.predict(record['state'], record['action'], task)
                    model.llm.tag = f'{phase}/{index}/update'
                    model.update(record['state'], record['action'], pred['prediction'],
                                 record['observation'], {'task':task})
                    mismatch = model._semantic_is_mismatch(pred['prediction'], record['observation'])
                    event = dict(phase=phase, index=index, **record, prediction=pred, mismatch=mismatch,
                                 evidence_before=before, evidence_after=evidence(model), update_completed=True)
                    learning.append(event)
                    save(f'{condition}_learning.json', learning)
                    save(f'{condition}_memory.json', model.checkpoint_state())
                    print(json.dumps(dict(**current, mismatch=mismatch, evidence=evidence(model))), flush=True)
                    if phase == 'curriculum' and (index+1) % 10 == 0:
                        evaluate(model, condition, f'after_{index+1}')
                        save(f'{condition}_memory_{index+1}.json', model.checkpoint_state())
                if phase == 'setup':
                    setup_probes = evaluate(model, condition, 'after_setup')
                    save(f'{condition}_setup_memory.json', model.checkpoint_state())
                    if any(r['lamp_on'] is not False for r in setup_probes):
                        save('status.json', dict(status='setup_already_degraded', condition=condition,
                             results=summary, interpretation='True preparation already changed the circuit prediction; no isolated measurement-curriculum effect claimed.'))
                        return
            result = dict(condition=condition, executed_actions=len(learning), evidence=evidence(model),
                          evaluation=evaluate(model, condition, 'final'))
            summary.append(result)
            save('results.json', summary)
    except Exception as exc:
        save('status.json', dict(status='interrupted', **current, error=str(exc), results=summary))
        raise
    save('status.json', dict(status='completed', results=[{k:v for k,v in r.items() if k!='evaluation'} for r in summary],
         interpretation=('Fixed thermal action curricula replayed from genuine simulator records. '
             + ('All30 real setup updates already belong to the supplied checkpoint; only the continuation is replayed here. '
                if args.start_after_setup else 'All30 setup actions learned separately from each original trajectory. ')
             + 'Researcher has not removed rules; disabling global debit, if configured, is a researcher intervention. One circuit probe with two views; no task-loss result.')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-v4-pro')
    parser.add_argument('--checkpoint', default='runs/conductivity_calibrated_official_pro_v0/bypass_memory.json')
    parser.add_argument('--queries', default='configs/conductivity_calibrated_pro_queries.json')
    parser.add_argument('--thermal-data', default='runs/native_measurement_tasks_v0/measure-melting-point-unknown-substance.json')
    parser.add_argument('--benign-data', default='runs/thermal_cross_task_probe_v0/records.json')
    parser.add_argument('--surprise-data', default='runs/thermal_cross_task_probe_v1/records.json')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--conditions', nargs='+', choices=['benign','surprise'], default=['benign','surprise'])
    parser.add_argument('--disable-negative-credit', action='store_true',
                        help='Researcher intervention removing only native global mismatch debit, not an attacker action.')
    parser.add_argument('--start-after-setup', action='store_true',
                        help='Supplied checkpoint already contains the real30-action thermal setup; continue without learning it twice.')
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
