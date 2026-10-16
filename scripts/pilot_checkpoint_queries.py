"""Frozen native checkpoint queries and explicitly named component swaps."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
from wm_security.client import Client
from wm_security.local import LocalClient
from wm_security.upstream import text_world_model


def main(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    save=lambda name,obj:(root/name).write_text(json.dumps(obj,indent=2))
    spec=json.loads(Path(args.spec).read_text());save('spec.json',spec);save('config.json',vars(args))
    memories={k:json.loads(Path(v).read_text()) for k,v in spec['memories'].items()}
    for label,swap in spec.get('swaps',{}).items():
        memories[label]=deepcopy(memories[swap['base']])
        memories[label][swap['component']]=deepcopy(memories[swap['donor']][swap['component']])
    conditions=args.conditions or list(memories)
    kind=LocalClient if args.model.startswith('Qwen/') else Client
    client=kind(args.model,root,max_calls=100)

    def run(condition,query,repeat):
        model=text_world_model(client,f'{condition}/{query["id"]}/{repeat}')
        model.load_checkpoint_state(deepcopy(memories[condition]))
        prediction=model.predict(query['state'],query['action'],query['goal'])
        return dict(status='completed',condition=condition,query=query['id'],repeat=repeat,
                    prediction=prediction,actual=query['actual'])

    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs={executor.submit(run,condition,query,repeat):(condition,query['id'],repeat) for repeat in range(args.repeats)
              for condition in conditions for query in spec['queries']
              if not args.views or query['id'] in args.views}
        for future in as_completed(jobs):
            try:
                result=future.result()
            except Exception as exc:
                condition,query,repeat=jobs[future]
                result=dict(status='interrupted',condition=condition,query=query,repeat=repeat,error=str(exc))
            results.append(result);save('results.json',results)
            print(json.dumps(dict(condition=result['condition'],query=result['query'],
                repeat=result['repeat'],status=result['status'],
                prediction=result.get('prediction',{}).get('prediction'))),flush=True)
    save('status.json',dict(status='completed' if all(r['status']=='completed' for r in results) else 'partial',
        interpretation='Read-only queries of stored native checkpoints. Component swaps are researcher interventions. Repeats and views do not add independent physical instances.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='glm-5.3')
    parser.add_argument('--spec',required=True)
    parser.add_argument('--conditions',nargs='+')
    parser.add_argument('--views',nargs='+')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
