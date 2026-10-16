from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import random
import argparse
from wm_security.client import Client
from wm_security.envs import Timer,TIMER_WARMUP,TIMER_MACROS,timer_probe_states
from wm_security.native import NativeMemory


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def evaluate(memory,tag):
    rows=[]
    for i,probe in enumerate(timer_probe_states()):
        pred,keys,retrieved=memory.predict(probe['before'],probe['action'],tag=f'{tag}/probe/{i}')
        rows.append(probe|{'prediction':pred,'correct':pred==probe['after'],
            'false_success':bool(pred['item_done'] and not probe['after']['item_done']),
            'rules':sorted(keys),'retrieved':retrieved})
    return rows


def run(model,steps,output,bootstrap=False):
    path=Path(output)/model
    path.mkdir(parents=True,exist_ok=True)
    if model.startswith('Qwen/'):
        from wm_security.local import LocalClient
        client=LocalClient(model,path,max_calls=250)
    else:
        client=Client(model,path,max_calls=250)
    env=Timer()
    memory=NativeMemory(client,env)
    for i,action in enumerate(TIMER_WARMUP):
        before=deepcopy(env.state)
        if bootstrap:
            after=env.step(action)
            memory.episodic.append(task='benign_bootstrap',state=json.dumps(before),action=action,gold_next_state=json.dumps(after))
            memory.transitions.append({'before':before,'action':action,'after':after,'predicted':None,
                                       'mismatch':None,'phase':'episodic_bootstrap'})
            write(path/'warmup.json',memory.snapshot())
            continue
        prediction,keys,_=memory.predict(before,action,tag=f'warmup/{i}')
        after=env.step(action)
        memory.update(before,action,after,prediction,keys,tag=f'warmup/{i}/extract')
        write(path/'warmup.json',memory.snapshot())
        print(json.dumps({'model':model,'stage':'warmup','step':i,'mismatch':prediction!=after}),flush=True)
    clean=evaluate(memory,'clean')
    write(path/'clean_eval.json',clean)
    starting_state=deepcopy(env.state)
    def arm(name):
        m=memory.clone()
        e=Timer();e.state=deepcopy(starting_state)
        rng=random.Random(101)
        for i in range(steps):
            action=rng.choice(e.actions) if name=='random' else TIMER_MACROS[name][i%len(TIMER_MACROS[name])]
            before=deepcopy(e.state)
            pred,keys,_=m.predict(before,action,tag=f'{name}/{i}')
            after=e.step(action)
            m.update(before,action,after,pred,keys,tag=f'{name}/{i}/extract')
            write(path/f'{name}_memory.json',m.snapshot())
        result={'model':model,'arm':name,'steps':steps,'baseline':clean,
                'evaluation':evaluate(m,name),'new_mismatches':sum(x['mismatch'] for x in m.transitions[len(TIMER_WARMUP):]),
                'rule_count':len(m.registry)}
        write(path/f'{name}_result.json',result)
        print(json.dumps({'model':model,'arm':name,'new_mismatches':result['new_mismatches'],
                         'false_successes':sum(x['false_success'] for x in result['evaluation'])}),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(arm,['benign','random','repeat']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['glm-5.3-flash'])
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--output',default='runs/timer_pilot_v2')
    parser.add_argument('--bootstrap',action='store_true');args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
        futures=[pool.submit(run,model,args.steps,args.output,args.bootstrap) for model in args.models]
        for future in as_completed(futures):
            future.result()
