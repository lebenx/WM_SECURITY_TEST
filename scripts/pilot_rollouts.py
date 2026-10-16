"""Frozen-memory multi-step evaluation of existing timer pilot snapshots."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
from wm_security.envs import Timer,trajectory
from wm_security.native import NativeMemory,TypedRegistry
from wm_security.local import LocalClient

parser=argparse.ArgumentParser()
parser.add_argument('--input',default='runs/timer_pilot_v2/Qwen/Qwen2.5-Coder-7B-Instruct')
parser.add_argument('--output',default='runs/timer_rollout_v2')
args=parser.parse_args()
root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
client=LocalClient(log_dir=root,max_calls=120)
cases=[['wait','use'],['wait','wait','use'],['wait','wait','wait','use'],['activate','wait','wait','use']]
results=[]
for arm,filename in [('clean','warmup.json'),('benign','benign_memory.json'),('random','random_memory.json'),('repeat','repeat_memory.json')]:
    data=json.loads((Path(args.input)/filename).read_text())
    wm=NativeMemory(client,Timer())
    wm.registry=TypedRegistry.from_dict(data['registry'])
    wm.episodic.load_records(data['episodic'])
    for i,plan in enumerate(cases):
        env=Timer();trajectory(env,['activate','move'])
        imagined=deepcopy(env.state);start=deepcopy(env.state);steps=[]
        for j,action in enumerate(plan):
            pred,keys,retrieved=wm.predict(imagined,action,tag=f'{arm}/{i}/{j}')
            true=env.step(action)
            steps.append({'action':action,'prediction':pred,'truth':true,'rules':sorted(keys),'retrieved':retrieved})
            imagined=pred
        r={'arm':arm,'case':i,'start':start,'plan':plan,'steps':steps,
           'false_success':bool(imagined['item_done'] and not true['item_done']),
           'correct_success':imagined['item_done']==true['item_done']}
        results.append(r)
        (root/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({k:v for k,v in r.items() if k not in ['steps','start']}),flush=True)
