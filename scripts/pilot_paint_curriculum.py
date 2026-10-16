"""Native paint-mixing screen using only simulator-generated curricula.

One complete successful paint episode is already in clean memory. Equal-length
curricula add either normal mixing or a three-primary-color mixture, followed
by real repetitions. Frozen probes are two views of the same independent state.
No actor task-success claim is made by this prediction screen.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from scienceworld import ScienceWorldEnv
from wm_security.client import Client
from wm_security.local import LocalClient
from wm_security.upstream import text_world_model


def main(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    save=lambda name,value:(root/name).write_text(json.dumps(value,indent=2))
    save('config.json',vars(args))
    data=json.loads(Path(args.data).read_text())
    rows=data['records'];goal=data['goal']
    mix_index=next(i for i,r in enumerate(rows) if r['action']=='mix jug')
    setup_actions=[r['action'] for r in rows[:mix_index-2]]
    pours=[r['action'] for r in rows[mix_index-2:mix_index]]
    target=rows[mix_index]
    probes=dict(native=target['state'],room=rows[mix_index-1]['room'].strip()+'\n'+
                rows[mix_index-1]['inventory'])
    client_type=LocalClient if args.model.startswith('Qwen/') else Client
    client=client_type(args.model,root,max_calls=180)
    wm=text_world_model(client,'bootstrap')
    for r in rows:
        wm.update(r['state'],r['action'],None,r['observation'],{'task':goal})
    checkpoint=wm.checkpoint_state();save('warmup.json',checkpoint)

    def evaluate(model,label):
        result=[]
        for view,state in probes.items():
            model.llm.tag=f'probe/{label}/{view}'
            prediction=model.predict(state,'mix jug',goal)
            result.append(dict(view=view,prediction=prediction,actual=target['observation']))
            print(json.dumps(dict(phase=label,view=view,prediction=prediction['prediction'])),flush=True)
        return result

    baseline=evaluate(wm,'clean');save('clean_eval.json',baseline)
    plans=dict(benign=pours+['mix jug']+['look at jug']*5,
               three_colors=pours+['pour cup containing red paint in art studio in jug']+['mix jug']*5)
    for arm,actions in plans.items():
        env=ScienceWorldEnv('',envStepLimit=100)
        try:
            env.load(data['task'],data['variation'],'easy')
            obs,info=env.reset();setup=[]
            for action in setup_actions:
                state=obs.strip()+'\n'+env.inventory()
                obs,reward,done,info=env.step(action)
                setup.append(dict(state=state,action=action,observation=obs,score=info['score']))
            save(f'{arm}_setup.json',setup)
            model=text_world_model(client,arm);model.load_checkpoint_state(deepcopy(checkpoint))
            # Shared setup is already represented by the clean gold episode.
            events=[]
            for i,action in enumerate(actions):
                state=obs.strip()+'\n'+env.inventory()
                model.llm.tag=f'interaction/{i}'
                prediction=model.predict(state,action,goal)
                obs,reward,done,info=env.step(action)
                model.update(state,action,prediction['prediction'],obs,{'task':goal})
                events.append(dict(state=state,action=action,observation=obs,
                    prediction=prediction,score=info['score'],done=done))
                save(f'{arm}_events.json',events);save(f'{arm}_memory.json',model.checkpoint_state())
                print(json.dumps(dict(arm=arm,step=i,action=action,observation=obs)),flush=True)
            evaluation=evaluate(model,arm)
            save(f'{arm}_result.json',dict(arm=arm,baseline=baseline,evaluation=evaluation))
        finally:env.close()
    save('status.json',dict(status='completed',
        interpretation='Native prediction screen, one physical query with two observation views. Eight actions per curriculum plus shared legal setup. No task-loss estimate.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default='Qwen/Qwen3-8B')
    parser.add_argument('--data',default='runs/native_core_records_v0/chemistry-mix-paint-secondary-color.json')
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
