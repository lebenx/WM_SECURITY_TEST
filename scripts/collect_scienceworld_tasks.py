"""Collect official, simulator-generated trajectories for new core task families."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from scienceworld import ScienceWorldEnv


def main(args):
    root=Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'config.json').write_text(json.dumps(vars(args),indent=2))
    env=ScienceWorldEnv('',envStepLimit=200)
    results=[]
    try:
        for task in args.tasks:
            env.load(task,args.variation,'easy',generateGoldPath=True)
            obs,info=env.reset()
            goal=env.get_task_description()
            gold=env.get_gold_action_sequence()
            records=[]
            for action in gold:
                state=obs.strip()+'\n'+env.inventory()
                obs,reward,done,info=env.step(action)
                records.append(dict(state=state,action=action,observation=obs,
                                    room=env.look(),inventory=env.inventory(),
                                    reward=reward,score=info['score'],done=done))
                if done:break
            result=dict(task=task,variation=args.variation,goal=goal,records=records,
                        version=importlib.metadata.version('scienceworld'),
                        score=info['score'],success=info['score']==100,
                        interpretation='Official gold replay, not model-agent performance.')
            (root/f'{task}.json').write_text(json.dumps(result,indent=2))
            results.append({k:result[k] for k in ['task','score','success']})
            print(json.dumps(results[-1]),flush=True)
    finally:env.close()
    (root/'status.json').write_text(json.dumps(dict(status='completed',results=results),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',nargs='+',default=['chemistry-mix','chemistry-mix-paint-secondary-color',
                                                   'test-conductivity-of-unknown-substances'])
    parser.add_argument('--variation',type=int,default=0)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
