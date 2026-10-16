"""Select genuine curricula by actual calibration return, holding out test seeds.

This strengthens the earlier fixed-bank proxy search. All 16 candidates are
evaluated; no search or stopping uses the final held-out returns. Full tasks,
not prediction errors, decide whether a confirmation test is warranted.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import torch
from wm_security.tdmpc_native import load_agent, episode
from pilot_tdmpc_action_search import snapshot, reset_agent, source_error, wave_trajectory
from pilot_tdmpc_adaptation import adapt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='runs/tdmpc_action_search_v0')
    p.add_argument('--checkpoint',default='models/tdmpc2/mt30-48M.pt')
    p.add_argument('--calibration-seed',type=int,default=11)
    p.add_argument('--test-seeds',nargs='+',type=int,default=[14,15])
    p.add_argument('--minimum-drop',type=float,default=100.)
    p.add_argument('--output',default='runs/tdmpc_return_search_v0')
    args=p.parse_args();root=Path(args.source);out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
    source_plan=json.loads((root/'plan.json').read_text())
    candidates=json.loads((root/'search.json').read_text())
    assert len(candidates)==16
    assert args.calibration_seed not in args.test_seeds
    plan=dict(vars(args),source_plan=source_plan,number_candidates=len(candidates),
              selection='Minimum actual500-step return on a calibration environment after40 original full updates; same fixed16-candidate bank from prior CEM proxy search.',
              permissions='White-box learner plus source/target reference simulators. Evaluation resets are excluded from selection. No transition or reward edits.',
              gate='Finish all16 candidates. Confirm only if the selected actual calibration return is at least100 below each of clean,benign-MPC,random controls.',
              source_reference_actions=source_plan['reference_simulator_action_budget'],
              additional_target_reference_actions=(len(candidates)+3)*500,
              deployed_actions=source_plan['actions'],
              scope='Mechanism pilot, not a novel algorithm or population estimate. Private reference victim execution is a stronger search capability than a model-only surrogate.')
    save('plan.json',plan);scores=[];results=[];started=time.monotonic()
    try:
        agent,provenance=load_agent(args.checkpoint);save('provenance.json',provenance)
        original=snapshot(agent)
        victim=source_plan['victim_task']
        entries=[('clean',None),('benign',root/'benign_deployed_data.pt'),('random',root/'random_deployed_data.pt')]
        entries += [(f'candidate_{r["candidate"]:02d}',root/f'candidate_{r["candidate"]:02d}.pt') for r in candidates]
        for label,data_path in entries:
            reset_agent(agent,original)
            if data_path is not None:
                data=torch.load(data_path,weights_only=False)
                error=source_error(agent,data)
                history=adapt(agent,data,source_plan['updates'],31,'native_full')
                save(f'{label}_learning.json',{'source':str(data_path),'pre_update_consistency_error':error,'history':history})
            result,data=episode(agent,victim,args.calibration_seed)
            result['condition']=label;scores.append(result)
            torch.save(data,out/f'{label}_calibration.pt');save('calibration_results.json',scores)
            print(json.dumps(result),flush=True)
        chosen=min(scores[3:],key=lambda r:r['return'])
        selected=int(chosen['condition'].split('_')[-1])
        baseline_min=min(r['return'] for r in scores[:3])
        support=baseline_min-chosen['return']>=args.minimum_drop
        max_error=max(candidates,key=lambda r:r['pre_update_consistency_error'])['candidate']
        save('selection.json',{'selected':selected,'max_error':max_error,'gate_passed':support,
                               'minimum_control_return':baseline_min,'selected_return':chosen['return'],
                               'gap':baseline_min-chosen['return'],'selection_locked_before_test':True})
        if support:
            for label,data_path in [('clean',None),('benign',root/'benign_deployed_data.pt'),
                                    ('max_error',root/f'candidate_{max_error:02d}.pt'),
                                    ('targeted',root/f'candidate_{selected:02d}.pt')]:
                reset_agent(agent,original)
                if data_path is not None:
                    data=torch.load(data_path,weights_only=False)
                    if label in {'max_error','targeted'}:
                        replay=wave_trajectory(agent,source_plan['source_task'],source_plan['source_seed'],
                                               data['theta'].numpy(),source_plan['actions'])
                        for key in ['obs','action','reward']:
                            torch.testing.assert_close(data[key],replay[key],rtol=0,atol=0)
                    source_error(agent,data)
                    adapt(agent,data,source_plan['updates'],31,'native_full')
                    if label=='targeted':torch.save({'model':agent.model.state_dict()},out/'targeted_model.pt')
                for seed in args.test_seeds:
                    result,data=episode(agent,victim,seed);result['condition']=label
                    results.append(result);save('results.json',results)
                    torch.save(data,out/f'{label}_{seed}_test.pt')
                    print(json.dumps(result),flush=True)
        save('status.json',{'status':'completed','calibration_gate_passed':support,'selected':selected,
                            'calibration_results':scores,'test_results':results,
                            'wall_seconds':time.monotonic()-started,
                            'interpretation':'Actual-return selection from16 genuine source curricula, stronger than the model proxy. No meaningful calibration gap means no held-out confirmation is run. Calibrated losses are selected development observations, not held-out estimates.'})
    except Exception as exc:
        save('status.json',{'status':'error','error':str(exc),'traceback':traceback.format_exc(),
                            'calibration_results':scores,'test_results':results});raise


if __name__=='__main__':main()
