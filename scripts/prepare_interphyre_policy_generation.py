"""Freeze a source-only policy-generation packet and untouched evaluation IDs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from interphyre import InterphyreEnv


INSTRUCTIONS = '''Write a pure Python experimental policy for the given native 2D physics task.
The goal is to place one red ball so the green and blue balls stay in contact for three seconds.
Gravity points downward. The observed scene contains positions, velocities, object type,
dynamic flags and sizes. Scene coordinates are [-5,5] in x and y. Legal radius is [0.1,1.5].
A placement must be fully inside the box and not overlap existing objects. The simulator
validates placements. There are 25 attempts per unseen scene; invalid placements also count.
Each query is a fresh independent copy of the initial scene, not a continuation.

Return ONLY Python code defining choose_action(scene, history, rng), returning [x,y,radius].
scene is the initial public scene dict: {"objects": {name: {"type":str,"color":str,
"x":float,"y":float,"vx":float,"vy":float,"angle":float,"angular_velocity":float,
"dynamic":bool,"size":{radius OR length,thickness OR bottom_width,top_width,height}}}, ...}.
history is a list of previous trial dicts with action, normalized, invalid, success, score,
physics_steps, min_gap, max_contact_duration, first_action_contact and (if valid) closest_error.
closest_error is green-minus-blue XY at minimum surface distance; min_gap is nonnegative.
score is 1-min_gap/initial_gap+maximum_contact_duration/3, native success is 2, invalid is -1.
rng is numpy.random.Generator. You may import only numpy and math. No other imports,
file/network/process access, simulator access, data loading, eval/exec or global mutable state.
All policy state must be reconstructed from history. Keep the code below 180 lines and one
call under 0.2 seconds. Generate continuous placements; do not merely rank a fixed list.
You can define helper functions and implement adaptive loops/branches or fit small models.
Inspect geometry and experiment outcomes to reason about mechanisms rather than just retrying
the same failed placement. The supplied evidence, if any, is from source scenes only; new
scene geometries and outcomes are unknown. Infer useful general structure, not seed IDs.
The two task families involve a lever launching a green ball toward a distant basket and a
ramp/top-basket interaction dropping a green ball toward a lower basket. Do not assume every
scene requires one specific strategy: blue objects and baskets can move too.
Ordinary random object-centered sampling, CMA-ES and GP Bayesian optimization are comparators.
Linear event secants alone are just local regression; no new-method claim is requested from
you. Produce your strongest concrete executable strategy under the stated interface.
'''


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    source = Path('runs/interphyre_continuous_search_v1/native_traces.jsonl')
    grouped = defaultdict(list)
    for line in source.read_text().splitlines():
        row = json.loads(line)
        grouped[row['level'], row['seed']].append(row)
    worlds = []
    for (level, seed), rows in sorted(grouped.items()):
        env = InterphyreEnv(level, seed=seed)
        env.reset()
        scene = env.describe_scene()
        env.close()
        unique = {tuple(r['action']): r for r in rows}
        ranked = sorted(unique.values(), key=lambda r: r['score'], reverse=True)
        selected = ranked[:6] + ranked[-2:]
        contacts = {}
        for row in ranked:
            contacts.setdefault(row['first_action_contact'], row)
        selected = list({tuple(r['action']): r for r in selected + list(contacts.values())}.values())
        def compact(row):
            keys = ('action','invalid','success','score','min_gap','max_contact_duration',
                    'first_action_contact','closest_error')
            return {k: row[k] for k in keys if k in row}
        worlds.append(dict(source_id=len(worlds), level=level, source_seed=seed,
                           scene=scene, selected_observations=[compact(r) for r in selected]))
    packet = dict(source_path=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  selection='six highest-score unique actions, two lowest, and the best for each observed first-contact label per source scene',
                  worlds=worlds)
    (out/'source_packet.json').write_text(json.dumps(packet,indent=2)+'\n')
    (out/'no_experience_prompt.txt').write_text(INSTRUCTIONS+'\nNo source experiment observations are supplied.\n')
    source_prompt = INSTRUCTIONS+'\nSOURCE EXPERIMENT EVIDENCE:\n'+json.dumps(worlds,separators=(',',':'))+'\n'
    (out/'experience_prompt.txt').write_text(source_prompt)
    plan = dict(source_seeds=[300,301,302,303], target_seeds=[400,401,402,403],
                levels=['catapult','pass_the_parcel'], queries=25,
                arms=['generated_with_experience','generated_without_experience','source_gp_bo','target_gp_bo'],
                authoring_model='deepseek-v4-pro', authoring_requests_planned=2,
                candidate_count_per_condition=1, target_feedback_before_freeze=False,
                source_packet=str(out/'source_packet.json'), source_worlds=len(worlds),
                selected_source_observations=sum(len(w['selected_observations']) for w in worlds),
                full_source_placement_queries=565, gpu_work=False,
                scope='Development test of executable policy generation, not a claim that program generation itself is novel, nor an ICLR-scale evaluation.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'run':str(out),'source_worlds':len(worlds),'selected_source_observations':plan['selected_source_observations'],'prompt_chars':len(source_prompt)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
