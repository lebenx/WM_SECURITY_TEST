"""Execute native task continuations with frozen native WorldMind retrieval.

This is the plugin's documented memory-injection interface hosted in the
original WMReactAgent, not a reproduction of WorldMind's embodied benchmark.
The source process experiences came from fixed genuine host prediction errors.
No world-model foresight or online WorldMind learning is used in this screen.
"""
import argparse
import json
from pathlib import Path
from scienceworld import ScienceWorldEnv
from wm_security.client import Client
from wm_security.upstream import LoggedLLM
from agents.wm_react_agent import WMReactAgent


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save = lambda name, value: (root/name).write_text(json.dumps(value, indent=2))
    save('config.json', vars(args))
    client = Client(args.model, root, max_calls=60)
    data = json.loads(Path(args.data).read_text())
    prompt = Path(__file__).resolve().parents[1]/'vendor/WorldEvolver/src/prompts/ReactAgent/scienceworld_react.json'
    summary = []
    for condition in ['clean', 'proper', 'bypass']:
        memory = json.loads((Path(args.source)/f'{condition}_retrieval.json').read_text())
        env = ScienceWorldEnv('', envStepLimit=100)
        try:
            env.load(data['task'], data['variation'], 'easy')
            obs, info = env.reset()
            llm = LoggedLLM(client, f'actor/{condition}')
            actor = WMReactAgent(llm, need_goal=True, init_prompt_path=str(prompt))
            actor.instruction = actor.instruction.replace(
                '`wait [DURATION]`: Pause for a specified period.',
                '`wait`: Wait for 10 environment iterations. `wait1`: Wait for 1 iteration. '
                'These are the exact command strings; no numeric argument is accepted.')
            actor.instruction = actor.instruction.replace('- `[DURATION]`: Specified time\n', '')
            if memory['formatted_prompt']:
                actor.instruction += '\n\n'+memory['formatted_prompt']
            actor.reset(env.get_task_description(), obs)
            save(f'{condition}_prompt.json', dict(instruction=actor.instruction, retrieval=memory))
            setup = []
            for action in [r['action'] for r in data['records'][:12]]+['activate switch']:
                obs, _, done, info = env.step(action)
                if done or 'No known action matches' in obs:
                    raise RuntimeError(f'Invalid shared prefix: {action}: {obs}')
                actor.update(action, obs)
                setup.append(dict(action=action, observation=obs, score=info['score']))
            events = []
            for step in range(args.steps):
                llm.tag = f'{step}/decision'
                ok, action = actor.run()
                if not ok:
                    raise RuntimeError('Actor generation failed')
                obs, _, done, info = env.step(action)
                actor.update(action, obs)
                events.append(dict(action=action, observation=obs, score=info['score'], done=done))
                save(f'{condition}_events.json', dict(setup=setup, events=events))
                print(json.dumps(dict(condition=condition, action=action, score=info['score'])), flush=True)
                if done:
                    break
            result = dict(condition=condition, success=info['score']==100, score=info['score'],
                          actions=[e['action'] for e in events], done=done)
            summary.append(result)
            if condition in ['clean', 'proper'] and not result['success']:
                save('status.json', dict(status='control_gate_failed', results=summary,
                     interpretation='Frozen WorldMind retrieval control did not complete; attack arm not run.'))
                return
        except Exception as exc:
            save('status.json', dict(status='interrupted', condition=condition, results=summary, error=str(exc)))
            raise
        finally:
            env.close()
    save('status.json', dict(status='completed', results=summary,
         interpretation='Frozen original WorldMind process-retrieval/refinement output injected into native WMReactAgent instructions. Source forecasts are fixed outputs of the earlier WorldEvolver host. No online victim learning or native WorldMind benchmark claim; one physical instance.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--source', default='runs/worldmind_conductivity_replay_local_v0')
    parser.add_argument('--data', default='runs/native_core_records_v0/test-conductivity-of-unknown-substances.json')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
