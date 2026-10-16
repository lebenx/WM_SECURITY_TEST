"""Can real successful measurement episodes repair shared predictions?

Fresh native actors receive observations but no world-model forecasts. This is
an explicit recovery intervention, not a claim about spontaneous verification
by forecast-assisted actors. Every executed decision is chosen by the actor;
the world model continues its original predict/update loop in the background.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from scienceworld import ScienceWorldEnv
from wm_security.local import LocalClient
from wm_security.upstream import LoggedLLM, text_world_model
from agents.wm_react_agent import WMReactAgent


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    save = lambda name, obj: (root/name).write_text(json.dumps(obj, indent=2))
    save('config.json', vars(args))
    data = json.loads(Path(args.data).read_text())
    spec = json.loads(Path(args.queries).read_text())
    client = LocalClient(args.model, root, max_calls=220)
    prompt = Path(__file__).resolve().parents[1]/'vendor/WorldEvolver/src/prompts/ReactAgent/scienceworld_react.json'
    summary = []

    def evaluate(wm, label, completed_episodes):
        result = []
        for query in spec['queries']:
            wm.llm.tag = f'probe/{completed_episodes}/{query["id"]}'
            prediction = wm.predict(query['state'], query['action'], query['goal'])
            row = dict(condition=label, completed_episodes=completed_episodes, view=query['id'],
                       prediction=prediction, actual=query['actual'])
            result.append(row)
            print(json.dumps(dict(condition=label, episodes=completed_episodes, view=query['id'],
                                  prediction=prediction['prediction'])), flush=True)
        return result

    for condition, checkpoint_name in [('proper', 'proper_memory.json'), ('bypass', 'bypass_memory.json')]:
        model = text_world_model(client, f'world/{condition}')
        model.load_checkpoint_state(json.loads((Path(args.source)/checkpoint_name).read_text()))
        probes = evaluate(model, condition, 0)
        episodes = []
        save(f'{condition}_probes.json', probes)
        for episode in range(args.episodes):
            env = ScienceWorldEnv('', envStepLimit=100)
            events = []
            try:
                env.load(data['task'], data['variation'], 'easy')
                obs, info = env.reset()
                goal = env.get_task_description()
                actor_llm = LoggedLLM(client, f'actor/{condition}/{episode}')
                actor = WMReactAgent(actor_llm, need_goal=True, init_prompt_path=str(prompt))
                actor.instruction = actor.instruction.replace(
                    '`wait [DURATION]`: Pause for a specified period.',
                    '`wait`: Wait for 10 environment iterations. `wait1`: Wait for 1 iteration. '
                    'These are the exact command strings; no numeric argument is accepted.')
                actor.instruction = actor.instruction.replace('- `[DURATION]`: Specified time\n', '')
                actor.reset(goal, obs)
                setup = []
                prefix = [r['action'] for r in data['records'][:12]] + ['activate switch']
                for action in prefix:
                    obs, _, done, info = env.step(action)
                    if done or 'No known action matches' in obs:
                        raise RuntimeError(f'Invalid prepared state: {action}: {obs}')
                    actor.update(action, obs)
                    setup.append(dict(action=action, observation=obs, score=info['score']))
                for step in range(args.steps):
                    actor_llm.tag = f'{step}/decision'
                    ok, action = actor.run()
                    if not ok:
                        raise RuntimeError('Recovery actor failed to generate an action')
                    state = obs.strip()+'\n'+env.inventory()
                    model.llm.tag = f'recovery/{episode}/{step}/forecast'
                    prediction = model.predict(state, action, goal)
                    obs, _, done, info = env.step(action)
                    actor.update(action, obs)
                    event = dict(step=step, state=state, action=action, observation=obs,
                                 prediction=prediction, score=info['score'], done=done, update_completed=False)
                    events.append(event)
                    save(f'{condition}_episode_{episode}.json', dict(setup=setup, events=events))
                    model.llm.tag = f'recovery/{episode}/{step}/update'
                    model.update(state, action, prediction['prediction'], obs, {'task':goal})
                    event['update_completed'] = True
                    save(f'{condition}_episode_{episode}.json', dict(setup=setup, events=events))
                    save(f'{condition}_memory_{episode+1}.json', model.checkpoint_state())
                    if done:
                        break
                result = dict(episode=episode, score=info['score'], success=info['score']==100,
                              actions=[e['action'] for e in events])
                episodes.append(result)
                save(f'{condition}_episodes.json', episodes)
                print(json.dumps(dict(condition=condition, **result)), flush=True)
                if not result['success']:
                    save('status.json', dict(status='recovery_actor_gate_failed', condition=condition,
                         episode=episode, completed_conditions=summary,
                         interpretation='The unassisted actor did not complete a corrective task; no claim that recovery failed despite successful users.'))
                    return
            except Exception as exc:
                save('status.json', dict(status='interrupted', condition=condition, episode=episode, error=str(exc)))
                raise
            finally:
                env.close()
            if episode+1 in [1, 3, args.episodes]:
                probes.extend(evaluate(model, condition, episode+1))
                save(f'{condition}_probes.json', probes)

        # A researcher-only retrieval intervention after all real updates.
        recent = text_world_model(client, f'recency/{condition}')
        recent.load_checkpoint_state(deepcopy(model.checkpoint_state()))
        original_rank = recent.library._rank_by_action

        def newest_ties(action, k):
            matches = original_rank(action, len(recent.library.records))
            return sorted(matches, key=lambda item: (item[1], item[0]), reverse=True)[:k]

        recent.library._rank_by_action = newest_ties
        save(f'{condition}_recency_probes.json', evaluate(recent, condition+'_recent', args.episodes))
        summary.append(dict(condition=condition, episodes=episodes))
    save('status.json', dict(status='completed', results=summary,
         interpretation='Fresh unassisted native actors perform real corrective task continuations while the shared model learns every executed continuation decision. Prepared wiring prefixes are observed by actors but not newly learned. This tests recovery under supplied successful experience, not spontaneous verification in the attacked policy. One physical instance.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--source', default='runs/conductivity_complete_learning_local_v0')
    parser.add_argument('--data', default='runs/native_core_records_v0/test-conductivity-of-unknown-substances.json')
    parser.add_argument('--queries', default='configs/conductivity_complete_queries.json')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
