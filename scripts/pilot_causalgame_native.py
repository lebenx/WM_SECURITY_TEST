"""CausalGame's native hybrid runner through the logged research gateway.

Only provider transport and output locations are adapted. Native prompts,
history visibility, tools, experiment budgets, exploration loop and scoring
remain upstream. This is a capability test, not an attack experiment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor' / 'CausalGame'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(VENDOR))


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def serve(args):
    import numpy as np
    import uvicorn
    from api.modules.agent.session import SessionManager

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CAUSALGAME_EXPERIMENT'] = args.experiment
    os.environ['ADMIN_TOKEN'] = secrets.token_urlsafe(32)
    records = args.output / 'backend_records'
    records.mkdir(parents=True, exist_ok=True)
    SessionManager.PERSISTENCE_DIR = str(records)
    # Import after selecting the experiment and the session storage location.
    from api.app import app
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)


@contextmanager
def backend(args):
    import requests

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [sys.executable, str(Path(__file__).resolve()), '--serve',
               '--output', str(args.output), '--experiment', args.experiment,
               '--seed', str(args.seed), '--port', str(port)]
    with (args.output / 'backend.log').open('w') as log:
        process = subprocess.Popen(command, cwd=args.output, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Native backend stopped; inspect backend.log')
                try:
                    response = requests.get(url + '/api/health', timeout=1)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError('Native backend did not become ready in 30 seconds')
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


class LoggedCompletions:
    """SDK shape adapter; the upstream handler still manages tool-call history."""

    def __init__(self, client):
        self.client = client

    def create(self, *, model, messages, tools=None, tool_choice='auto'):
        from openai.types.chat import ChatCompletion

        if model != self.client.model:
            raise ValueError('Native handler and logged transport model differ')
        result = self.client.completion(
            messages, tools=tools, tool_choice=tool_choice,
            tag=f'native_hybrid_{self.client.calls + 1}', max_tokens=4096)
        return ChatCompletion.model_validate(result)


def run(args):
    from wm_security.client import Client

    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        'purpose': 'clean native capability gate; no attacker or new mechanism claimed',
        'model': args.model, 'backend_identity_verified': False,
        'experiment': args.experiment, 'seed': args.seed,
        'commit': subprocess.check_output(
            ['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip(),
        'mode': 'hybrid', 'native_runner': 'run_agent.run_agent_session',
        'runner_max_turns_argument': 100,
        'turn_note': 'Upstream maps default 100 to deployment budget + 10 (20 turns).',
        'max_api_requests': args.max_calls, 'api_timeout_s': args.api_timeout,
        'generation': {'temperature': 0, 'max_tokens': 4096, 'thinking': False,
                       'retry_truncated': True},
        'seed_scope': 'Python random and NumPy in a fresh single-session backend; UUIDs vary.',
        'adaptations': ['logged compatible transport', 'model registration',
                        'isolated backend/session/workspace paths',
                        'neutral temporary workspace path hides experiment identifiers',
                        'localhost bypasses the external HTTP proxy'],
        'collect_only': args.collect_only,
        'packages': {name: importlib.metadata.version(name) for name in
                     ['openai', 'requests', 'numpy', 'pandas', 'fastapi', 'pydantic']},
    }
    save(args.output / 'plan.json', plan)
    old_cwd = Path.cwd()
    # Upstream includes the absolute workspace path in the actor prompt. Run
    # under a neutral name so a descriptive output directory cannot leak the
    # experiment's hidden mechanism. Archive the workspace only after the run.
    work_root = Path(tempfile.mkdtemp(prefix='mission-'))
    os.chdir(work_root)
    os.environ['CAUSALGAME_EXPERIMENT'] = args.experiment
    no_proxy = ','.join(filter(None, [os.environ.get('NO_PROXY'),
                                     os.environ.get('no_proxy'), '127.0.0.1', 'localhost']))
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = no_proxy
    started = time.monotonic()
    transport = None
    handlers = []
    try:
        with backend(args) as url:
            from agent.client import CanyonClient
            if args.collect_only:
                client = CanyonClient(url, model_name='offline-smoke',
                                      experiment=args.experiment, execution_mode='hybrid')
                before = client.get_status()
                space = client.get_action_space()
                history = client.get_history()
                design = {k: v['default'] for k, v in space['action_space']['numerical'].items()}
                deployment = client.deploy_drone(design, count=1)
                result = {'status': 'offline_tool_smoke', 'api_requests': 0,
                          'before': before, 'action_space': space, 'history': history,
                          'deployment': deployment, 'after': client.get_status()}
            else:
                import run_agent as native
                from agent.openai import OpenAIHandler

                transport = Client(args.model, args.output, timeout=args.api_timeout,
                                   max_calls=args.max_calls, retry_truncated=True)
                native.get_model_config = lambda model: {'provider': 'logged-compatible'}

                def make_handler(model_name, system_instruction, enable_thinking=True):
                    handler = OpenAIHandler(
                        api_key='transport-adapter-placeholder', model_name=model_name,
                        base_url=url, system_instruction=system_instruction,
                        enable_thinking=False)
                    handler.client.close()
                    handler.client = SimpleNamespace(chat=SimpleNamespace(
                        completions=LoggedCompletions(transport)))
                    handlers.append(handler)
                    return handler

                native.create_agent_handler = make_handler
                result = native.run_agent_session(
                    model_name=args.model, experiment=args.experiment,
                    base_url=url, enable_thinking=False, verbose=True,
                    execution_mode='hybrid')
                result['api_requests'] = transport.calls
                if handlers:
                    handler = handlers[0]
                    save(args.output / 'conversation.json', handler.history)
                    client = handler.locals['client']
                    result['final_status'] = client.get_status()
                    # Upstream reports usage sampled at the start of its outer
                    # turn, before potentially several nested deployments.
                    result['native_runner_reported_drones_used'] = result['drones_used']
                    result['drones_used'] = result['final_status']['drones_used']
                    result['task_success'] = result['victory']
                    save(args.output / 'public_history.json', client.get_history())
            result['elapsed_s'] = time.monotonic() - started
            save(args.output / 'summary.json', result)
            print(json.dumps({k: v for k, v in result.items()
                              if k not in {'final_status', 'before', 'after', 'history',
                                           'action_space', 'deployment'}}, ensure_ascii=False), flush=True)
    except Exception as exc:
        error = str(exc)
        if transport:
            error = error.replace(transport.config['api_key'], '[REDACTED]')
        save(args.output / 'summary.json', {'status': 'interrupted', 'error': error,
             'api_requests': transport.calls if transport else 0,
             'elapsed_s': time.monotonic() - started})
        if handlers:
            save(args.output / 'conversation.json', handlers[0].history)
        print(json.dumps({'status': 'interrupted', 'error': error}), flush=True)
        return 1
    finally:
        os.chdir(old_cwd)
        workspace = work_root / 'agent_workspaces'
        if workspace.exists():
            shutil.copytree(workspace, args.output / 'agent_workspaces', dirs_exist_ok=True)
        shutil.rmtree(work_root)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--experiment', default='antenna_trap')
    parser.add_argument('--model', default='glm-5.3')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--max-calls', type=int, default=80)
    parser.add_argument('--api-timeout', type=float, default=180)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--port', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.serve:
        serve(args)
    else:
        raise SystemExit(run(args))
