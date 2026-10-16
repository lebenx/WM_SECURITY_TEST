"""One source-frozen policy authoring request; generated code is reviewed separately."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path

from wm_security.client import Client


def main(args):
    root = Path(args.run)
    condition = args.condition
    out = root / condition
    out.mkdir(exist_ok=False)
    filename = 'experience_prompt.txt' if condition == 'with_experience' else 'no_experience_prompt.txt'
    prompt = (root / filename).read_text()
    (out / 'prompt.txt').write_text(prompt)
    client = Client('deepseek-v4-pro', log_dir=out, timeout=90, max_calls=1,
                    retry_truncated=True)
    client.config = dict(client.config, base_url=client.config['fallback_url'],
                         api_key=client.config['fallback_api_key'])
    status = dict(condition=condition, model=client.model, endpoint=client.config['base_url'],
                  primary_called=False, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  target_feedback_used=False)
    try:
        response = client.completion([{'role': 'user', 'content': prompt}], max_tokens=8192,
                                     tag='interphyre_policy_' + condition)
        answer = response['choices'][0]['message']['content']
        (out / 'answer.txt').write_text(answer)
        code = answer.strip()
        if code.startswith('```'):
            code = code.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        ast.parse(code)
        (out / 'policy.py').write_text(code + '\n')
        status.update(status='authored_pending_review', code_sha256=hashlib.sha256((code+'\n').encode()).hexdigest(),
                      code_lines=len(code.splitlines()), finish_reason=response['choices'][0]['finish_reason'],
                      usage=response.get('usage'))
    except Exception as exc:
        error = str(exc)
        for field in ('api_key', 'fallback_api_key'):
            if client.config.get(field):
                error = error.replace(client.config[field], '[REDACTED]')
        status.update(status='failed', error=error)
    status.update(requests=client.calls, finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--condition', choices=['with_experience', 'without_experience'], required=True)
    main(p.parse_args())
