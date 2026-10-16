"""Small OpenAI-compatible client; credentials never enter experiment records."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid
import requests


logger = logging.getLogger(__name__)


class ProviderContentFilterError(RuntimeError):
    """Provider declined generation; do not retry through another endpoint."""


def parse_json(content: str):
    text = content.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for pos, char in enumerate(text):
            if char in '{[':
                try:
                    value, _ = json.JSONDecoder().raw_decode(text[pos:])
                    return value
                except json.JSONDecodeError:
                    pass
        raise ValueError('No valid JSON in model response')


class Client:
    def __init__(self, model='gemini-3.8-flash', log_dir='runs', timeout=90, max_calls=500,
                 thinking=False, reasoning_effort='high', retry_truncated=False):
        profile = 'gemini.json' if model.startswith('gemini-') else 'api.json'
        cfg_path = Path(os.environ.get('WM_API_CONFIG',
                                      str(Path.home()/'.config/wm-security'/profile)))
        self.config = json.loads(cfg_path.read_text())
        self.model = model
        self.timeout = timeout
        self.max_calls = max_calls
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.retry_truncated = retry_truncated
        self.calls = 0
        self.lock = threading.Lock()
        self.path = Path(log_dir) / 'calls.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record):
        with self.lock:
            with self.path.open('a') as out:
                out.write(json.dumps(record, ensure_ascii=False) + '\n')

    def chat(self, messages, max_tokens=1536, temperature=0, tag='', json_mode=False):
        result = self.completion(messages, max_tokens=max_tokens, temperature=temperature,
                                 tag=tag, json_mode=json_mode)
        return result['choices'][0]['message']['content']

    def completion(self, messages, max_tokens=1536, temperature=0, tag='',
                   json_mode=False, tools=None, tool_choice='auto'):
        """Return the full response for native tool-call handlers.

        The text-only chat interface keeps its existing validation. A tool-only
        assistant message is valid only when this request supplied tools.
        """
        if max_tokens <= 0:
            raise ValueError('max_tokens must be positive')
        # Keep the caller's output budget for ordinary requests.  Some
        # OpenAI-compatible gateways delay or reject a request when a tiny
        # probe is advertised with a multi-thousand-token budget.  Thinking
        # requests still receive the larger allowance below because hidden
        # reasoning consumes part of that budget.
        body = {'model': self.model, 'messages': messages, 'temperature': temperature,
                'max_tokens': int(max_tokens), 'stream': False,
                'thinking': {'type': 'disabled'}, 'enable_thinking': False}
        if self.thinking:
            body.update(thinking={'type': 'enabled'}, enable_thinking=True,
                        reasoning_effort=self.reasoning_effort,
                        max_tokens=max(8192, int(max_tokens)))
        if self.config.get('provider') == 'google_gemini':
            # Gemini's official compatibility endpoint uses reasoning_effort;
            # the old gateway-specific flags are not part of this API.
            body.pop('thinking', None)
            body.pop('enable_thinking', None)
            body['reasoning_effort'] = self.reasoning_effort if self.thinking else 'low'
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        # A verified endpoint may be marked preferred when the nominal primary
        # gateway is intermittent or returns reasoning-only messages.  Keep
        # the nominal primary and configured fallback in the retry chain, but
        # never duplicate the same URL/key pair.
        endpoints = []
        seen_endpoints = set()

        def add_endpoint(url, key):
            if not url or not key or (url, key) in seen_endpoints:
                return
            seen_endpoints.add((url, key))
            endpoints.append((url, key))

        add_endpoint(self.config.get('preferred_url'),
                     self.config.get('preferred_api_key'))
        add_endpoint(self.config.get('base_url'), self.config.get('api_key'))
        add_endpoint(self.config.get('fallback_url'),
                     self.config.get('fallback_api_key') or self.config.get('api_key'))
        if not endpoints:
            raise RuntimeError('No API endpoint configured')
        errors = []
        length_retries = set()
        for attempt, (endpoint, api_key) in enumerate(endpoints):
            with self.lock:
                if self.calls >= self.max_calls:
                    raise RuntimeError('Session API call ceiling reached')
                self.calls += 1
            record = {'id': str(uuid.uuid4()), 'tag': tag, 'time': time.time(),
                      'endpoint': endpoint, 'request': body}
            # A larger reasoning/output allowance also needs enough time to
            # finish. The retry remains bounded to the same two attempts.
            request_timeout = self.timeout * max(1, body['max_tokens'] / 4096)
            record['timeout_s'] = request_timeout
            start = time.monotonic()
            try:
                r = requests.post(endpoint + '/chat/completions',
                    headers={'Authorization': 'Bearer ' + api_key},
                    json=body, timeout=request_timeout)
                record['http_status'] = r.status_code
                if r.status_code != 200:
                    raise RuntimeError('HTTP ' + str(r.status_code) + ': ' + r.text[:500])
                result = r.json()
                record['response'] = result
                choices = result.get('choices') if isinstance(result, dict) else None
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise RuntimeError('Response contained no valid completion choice')
                choice = choices[0]
                finish_reason = choice.get('finish_reason')
                if isinstance(finish_reason, str) and finish_reason.startswith('content_filter'):
                    raise ProviderContentFilterError(
                        'Response blocked by provider: finish_reason=' + finish_reason)
                message = choice.get('message')
                if not isinstance(message, dict):
                    raise RuntimeError('Response contained no assistant message; finish_reason='
                                       + str(finish_reason))
                content = message.get('content')
                if self.retry_truncated and result['choices'][0].get('finish_reason') == 'length':
                    raise RuntimeError('Response truncated at output token limit')
                if not content and not (tools and message.get('tool_calls')):
                    raise RuntimeError('Response contained no visible content')
                record['latency_s'] = time.monotonic() - start
                self.log(record)
                if errors:
                    logger.warning(
                        'API retry succeeded: model=%s endpoint=%s tag=%s run=%s',
                        self.model, endpoint, tag, self.path.parent)
                return result
            except Exception as exc:
                error = str(exc)
                for key_name in ['api_key', 'fallback_api_key', 'backup_api_key',
                                 'preferred_api_key']:
                    secret = self.config.get(key_name)
                    if secret:
                        error = error.replace(secret, '[REDACTED]')
                record['error'] = error
                record['latency_s'] = time.monotonic() - start
                self.log(record)
                errors.append(error)
                # Some compatible gateways count hidden reasoning against the
                # output limit even when thinking is disabled. Preserve the
                # failed request, then give this endpoint one bounded retry
                # with more room. Insert it before the next endpoint instead
                # of overwriting the configured fallback.
                response = record.get('response')
                choices = response.get('choices') if isinstance(response, dict) else None
                first_choice = (choices[0] if isinstance(choices, list) and choices
                                and isinstance(choices[0], dict) else {})
                endpoint_key = (endpoint, api_key)
                if (first_choice.get('finish_reason') == 'length'
                        and endpoint_key not in length_retries
                        and body['max_tokens'] < 16384
                        and self.calls < self.max_calls):
                    length_retries.add(endpoint_key)
                    body = dict(body, max_tokens=min(16384, body['max_tokens'] * 2))
                    # A token-limit failure is not an endpoint outage. Retry
                    # on the same endpoint so the budget change is isolated;
                    # the original next endpoint remains after that retry.
                    endpoints.insert(attempt + 1, endpoint_key)
                blocked = isinstance(exc, ProviderContentFilterError)
                next_endpoint = (endpoints[attempt + 1][0]
                                 if not blocked and attempt + 1 < len(endpoints) and self.calls < self.max_calls
                                 else None)
                logger.warning(
                    'API request failed: model=%s endpoint=%s tag=%s run=%s error=%s; %s',
                    self.model, endpoint, tag, self.path.parent, error,
                    ('retrying ' + next_endpoint) if next_endpoint else
                    ('provider content filter; not retrying' if blocked else 'no retries remain'))
                if blocked:
                    raise ProviderContentFilterError(error) from None
        raise RuntimeError('; '.join(errors))

    def json(self, system, payload, tag='', max_tokens=1536):
        response = self.chat([{'role':'system', 'content':system},
                              {'role':'user', 'content':json.dumps(payload, ensure_ascii=False)}],
                              max_tokens=max_tokens, tag=tag)
        return parse_json(response)
