import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from wm_security.client import Client, ProviderContentFilterError


def response(content='ok', finish_reason='stop'):
    return Mock(status_code=200, json=lambda: {
        'choices': [{'message': {'content': content}, 'finish_reason': finish_reason}]})


class EndpointCredentialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            'base_url': 'https://primary.invalid/v1',
            'fallback_url': 'https://fallback.invalid/v1',
            'api_key': 'test-primary-secret',
            'fallback_api_key': 'test-fallback-secret',
        }
        self.config_path = self.root / 'api.json'
        self.write_config()
        env = patch.dict(os.environ, {'WM_API_CONFIG': str(self.config_path)})
        env.start()
        self.addCleanup(env.stop)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def client(self, **kwargs):
        return Client(model='same-model', log_dir=self.root / 'run', **kwargs)

    @patch('wm_security.client.requests.post')
    def test_fallback_uses_own_key_and_keeps_model(self, post):
        post.side_effect = [requests.Timeout('timed out'), response()]
        with self.assertLogs('wm_security.client', level='WARNING') as logs:
            self.assertEqual(self.client().chat([{'role': 'user', 'content': 'probe'}]), 'ok')
        first, second = post.call_args_list
        self.assertEqual(first.args[0], self.config['base_url'] + '/chat/completions')
        self.assertEqual(second.args[0], self.config['fallback_url'] + '/chat/completions')
        self.assertEqual(first.kwargs['headers']['Authorization'], 'Bearer test-primary-secret')
        self.assertEqual(second.kwargs['headers']['Authorization'], 'Bearer test-fallback-secret')
        self.assertEqual(first.kwargs['json'], second.kwargs['json'])
        self.assertIn('API request failed', logs.output[0])
        self.assertIn('API retry succeeded', logs.output[1])

    @patch('wm_security.client.requests.post')
    def test_preferred_endpoint_uses_its_own_key_before_primary(self, post):
        self.config.update(
            preferred_url='https://verified.invalid/v1',
            preferred_api_key='test-preferred-secret',
        )
        self.write_config()
        post.return_value = response()
        self.assertEqual(self.client().chat([{'role': 'user', 'content': 'probe'}]), 'ok')
        call = post.call_args
        self.assertEqual(call.args[0], self.config['preferred_url'] + '/chat/completions')
        self.assertEqual(call.kwargs['headers']['Authorization'],
                         'Bearer ' + self.config['preferred_api_key'])

    @patch('wm_security.client.requests.post')
    def test_preferred_failure_is_logged_before_primary_retry(self, post):
        self.config.update(preferred_url='https://verified.invalid/v1',
                           preferred_api_key='test-preferred-secret')
        self.write_config()
        post.side_effect = [requests.Timeout('preferred unavailable'), response()]
        with self.assertLogs('wm_security.client', level='WARNING') as logs:
            self.assertEqual(self.client().chat([{'role': 'user', 'content': 'probe'}]), 'ok')
        self.assertEqual(post.call_args_list[1].args[0],
                         self.config['base_url'] + '/chat/completions')
        self.assertIn('preferred unavailable', logs.output[0])
        self.assertIn('API retry succeeded', logs.output[1])

    @patch('wm_security.client.requests.post')
    def test_output_limit_retry_keeps_primary_key(self, post):
        post.side_effect = [response('partial', 'length'), response()]
        with self.assertLogs('wm_security.client', level='WARNING'):
            self.client(retry_truncated=True).chat([{'role': 'user', 'content': 'probe'}])
        first, second = post.call_args_list
        self.assertEqual(first.args[0], second.args[0])
        self.assertEqual(first.kwargs['headers'], second.kwargs['headers'])
        self.assertEqual(second.kwargs['headers']['Authorization'], 'Bearer test-primary-secret')
        self.assertEqual(second.kwargs['json']['max_tokens'], 3072)

    @patch('wm_security.client.requests.post')
    def test_small_probe_preserves_requested_output_budget(self, post):
        post.return_value = response()
        self.client().chat([{'role': 'user', 'content': 'probe'}], max_tokens=30)
        self.assertEqual(post.call_args.kwargs['json']['max_tokens'], 30)

    @patch('wm_security.client.requests.post')
    def test_output_limit_retry_preserves_fallback_after_primary_retry(self, post):
        post.side_effect = [response('partial', 'length'),
                            response('partial', 'length'), response()]
        with self.assertLogs('wm_security.client', level='WARNING'):
            self.client(retry_truncated=True).chat([{'role': 'user', 'content': 'probe'}])
        first, retry, fallback = post.call_args_list
        self.assertEqual(first.args[0], retry.args[0])
        self.assertEqual(retry.args[0], self.config['base_url'] + '/chat/completions')
        self.assertEqual(fallback.args[0], self.config['fallback_url'] + '/chat/completions')
        self.assertEqual(fallback.kwargs['headers']['Authorization'],
                         'Bearer test-fallback-secret')

    @patch('wm_security.client.requests.post')
    def test_both_keys_redacted_from_errors_and_alerts(self, post):
        message = 'failed test-primary-secret test-fallback-secret'
        post.side_effect = [requests.Timeout(message), requests.Timeout(message)]
        with self.assertLogs('wm_security.client', level='WARNING') as logs:
            with self.assertRaises(RuntimeError) as error:
                self.client().chat([{'role': 'user', 'content': 'probe'}])
        visible = str(error.exception) + '\n'.join(logs.output)
        visible += (self.root / 'run' / 'calls.jsonl').read_text()
        for key in ['api_key', 'fallback_api_key']:
            self.assertNotIn(self.config[key], visible)

    @patch('wm_security.client.requests.post')
    def test_preferred_key_is_redacted_from_errors_and_alerts(self, post):
        self.config.update(preferred_url='https://verified.invalid/v1',
                           preferred_api_key='test-preferred-secret')
        self.write_config()
        message = 'failed test-preferred-secret'
        post.side_effect = [requests.Timeout(message), requests.Timeout(message),
                             requests.Timeout(message)]
        with self.assertLogs('wm_security.client', level='WARNING') as logs:
            with self.assertRaises(RuntimeError) as error:
                self.client().chat([{'role': 'user', 'content': 'probe'}])
        visible = str(error.exception) + '\n'.join(logs.output)
        visible += (self.root / 'run' / 'calls.jsonl').read_text()
        self.assertNotIn(self.config['preferred_api_key'], visible)

    @patch('wm_security.client.requests.post')
    def test_existing_single_key_configuration_still_works(self, post):
        self.config.pop('fallback_api_key')
        self.write_config()
        post.side_effect = [requests.Timeout('timed out'), response()]
        with self.assertLogs('wm_security.client', level='WARNING'):
            self.client().chat([{'role': 'user', 'content': 'probe'}])
        self.assertEqual(post.call_args_list[1].kwargs['headers']['Authorization'],
                         'Bearer test-primary-secret')

    @patch('wm_security.client.requests.post')
    def test_gemini_uses_supported_fields_and_preserves_json_mode(self, post):
        self.config.update(provider='google_gemini')
        self.config.pop('fallback_url')
        self.write_config()
        post.return_value = response('{"answer":43}')
        self.client().chat([{'role': 'user', 'content': 'probe'}], json_mode=True)
        body = post.call_args.kwargs['json']
        self.assertNotIn('thinking', body)
        self.assertNotIn('enable_thinking', body)
        self.assertEqual(body['reasoning_effort'], 'low')
        self.assertEqual(body['response_format'], {'type': 'json_object'})

    @patch('wm_security.client.requests.post')
    def test_gemini_failure_does_not_call_another_provider(self, post):
        self.config.update(provider='google_gemini')
        self.config.pop('fallback_url')
        self.write_config()
        post.side_effect = requests.Timeout('failed test-primary-secret')
        with self.assertLogs('wm_security.client', level='WARNING') as logs:
            with self.assertRaises(RuntimeError):
                self.client().chat([{'role': 'user', 'content': 'probe'}])
        self.assertEqual(post.call_count, 1)
        self.assertIn('no retries remain', logs.output[0])
        self.assertNotIn('test-primary-secret', logs.output[0])

    @patch('wm_security.client.requests.post')
    def test_call_ceiling_alert_does_not_promise_a_retry(self, post):
        post.side_effect = requests.Timeout('timed out')
        with self.assertLogs('wm_security.client', level='WARNING') as logs:
            with self.assertRaises(RuntimeError):
                self.client(max_calls=1).chat([{'role': 'user', 'content': 'probe'}])
        self.assertEqual(post.call_count, 1)
        self.assertNotIn('retrying ', logs.output[0])

    @patch('wm_security.client.requests.post')
    def test_gemini_model_loads_its_own_profile(self, post):
        profile = self.root / '.config/wm-security/gemini.json'
        profile.parent.mkdir(parents=True)
        gemini = {'base_url': 'https://gemini.invalid/v1beta/openai',
                  'api_key': 'test-gemini-secret', 'provider': 'google_gemini'}
        profile.write_text(json.dumps(gemini))
        post.return_value = response()
        with patch.dict(os.environ, {}, clear=True), patch('wm_security.client.Path.home', return_value=self.root):
            c = Client(model='gemini-3.8-flash', log_dir=self.root/'gemini_run')
            c.chat([{'role':'user','content':'probe'}])
        self.assertEqual(post.call_args.args[0], gemini['base_url']+'/chat/completions')
        self.assertEqual(post.call_args.kwargs['headers']['Authorization'], 'Bearer test-gemini-secret')

    @patch('wm_security.client.requests.post')
    def test_provider_filter_without_message_is_clear_and_never_retried(self, post):
        post.return_value = Mock(status_code=200, json=lambda: {
            'choices': [{'finish_reason': 'content_filter: OTHER', 'index': 0}]})
        with self.assertLogs('wm_security.client', level='WARNING') as logs:
            with self.assertRaisesRegex(ProviderContentFilterError, 'content_filter: OTHER'):
                self.client().chat([{'role': 'user', 'content': 'probe'}])
        self.assertEqual(post.call_count, 1)
        self.assertIn('provider content filter; not retrying', logs.output[0])
        record = json.loads((self.root/'run/calls.jsonl').read_text())
        self.assertEqual(record['http_status'], 200)
        self.assertIn('Response blocked by provider', record['error'])

    @patch('wm_security.client.requests.post')
    def test_missing_message_has_a_descriptive_error(self, post):
        self.config.pop('fallback_url')
        self.write_config()
        post.return_value = Mock(status_code=200, json=lambda: {
            'choices': [{'finish_reason': 'stop'}]})
        with self.assertLogs('wm_security.client', level='WARNING'):
            with self.assertRaisesRegex(RuntimeError, 'no assistant message; finish_reason=stop'):
                self.client().chat([{'role': 'user', 'content': 'probe'}])

    @patch('wm_security.client.requests.post')
    def test_empty_choices_has_a_descriptive_error(self, post):
        self.config.pop('fallback_url')
        self.write_config()
        post.return_value = Mock(status_code=200, json=lambda: {'choices': []})
        with self.assertLogs('wm_security.client', level='WARNING'):
            with self.assertRaisesRegex(RuntimeError, 'no valid completion choice'):
                self.client().chat([{'role': 'user', 'content': 'probe'}])


if __name__ == '__main__':
    unittest.main()
