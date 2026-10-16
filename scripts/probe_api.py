from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import time
from wm_security.client import Client


def probe(model, log_dir, timeout=30):
    client = Client(model, log_dir=log_dir, timeout=timeout, max_calls=1)
    start = time.monotonic()
    try:
        result = client.json('Return only a JSON object with an integer field named answer.',
                             {'question':'What is 17 + 26?'}, tag='connectivity', max_tokens=512)
        answer = result.get('answer') if isinstance(result, dict) else result
        return {'requested_model': model, 'result': result,
                'correct': type(answer) is int and answer == 43,
                'latency_s': time.monotonic() - start, 'requests': client.calls}
    except Exception as exc:
        return {'requested_model': model, 'correct': False, 'error': str(exc),
                'latency_s': time.monotonic() - start, 'requests': client.calls}


def main():
    parser = argparse.ArgumentParser(description='Check the selected experiment API with one arithmetic request per model.')
    parser.add_argument('--models', nargs='+', default=['gemini-3.8-flash'])
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--log-dir', type=Path,
                        default=Path('runs') / ('api_probe_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    results = []
    with ThreadPoolExecutor(max_workers=min(3, len(args.models))) as pool:
        tasks = [pool.submit(probe, name, args.log_dir, args.timeout) for name in args.models]
        for task in as_completed(tasks):
            result = task.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    return 0 if all(result['correct'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
