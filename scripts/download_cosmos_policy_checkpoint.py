#!/usr/bin/env python3
"""Resume a pinned public official checkpoint in checked HTTP byte ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
REPO = 'nvidia/Cosmos-Policy-LIBERO-Predict2-2B'
REVISION = 'cb689ec0e3347c13667d70a78a3447388f5c3bb8'
NAME = 'Cosmos-Policy-LIBERO-Predict2-2B.pt'
SIZE = 3913017345
SHA = '8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2'
OUT = ROOT / 'models/CosmosPolicy' / NAME
CHUNK = 32 * 1024 * 1024


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prefix = OUT.with_suffix(OUT.suffix + '.partial')
    start = prefix.stat().st_size if prefix.exists() else 0
    parts = OUT.with_suffix(OUT.suffix + '.parts')
    parts.mkdir(exist_ok=True)
    bounds = [(lo, min(SIZE - 1, lo + CHUNK - 1)) for lo in range(start, SIZE, CHUNK)]
    begun = time.monotonic()
    def fetch(pair):
        lo, hi = pair
        dest = parts / f'{lo}-{hi}.part'
        if dest.exists() and dest.stat().st_size == hi - lo + 1:
            return pair
        temporary = dest.with_suffix('.tmp')
        for attempt in range(8):
            try:
                url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}?download=true&range_start={lo}'
                with requests.get(url, headers={'Range': f'bytes={lo}-{hi}'}, stream=True, timeout=(45, 120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {lo}-{hi}/{SIZE}':
                        raise ValueError('Server did not honor exact requested range')
                    with temporary.open('wb') as handle:
                        for block in response.iter_content(1024 * 1024):
                            handle.write(block)
                if temporary.stat().st_size != hi - lo + 1:
                    raise ValueError('Truncated range response')
                temporary.replace(dest)
                return pair
            except (requests.RequestException, ValueError) as error:
                print(json.dumps({'range': pair, 'retry': attempt + 1, 'error': type(error).__name__}), flush=True)
                if attempt == 7:
                    raise
        raise RuntimeError('Unreachable')
    missing = [pair for pair in bounds if not (parts / f'{pair[0]}-{pair[1]}.part').exists()]
    print(json.dumps({'missing_chunks': len(missing), 'total_chunks': len(bounds), 'prefix_bytes': start}), flush=True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(fetch, pair) for pair in missing]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            print(json.dumps({'recovered_chunks': index, 'missing_chunks': len(missing), 'elapsed_seconds': time.monotonic() - begun}), flush=True)
    h = hashlib.sha256()
    assembled = OUT.with_suffix(OUT.suffix + '.verified_tmp')
    sources = ([prefix] if start else []) + [parts / f'{lo}-{hi}.part' for lo, hi in bounds]
    with assembled.open('wb') as target:
        for source in sources:
            with source.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    target.write(block)
                    h.update(block)
    actual = h.hexdigest()
    if actual != SHA or assembled.stat().st_size != SIZE:
        raise ValueError(f'Complete checkpoint verification failed: {actual}')
    assembled.replace(OUT)
    manifest = {'repo': REPO, 'revision': REVISION, 'filename': NAME, 'bytes': SIZE, 'sha256': actual,
                'verified': True, 'download_method': 'Pinned public HTTPS URL; exact byte ranges and full SHA256',
                'elapsed_seconds_resume': time.monotonic() - begun, 'local_path': str(OUT)}
    (OUT.parent / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
