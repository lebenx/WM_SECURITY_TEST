"""Verify the downloaded checkpoint inventory without importing model packages."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['le-wm','dino_wm','planet','tdmpc2','dreamerv3'])
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text())
    rows=[r for r in manifest['files'] if args.model is None or r['model']==args.model]
    problems=[]
    for row in rows:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):
            problems.append(row['path']+': invalid path')
            continue
        if not path.is_file():
            problems.append(row['path']+': missing')
            continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                h.update(chunk)
        if path.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:
            problems.append(row['path']+': checksum or size mismatch')
    print(json.dumps(dict(requested=len(rows),verified=len(rows)-len(problems),issues=problems),indent=2))
    raise SystemExit(bool(problems))


if __name__=='__main__':
    main()
