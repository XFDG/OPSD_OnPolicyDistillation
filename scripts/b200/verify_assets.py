"""Check migrated immutable inputs against Beijing SHA256 manifest."""
import hashlib
import json
from pathlib import Path
import sys
root, manifest = map(Path, sys.argv[1:])
items = json.loads(manifest.read_text())
for i, item in enumerate(items, 1):
    path = root / item['path']
    assert path.is_file() and path.stat().st_size == item['bytes'], path
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    assert h.hexdigest() == item['sha256'], path
    print(f'[assets {i}/{len(items)}] SHA256 PASS {item["path"]}', flush=True)
