"""Stream this launch's Ray actor output despite driver forwarding delays."""
import os
from pathlib import Path
import sys
import time
root, pid, run = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
started = time.time() - 10
positions = {}
with (run / 'ray-live.log').open('a', buffering=1) as evidence:
    while True:
        for logs in root.glob('**/session_*/logs'):
            if logs.parent.name == 'session_latest' or logs.parent.stat().st_mtime < started:
                continue
            for path in logs.glob('worker-*'):
                if path.suffix not in ('.out', '.err'):
                    continue
                with path.open('rb') as f:
                    header = f.read(256)
                    if b':actor_name:OPDTaskRunner' not in header:
                        continue
                    f.seek(positions.get(path, 0))
                    raw = f.read()
                    positions[path] = f.tell()
                if raw:
                    text = raw.decode(errors='replace')
                    evidence.write(text)
                    print('[ray-live] '+text, end='', flush=True)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(2)
